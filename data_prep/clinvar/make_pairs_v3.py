#!/usr/bin/env python
"""Build ClinVar pair dataset (v3 schema) from a ClinVar VCF and a reference FASTA.

This script is a refactor of your original `make_clinvar_pairs_v3.py`:
- Removes hard-coded paths/constants (argparse CLI).
- Uses shared FASTA loader: `data_prep.common.fasta.load_fasta_as_dict`.
- Keeps dataset logic the same:
  * Parse SNV + short indels from ClinVar VCF (PASS/. only by default)
  * Cluster nearby variants per chromosome (span-based)
  * For each cluster, cut a fixed window around the median position
  * Emit single-variant records and multi-variant combo records

Output JSONL record schema (v3) matches your previous format.

Typical usage
  python -m data_prep.clinvar.make_pairs_v3 \
    --ref_fasta data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa \
    --vcf_path data/vcf/clinvar.vcf \
    --out_path data/processed/clinvar_pairs_v3.jsonl

Notes
- The FASTA loader loads all sequences into memory. GRCh38 can be large.
- Indels longer than --max_ref_len/--max_alt_len are skipped.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from data_prep.common.fasta import load_fasta_as_dict


# ------------------------------
# Small helpers
# ------------------------------

def _open_text_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, mode="rt", encoding="utf-8", newline="")


def parse_info(info_str: str) -> Dict[str, object]:
    """Parse VCF INFO string 'A=B;C=D;FLAG' -> dict."""
    d: Dict[str, object] = {}
    for part in info_str.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        else:
            d[part] = True
    return d


def parse_clinvar_labels(clnsig_raw: str) -> Tuple[Dict[str, int], str]:
    """ClinVar CLNSIG string -> one-hot labels + original string."""
    clnsig = (clnsig_raw or "").lower()

    # Keep the original behavior (substring checks)
    pathogenic = any(s in clnsig for s in ["pathogenic", "likely_pathogenic"])
    benign = any(s in clnsig for s in ["benign", "likely_benign"])
    uncertain = ("uncertain" in clnsig) or ("conflicting" in clnsig)

    return (
        {
            "pathogenic": int(pathogenic),
            "benign": int(benign),
            "uncertain": int(uncertain),
        },
        clnsig_raw,
    )


def labels_to_str(labels: Dict[str, int]) -> str:
    """Convert one-hot-ish dict into a representative label_str."""
    if labels.get("pathogenic", 0) == 1 and labels.get("benign", 0) == 0:
        return "pathogenic"
    if labels.get("benign", 0) == 1 and labels.get("pathogenic", 0) == 0:
        return "benign"
    return "uncertain"


def aggregate_label_str(label_strs: Iterable[str]) -> str:
    """Aggregate multiple label_str values into one."""
    s = set(label_strs)
    if "pathogenic" in s and "benign" not in s:
        return "pathogenic"
    if "benign" in s and "pathogenic" not in s:
        return "benign"
    return "uncertain"


def levenshtein_distance(a: str, b: str) -> int:
    """Levenshtein distance with unit costs (small strings only)."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j

    for i in range(1, la + 1):
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[la][lb]


def get_window_by_center(seq: str, center_pos: int, window: int) -> Tuple[str, int]:
    """Slice a window of length `window` centered around 1-based center_pos."""
    half = window // 2
    seq_len = len(seq)

    start = max(1, center_pos - half)
    end = min(seq_len, center_pos + half - 1)

    # adjust length
    if end - start + 1 < window:
        diff = window - (end - start + 1)
        start = max(1, start - diff // 2)
        end = min(seq_len, start + window - 1)

    subseq = seq[start - 1 : end]
    if len(subseq) != window:
        raise ValueError(f"Failed to slice window={window} at center={center_pos} (start={start}, end={end})")
    return subseq, start


def variants_overlap(v1: Dict[str, object], v2: Dict[str, object]) -> bool:
    """Check overlap of two variants on reference using REF length."""
    s1 = int(v1["pos"])
    e1 = s1 + len(str(v1["ref_allele"])) - 1
    s2 = int(v2["pos"])
    e2 = s2 + len(str(v2["ref_allele"])) - 1
    return not (e1 < s2 or e2 < s1)


def apply_variants_to_window(ref_seq: str, win_start: int, variants: List[Dict[str, object]]) -> Optional[str]:
    """Apply multiple variants (SNV + short indels) to a reference window."""
    if not variants:
        return ref_seq

    vars_sorted = sorted(variants, key=lambda v: int(v["pos"]))

    out: List[str] = []
    cursor_genome = win_start
    cursor_ref_idx = 0
    ref_len = len(ref_seq)

    for v in vars_sorted:
        pos = int(v["pos"])
        ref = str(v["ref_allele"])  # upper-case already
        alt = str(v["alt_allele"])  # upper-case already
        ref_len_v = len(ref)

        if pos < cursor_genome:
            # overlapping or unsorted; caller should filter
            return None

        unchanged_len = pos - cursor_genome
        if unchanged_len > 0:
            out.append(ref_seq[cursor_ref_idx : cursor_ref_idx + unchanged_len])
            cursor_ref_idx += unchanged_len
            cursor_genome += unchanged_len

        # now cursor_genome == pos
        if cursor_ref_idx + ref_len_v > ref_len:
            return None

        ref_subseq = ref_seq[cursor_ref_idx : cursor_ref_idx + ref_len_v]
        if ref_subseq != ref:
            return None

        out.append(alt)
        cursor_ref_idx += ref_len_v
        cursor_genome += ref_len_v

    if cursor_ref_idx < ref_len:
        out.append(ref_seq[cursor_ref_idx:])

    return "".join(out)


# ------------------------------
# VCF parsing
# ------------------------------

def _match_chrom_name(vcf_chrom: str, fasta_keys: Iterable[str]) -> Optional[str]:
    """Best-effort chrom name matching between VCF and FASTA."""
    if vcf_chrom in fasta_keys:
        return vcf_chrom
    if f"chr{vcf_chrom}" in fasta_keys:
        return f"chr{vcf_chrom}"
    if vcf_chrom.startswith("chr") and vcf_chrom[3:] in fasta_keys:
        return vcf_chrom[3:]
    return None


def load_variants_from_vcf(
    vcf_path: str,
    chrom_seqs: Dict[str, str],
    *,
    pass_only: bool,
    max_ref_len: int,
    max_alt_len: int,
) -> Tuple[Dict[str, List[Dict[str, object]]], Counter]:
    """Parse SNV + short indels from a (possibly gzipped) ClinVar VCF."""
    variants_by_chrom: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    stats: Counter = Counter()

    fasta_keys = set(chrom_seqs.keys())

    with _open_text_maybe_gzip(vcf_path) as f_in:
        for line in tqdm(f_in, desc="parse VCF"):
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                stats["skip_bad_line"] += 1
                continue

            chrom, pos_str, vid, ref, alt_str, qual, filt, info = fields[:8]
            try:
                pos = int(pos_str)
            except ValueError:
                stats["skip_bad_pos"] += 1
                continue

            ref = ref.upper()
            alts = [a.upper() for a in alt_str.split(",")]

            if pass_only and filt not in ("PASS", "."):
                stats["skip_filter"] += 1
                continue

            if len(ref) > max_ref_len:
                stats["skip_ref_too_long"] += 1
                continue
            if any(len(a) > max_alt_len for a in alts):
                stats["skip_alt_too_long"] += 1
                continue

            chrom_key = _match_chrom_name(chrom, fasta_keys)
            if chrom_key is None:
                stats["skip_chrom_unmatched"] += 1
                continue

            info_dict = parse_info(info)
            clnsig_raw = str(info_dict.get("CLNSIG", ""))
            labels, clnsig_str = parse_clinvar_labels(clnsig_raw)
            label_str = labels_to_str(labels)

            for alt in alts:
                if len(alt) > max_alt_len:
                    stats["skip_alt_too_long"] += 1
                    continue

                v = {
                    "chrom": chrom_key,
                    "pos": pos,
                    "ref_allele": ref,
                    "alt_allele": alt,
                    "labels": labels,
                    "label_str": label_str,
                    "clnsig": clnsig_str,
                    "vcf_id": vid,
                    "variant_key": f"{pos}_{ref}>{alt}",
                }
                variants_by_chrom[chrom_key].append(v)
                stats["variants_kept"] += 1

    for chrom in variants_by_chrom:
        variants_by_chrom[chrom].sort(key=lambda x: int(x["pos"]))

    return variants_by_chrom, stats


def build_clusters(variants_by_chrom: Dict[str, List[Dict[str, object]]], max_cluster_span: int) -> List[Tuple[str, List[Dict[str, object]]]]:
    """Cluster variants by position span per chromosome."""
    clusters: List[Tuple[str, List[Dict[str, object]]]] = []

    for chrom, vars_list in variants_by_chrom.items():
        if not vars_list:
            continue

        current = [vars_list[0]]
        cluster_start_pos = int(vars_list[0]["pos"])

        for v in vars_list[1:]:
            if int(v["pos"]) - cluster_start_pos <= max_cluster_span:
                current.append(v)
            else:
                clusters.append((chrom, current))
                current = [v]
                cluster_start_pos = int(v["pos"])

        if current:
            clusters.append((chrom, current))

    return clusters


# ------------------------------
# Main dataset builder
# ------------------------------

def build_pairs_v3(
    *,
    chrom_seqs: Dict[str, str],
    variants_by_chrom: Dict[str, List[Dict[str, object]]],
    out_path: Path,
    window_size: int,
    margin: int,
    max_cluster_span: int,
    max_muts_per_combo: int,
    max_combos_per_cluster: int,
    max_clusters_total: Optional[int],
    seed: int,
    ensure_ascii: bool,
) -> Counter:
    """Write v3 records to out_path; returns stats."""

    stats: Counter = Counter()

    clusters = build_clusters(variants_by_chrom, max_cluster_span)
    stats["clusters_total"] = len(clusters)

    rnd = random.Random(seed)
    rnd.shuffle(clusters)

    if max_clusters_total is not None and len(clusters) > max_clusters_total:
        clusters = clusters[:max_clusters_total]
    stats["clusters_sampled"] = len(clusters)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for chrom, vars_list in tqdm(clusters, desc="build windows / variants"):
            seq = chrom_seqs[chrom]
            seq_len = len(seq)

            positions = sorted(int(v["pos"]) for v in vars_list)
            center_pos = positions[len(positions) // 2]

            if center_pos < margin or center_pos > seq_len - margin:
                stats["skip_center_near_ends"] += 1
                continue

            try:
                ref_seq, win_start = get_window_by_center(seq, center_pos, window_size)
            except Exception:
                stats["skip_window_slice"] += 1
                continue

            win_end = win_start + window_size - 1
            group_id = f"{chrom}:{win_start}-{win_end}"

            # Filter variants that fully fit into the window and match FASTA reference.
            valid_vars: List[Dict[str, object]] = []
            for v in vars_list:
                pos = int(v["pos"])
                ref = str(v["ref_allele"])
                ref_len_v = len(ref)

                if pos < win_start or (pos + ref_len_v - 1) > win_end:
                    stats["skip_variant_outside_window"] += 1
                    continue

                local_start_idx = pos - win_start
                local_end_idx = local_start_idx + ref_len_v

                if ref_seq[local_start_idx:local_end_idx] != ref:
                    stats["skip_ref_mismatch"] += 1
                    continue

                v_local = dict(v)
                v_local["local_start_idx"] = local_start_idx
                v_local["local_end_idx"] = local_end_idx
                valid_vars.append(v_local)

            if not valid_vars:
                stats["skip_empty_valid_vars"] += 1
                continue

            stats["clusters_used"] += 1

            # 1) Single-variant records
            for v in valid_vars:
                var_seq = apply_variants_to_window(ref_seq, win_start, [v])
                if var_seq is None:
                    stats["skip_apply_single"] += 1
                    continue

                n_mut = levenshtein_distance(str(v["ref_allele"]), str(v["alt_allele"]))

                rec_id = f"clinvar_single_{group_id}_{v['pos']}_{v['ref_allele']}>{v['alt_allele']}"

                out = {
                    "id": rec_id,
                    "source": "clinvar",
                    "chrom": chrom,
                    "pos": int(v["pos"]),
                    "ref_allele": str(v["ref_allele"]),
                    "alt_allele": str(v["alt_allele"]),
                    "ref_seq": ref_seq,
                    "var_seq": var_seq,
                    "alt_seq": var_seq,
                    "window_size": window_size,
                    "group_id": group_id,
                    "n_mut": n_mut,
                    "label_str": str(v["label_str"]),
                    "labels": v["labels"],
                    "variant_ids": [str(v["variant_key"])],
                    "is_combo": False,
                    "extra": {
                        "db": "clinvar",
                        "clinvar_clnsig": str(v.get("clnsig", "")),
                        "vcf_id": str(v.get("vcf_id", "")),
                        "n_variants": 1,
                    },
                }
                f_out.write(json.dumps(out, ensure_ascii=ensure_ascii) + "\n")
                stats["records_written"] += 1
                stats["single_written"] += 1

            # 2) Multi-variant combo records
            if len(valid_vars) >= 2:
                all_indices = list(range(len(valid_vars)))
                all_combos: List[Tuple[int, ...]] = []

                max_k = min(max_muts_per_combo, len(valid_vars))
                for k in range(2, max_k + 1):
                    all_combos.extend(combinations(all_indices, k))

                rnd.shuffle(all_combos)
                all_combos = all_combos[:max_combos_per_cluster]

                for idxs in all_combos:
                    used = [valid_vars[i] for i in idxs]

                    # skip overlapping REF regions
                    overlapped = False
                    for i in range(len(used)):
                        for j in range(i + 1, len(used)):
                            if variants_overlap(used[i], used[j]):
                                overlapped = True
                                break
                        if overlapped:
                            break
                    if overlapped:
                        stats["skip_combo_overlap"] += 1
                        continue

                    var_seq = apply_variants_to_window(ref_seq, win_start, used)
                    if var_seq is None:
                        stats["skip_apply_combo"] += 1
                        continue

                    n_mut = sum(
                        levenshtein_distance(str(vv["ref_allele"]), str(vv["alt_allele"]))
                        for vv in used
                    )

                    label_list = [str(vv["label_str"]) for vv in used]
                    combo_label_str = aggregate_label_str(label_list)
                    variant_keys = [str(vv["variant_key"]) for vv in used]

                    rec_id = f"clinvar_combo_{group_id}_" + "-".join(sorted(variant_keys))

                    out = {
                        "id": rec_id,
                        "source": "clinvar",
                        "chrom": chrom,
                        "pos": None,
                        "ref_allele": None,
                        "alt_allele": None,
                        "ref_seq": ref_seq,
                        "var_seq": var_seq,
                        "alt_seq": var_seq,
                        "window_size": window_size,
                        "group_id": group_id,
                        "n_mut": n_mut,
                        "label_str": combo_label_str,
                        "labels": {
                            "pathogenic": int(combo_label_str == "pathogenic"),
                            "benign": int(combo_label_str == "benign"),
                            "uncertain": int(combo_label_str == "uncertain"),
                        },
                        "variant_ids": variant_keys,
                        "is_combo": True,
                        "extra": {
                            "db": "clinvar",
                            "n_variants": len(used),
                        },
                    }
                    f_out.write(json.dumps(out, ensure_ascii=ensure_ascii) + "\n")
                    stats["records_written"] += 1
                    stats["combo_written"] += 1

    return stats


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ClinVar pairs JSONL (v3 schema) from VCF + FASTA")

    p.add_argument(
        "--ref_fasta",
        type=str,
        default="data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa",
        help="Reference FASTA path (can be .gz)",
    )
    p.add_argument(
        "--vcf_path",
        type=str,
        default="data/vcf/clinvar.vcf",
        help="ClinVar VCF path (can be .gz). Use an uncompressed file if you want faster IO.",
    )
    p.add_argument(
        "--out_path",
        type=str,
        default="data/processed/clinvar_pairs_v3.jsonl",
        help="Output JSONL path",
    )

    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument(
        "--margin",
        type=int,
        default=None,
        help="Skip windows too close to chrom ends. Default: window_size//2 + 10",
    )

    p.add_argument("--max_cluster_span", type=int, default=512, help="Max span (bp) within a cluster")
    p.add_argument("--max_muts_per_combo", type=int, default=3, help="Max variants per combo record")
    p.add_argument("--max_combos_per_cluster", type=int, default=16, help="Max combo records per cluster")

    p.add_argument("--max_ref_len", type=int, default=50, help="Skip variants with REF longer than this")
    p.add_argument("--max_alt_len", type=int, default=50, help="Skip variants with any ALT longer than this")

    p.add_argument(
        "--max_clusters_total",
        type=int,
        default=20000,
        help="Sample up to this many clusters total. Set to 0 or negative to keep all.",
    )

    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--pass_only",
        action="store_true",
        default=True,
        help="Use only FILTER=PASS or '.' records (default: on)",
    )
    p.add_argument(
        "--allow_non_pass",
        action="store_true",
        help="If set, do NOT filter by FILTER field.",
    )

    p.add_argument(
        "--ensure_ascii",
        action="store_true",
        default=False,
        help="Use json.dumps(..., ensure_ascii=True). Default is False (UTF-8 friendly).",
    )

    return p


def main() -> None:
    args = _build_argparser().parse_args()

    ref_fasta = args.ref_fasta
    vcf_path = args.vcf_path
    out_path = Path(args.out_path)

    window_size = int(args.window_size)
    margin = int(args.margin) if args.margin is not None else (window_size // 2 + 10)

    max_clusters_total = int(args.max_clusters_total)
    if max_clusters_total <= 0:
        max_clusters_total = None

    pass_only = bool(args.pass_only) and (not bool(args.allow_non_pass))

    print(f"[info] loading FASTA: {ref_fasta}")
    chrom_seqs = load_fasta_as_dict(ref_fasta)

    print(f"[info] parsing VCF: {vcf_path} (pass_only={pass_only})")
    variants_by_chrom, vcf_stats = load_variants_from_vcf(
        vcf_path,
        chrom_seqs,
        pass_only=pass_only,
        max_ref_len=int(args.max_ref_len),
        max_alt_len=int(args.max_alt_len),
    )

    # build + write
    build_stats = build_pairs_v3(
        chrom_seqs=chrom_seqs,
        variants_by_chrom=variants_by_chrom,
        out_path=out_path,
        window_size=window_size,
        margin=margin,
        max_cluster_span=int(args.max_cluster_span),
        max_muts_per_combo=int(args.max_muts_per_combo),
        max_combos_per_cluster=int(args.max_combos_per_cluster),
        max_clusters_total=max_clusters_total,
        seed=int(args.seed),
        ensure_ascii=bool(args.ensure_ascii),
    )

    print("\n[stats] VCF parse")
    for k, v in vcf_stats.most_common():
        print(f"  - {k}: {v}")

    print("\n[stats] build")
    for k, v in build_stats.most_common():
        print(f"  - {k}: {v}")

    print(f"\n[done] wrote: {out_path}")


if __name__ == "__main__":
    main()
