#!/usr/bin/env python
"""data_prep/synthetic/make_pseudo_test_pairs.py

Create a **pseudo-test** dataset of (ref_seq, var_seq, n_mut) pairs by sampling
windows from a reference genome FASTA (e.g., GRCh38) and injecting SNVs.

Output JSONL records look like:

    {
      "id": "pseudo_chr1:123-1147_n4",
      "source": "pseudo_test_synth",
      "chrom": "chr1",
      "start": 123,
      "end": 1147,
      "ref_seq": "...",
      "var_seq": "...",
      "n_mut": 4,
      "group_id": "pseudo_chr1:123-1147",
      "mutations": [{"pos": 10, "ref": "A", "alt": "G", "type": "snv"}, ...]
    }

Notes
-----
- Samples reference windows (group_id) and creates multiple mutation levels per window.
- Designed for local sanity checks / private metrics (n-gram, Hamming, PCC, etc.).
- Loads the whole FASTA into memory (GRCh38 can be large).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from data_prep.common.fasta import load_fasta_as_dict


DNA_BASES = ["A", "C", "G", "T"]


@dataclass
class ChromSeq:
    name: str
    seq: str


def _load_chroms(fasta_path: str, verbose: bool = False) -> List[ChromSeq]:
    """Load FASTA into a list of ChromSeq (uppercased). Supports .gz via common loader."""
    d = load_fasta_as_dict(fasta_path)
    chroms = [ChromSeq(name=k, seq=v.upper()) for k, v in d.items()]
    print(f"[FASTA] loaded {len(chroms)} records from {fasta_path}")
    if verbose:
        for c in chroms:
            print(f"    - {c.name}: len={len(c.seq)}")
    return chroms


def inject_snvs(ref_seq: str, n_mut: int, rng: random.Random) -> Tuple[str, List[Dict]]:
    """Inject `n_mut` SNVs into `ref_seq`.

    - If n_mut > number of valid A/C/G/T positions, samples with replacement.
    - Returns (var_seq, mutations). `n_mut` in output can be len(mutations).
    """
    if n_mut <= 0:
        return ref_seq, []

    seq_list = list(ref_seq)
    candidate_positions = [i for i, b in enumerate(seq_list) if b in DNA_BASES]
    if not candidate_positions:
        return ref_seq, []

    if n_mut <= len(candidate_positions):
        positions = rng.sample(candidate_positions, n_mut)
    else:
        positions = [rng.choice(candidate_positions) for _ in range(n_mut)]

    mutations: List[Dict] = []
    for pos in positions:
        ref_b = seq_list[pos]
        alt_b = rng.choice([b for b in DNA_BASES if b != ref_b])
        seq_list[pos] = alt_b
        mutations.append({"pos": pos, "ref": ref_b, "alt": alt_b, "type": "snv"})

    return "".join(seq_list), mutations


def sample_ref_windows(
    chroms: List[ChromSeq],
    window_size: int,
    num_groups: int,
    rng: random.Random,
    max_N_ratio: float = 0.1,
    allowed_chrom_prefixes: Tuple[str, ...] = ("chr",),
    max_trials_factor: int = 20,
) -> List[Tuple[str, int, int, str]]:
    """Sample reference windows.

    Returns tuples of (chrom, start, end, ref_seq) where start is 0-based and end is exclusive.
    """
    # Filter usable chroms
    usable: List[ChromSeq] = []
    for c in chroms:
        if not allowed_chrom_prefixes or any(c.name.startswith(p) for p in allowed_chrom_prefixes):
            usable.append(c)

    if not usable:
        raise RuntimeError("No usable chromosomes found for sampling.")

    lengths = [len(c.seq) for c in usable]
    total_len = sum(lengths)
    probs = [l / total_len for l in lengths]

    windows: List[Tuple[str, int, int, str]] = []
    trials = 0
    max_trials = max(1, num_groups) * max_trials_factor

    while len(windows) < num_groups and trials < max_trials:
        trials += 1
        chrom_idx = rng.choices(range(len(usable)), weights=probs, k=1)[0]
        chrom = usable[chrom_idx]
        if len(chrom.seq) < window_size:
            continue

        start = rng.randint(0, len(chrom.seq) - window_size)
        end = start + window_size
        ref_seq = chrom.seq[start:end]

        n_ratio = ref_seq.count("N") / window_size
        if n_ratio > max_N_ratio:
            continue

        windows.append((chrom.name, start, end, ref_seq))

    print(f"[windows] sampled {len(windows)} windows (requested={num_groups}, trials={trials})")
    return windows


def make_pseudo_pairs(
    fasta_path: str,
    output_path: str,
    window_size: int,
    num_groups: int,
    n_mut_list: List[int],
    seed: int = 42,
    max_N_ratio: float = 0.1,
    allowed_chrom_prefixes: Tuple[str, ...] = ("chr",),
    source: str = "pseudo_test_synth",
    id_prefix: str = "pseudo",
    verbose_fasta: bool = False,
) -> None:
    rng = random.Random(seed)

    chroms = _load_chroms(fasta_path, verbose=verbose_fasta)
    windows = sample_ref_windows(
        chroms=chroms,
        window_size=window_size,
        num_groups=num_groups,
        rng=rng,
        max_N_ratio=max_N_ratio,
        allowed_chrom_prefixes=allowed_chrom_prefixes,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for (chrom, start, end, ref_seq) in windows:
            group_id = f"{id_prefix}_{chrom}:{start}-{end}"

            for n_mut in n_mut_list:
                var_seq, mutations = inject_snvs(ref_seq, n_mut=n_mut, rng=rng)
                rec_id = f"{group_id}_n{n_mut}"
                rec = {
                    "id": rec_id,
                    "source": source,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "ref_seq": ref_seq,
                    "var_seq": var_seq,
                    "n_mut": len(mutations),
                    "group_id": group_id,
                    "mutations": mutations,
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                record_count += 1

    print(f"[done] wrote {record_count} records to {out_path}")


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _parse_csv_strs(s: str) -> Tuple[str, ...]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return tuple(items)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create pseudo-test (ref, var, n_mut) pairs from a FASTA reference.")
    p.add_argument("--fasta_path", type=str, required=True, help="Path to reference FASTA (.fa/.fasta, optionally .gz)")
    p.add_argument("--output_path", type=str, required=True, help="Output JSONL path (e.g., data/processed/pseudo_test_pairs.jsonl)")
    p.add_argument("--window_size", type=int, default=1024, help="Window size for each ref sequence")
    p.add_argument("--num_groups", type=int, default=10_000, help="Number of reference windows (groups) to sample")
    p.add_argument("--n_mut_list", type=str, default="0,1,2,4,8", help="Comma-separated mutation counts per group")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--max_N_ratio", type=float, default=0.1, help="Max allowed ratio of 'N' bases per window")
    p.add_argument(
        "--allowed_chrom_prefixes",
        type=str,
        default="chr",
        help="Comma-separated prefixes to include (default: 'chr'). Use empty string to include all.",
    )
    p.add_argument("--source", type=str, default="pseudo_test_synth", help="Record 'source' field")
    p.add_argument("--id_prefix", type=str, default="pseudo", help="Prefix for group_id (default: 'pseudo')")
    p.add_argument("--verbose_fasta", action="store_true", help="Print per-record FASTA lengths")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_mut_list = _parse_csv_ints(args.n_mut_list)
    allowed = _parse_csv_strs(args.allowed_chrom_prefixes) if args.allowed_chrom_prefixes else tuple()

    print(f"[config] window_size={args.window_size}, num_groups={args.num_groups}, seed={args.seed}")
    print(f"[config] n_mut_list={n_mut_list}, max_N_ratio={args.max_N_ratio}")
    print(f"[config] allowed_chrom_prefixes={allowed if allowed else '(all)'}")

    make_pseudo_pairs(
        fasta_path=args.fasta_path,
        output_path=args.output_path,
        window_size=args.window_size,
        num_groups=args.num_groups,
        n_mut_list=n_mut_list,
        seed=args.seed,
        max_N_ratio=args.max_N_ratio,
        allowed_chrom_prefixes=allowed,
        source=args.source,
        id_prefix=args.id_prefix,
        verbose_fasta=args.verbose_fasta,
    )


if __name__ == "__main__":
    main()
