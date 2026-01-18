#!/usr/bin/env python
import json
from pathlib import Path

from tqdm import tqdm

from data_prep.common.fasta import load_fasta_as_dict

# GRCh38 레퍼런스 FASTA (synthetic에서 쓴 거랑 동일하게 맞추기)
REF_FASTA = "data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"

# 🔴 압축 해제된 ClinVar VCF 경로 (여기만 실제 파일명에 맞춰주면 됨)
VCF_PATH = "data/vcf/clinvar.vcf"

OUT_PATH = "data/processed/vcf_pairs.jsonl"

WINDOW_SIZE = 1024
MARGIN = WINDOW_SIZE // 2 + 10  # 양 끝에서 여유 두기


def parse_info(info_str: str):
    """
    INFO 필드를 'A=B;C=D;FLAG' 형태에서 dict로 변환.
    """
    d = {}
    for part in info_str.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        else:
            d[part] = True
    return d


def parse_clinvar_labels(clnsig_raw: str):
    """
    ClinVar의 CLNSIG 문자열을 기반으로 (pathogenic/benign/uncertain) 라벨 생성.
    """
    clnsig = clnsig_raw.lower()

    pathogenic = any(s in clnsig for s in ["pathogenic", "likely_pathogenic"])
    benign = any(s in clnsig for s in ["benign", "likely_benign"])
    uncertain = "uncertain" in clnsig or "conflicting" in clnsig

    return {
        "pathogenic": int(pathogenic),
        "benign": int(benign),
        "uncertain": int(uncertain),
    }, clnsig_raw

def labels_to_str(labels: dict) -> str:
    """
    labels: {"pathogenic": 0/1, "benign": 0/1, "uncertain": 0/1}
    → 하나의 대표 label_str로 압축.
    우승 목표니까 규칙을 조금 보수적으로:
      - pathogenic=1 & benign=0 → "pathogenic"
      - benign=1 & pathogenic=0 → "benign"
      - 나머지는 전부 "uncertain"으로 몰아넣기
    """
    if labels.get("pathogenic", 0) == 1 and labels.get("benign", 0) == 0:
        return "pathogenic"
    if labels.get("benign", 0) == 1 and labels.get("pathogenic", 0) == 0:
        return "benign"
    return "uncertain"


def count_mismatch(a: str, b: str) -> int:
    """
    ref_seq / alt_seq 의 mismatch 개수.
    SNV만 쓰고 있어서 보통 1이지만, 확장성 위해 일반화.
    """
    assert len(a) == len(b), "ref_seq / alt_seq length mismatch"
    return sum(1 for x, y in zip(a, b) if x != y)


def get_window(seq: str, pos: int, window: int):
    """
    seq: 크로모좀 전체 시퀀스 (0-based 문자열)
    pos: 1-based 위치 (VCF 스타일)
    """
    half = window // 2
    seq_len = len(seq)

    start = max(1, pos - half)
    end = min(seq_len, pos + half - 1)

    if end - start + 1 < window:
        diff = window - (end - start + 1)
        start = max(1, start - diff // 2)
        end = min(seq_len, start + window - 1)

    subseq = seq[start - 1 : end]
    assert len(subseq) == window, (pos, start, end, len(subseq))
    center_idx = pos - start
    return subseq, start, center_idx


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print(f"[info] loading FASTA from {REF_FASTA}")
    chrom_seqs = load_fasta_as_dict(REF_FASTA)

    print(f"[info] reading ClinVar VCF from {VCF_PATH}")
    with open(VCF_PATH, "r", encoding="utf-8") as f_in, open(
        OUT_PATH, "w", encoding="utf-8"
    ) as f_out:
        pbar = tqdm(desc="clinvar")

        for line in f_in:
            # 헤더/코멘트는 스킵
            if line.startswith("#"):
                continue

            pbar.update(1)

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom, pos_str, vid, ref, alt_str, qual, filt, info = fields[:8]
            pos = int(pos_str)
            ref = ref.upper()
            alts = [a.upper() for a in alt_str.split(",")]

            # SNV만 사용 (indel, MNV 등은 스킵)
            if len(ref) != 1:
                continue
            if not all(len(a) == 1 for a in alts):
                continue

            # FILTER: PASS 또는 . 만 사용 (필요하면 조건 완화 가능)
            if filt not in ("PASS", "."):
                continue

            # chrom 이름 FASTA 키와 맞추기
            if chrom in chrom_seqs:
                chrom_key = chrom
            elif f"chr{chrom}" in chrom_seqs:
                chrom_key = f"chr{chrom}"
            elif chrom.startswith("chr") and chrom[3:] in chrom_seqs:
                chrom_key = chrom[3:]
            else:
                # FASTA에 없는 크로모좀 이름이면 스킵
                continue

            seq = chrom_seqs[chrom_key]
            seq_len = len(seq)

            # 양 끝에서 MARGIN 안쪽은 window 만들기 어려우니 스킵
            if pos < MARGIN or pos > seq_len - MARGIN:
                continue

            ref_seq, win_start, center_idx = get_window(seq, pos, WINDOW_SIZE)

            # FASTA 기준 ref mismatch면 좌표 어긋난 거라 스킵
            if ref_seq[center_idx] != ref:
                continue

            info_dict = parse_info(info)
            clnsig_raw = info_dict.get("CLNSIG", "")
            labels, clnsig = parse_clinvar_labels(clnsig_raw)
            label_str = labels_to_str(labels)

            for alt in alts:
                alt_seq_list = list(ref_seq)
                alt_seq_list[center_idx] = alt
                alt_seq = "".join(alt_seq_list)

                rec_id = f"clinvar_{chrom_key}_{pos}_{ref}>{alt}"

                # 🔹 metric 학습용 필드
                n_mut = count_mismatch(ref_seq, alt_seq)
                group_id = f"{chrom_key}:{win_start}-{win_start + WINDOW_SIZE - 1}"

                out = {
                    "id": rec_id,
                    "source": "clinvar",
                    "chrom": chrom_key,
                    "pos": pos,
                    "ref_allele": ref,
                    "alt_allele": alt,
                    "ref_seq": ref_seq,
                    "alt_seq": alt_seq,
                    "var_seq": alt_seq,           # (옵션) alias
                    "window_size": WINDOW_SIZE,
                    "center_index": center_idx,

                    # 🔹 metric loss v1용 필드
                    "label_str": label_str,       # "benign" / "uncertain" / "pathogenic"
                    "n_mut": n_mut,
                    "group_id": group_id,

                    "labels": labels,
                    "extra": {
                        "db": "clinvar",
                        "clinvar_clnsig": clnsig,
                        "vcf_id": vid,
                    },
                }
                f_out.write(json.dumps(out) + "\n")


        pbar.close()

    print(f"[done] wrote VCF pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
