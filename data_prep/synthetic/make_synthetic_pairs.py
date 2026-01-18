#!/usr/bin/env python
import json
import random
from pathlib import Path

from tqdm import tqdm

from data_prep.common.fasta import load_fasta_as_dict

# TODO: 네가 실제로 갖고 있는 파일 이름/확장자에 맞게 수정해줘!
# 예시1) 압축 풀었으면:  Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa
# 예시2) gzip 그대로라면: Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz  (+ fasta_utils에서 gzip 지원)
REF_FASTA = "data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"

OUT_PATH = "data/processed/synthetic_pairs.jsonl"

WINDOW_SIZE = 1024
MARGIN = WINDOW_SIZE // 2 + 10  # 양 끝에서 안전 거리
SYN_PER_CHROM = 2000            # 크로모좀당 생성할 synthetic 변이 수

BASES = ["A", "C", "G", "T"]


def pick_alt(ref: str) -> str:
    """ref와 다른 염기를 하나 랜덤으로 뽑기."""
    return random.choice([b for b in BASES if b != ref])


def get_window(seq: str, pos: int, window: int):
    """
    seq: 해당 크로모좀 전체 시퀀스 (0-based 인덱스)
    pos: 1-based 염기 위치 (VCF 스타일)
    """
    half = window // 2
    seq_len = len(seq)

    # 1차적으로 중앙 기준 window/2 씩 양쪽 확보
    start = max(1, pos - half)
    end = min(seq_len, pos + half - 1)

    # 앞으로/뒤로 당겨서 정확히 window 길이 맞추기
    if end - start + 1 < window:
        diff = window - (end - start + 1)
        start = max(1, start - diff // 2)
        end = min(seq_len, start + window - 1)

    subseq = seq[start - 1 : end]  # 파이썬 슬라이스는 0-based, end exclusive
    assert len(subseq) == window, (pos, start, end, len(subseq))

    center_idx = pos - start
    return subseq, start, center_idx

def count_mismatch(a: str, b: str) -> int:
    """
    ref_seq 와 alt_seq 의 mismatch 개수를 센다.
    (길이는 동일하다고 가정)
    """
    assert len(a) == len(b), "ref_seq / alt_seq length mismatch"
    return sum(1 for x, y in zip(a, b) if x != y)

def is_main_chrom(name: str) -> bool:
    """
    짧은 스캐폴드(KI..., GL..., chrUn...) 같은 거 제외하고
    메인 크로모좀(1~22, X, Y, MT)만 사용하기 위한 필터.
    """
    # Ensembl 스타일: "1", "2", ..., "22", "X", "Y", "MT"
    if name.isdigit():
        return True
    if name in ["X", "Y", "M", "MT"]:
        return True

    # NCBI 스타일: "chr1", "chrX", ...
    if name.startswith("chr"):
        core = name[3:]
        if core.isdigit():
            return True
        if core in ["X", "Y", "M", "MT"]:
            return True

    return False


def main():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print(f"[info] loading FASTA from {REF_FASTA}")
    chrom_seqs = load_fasta_as_dict(REF_FASTA)

    # 메인 크로모좀만 사용 (KI270xxx, GL000xxx, chrUn 등은 제외)
    chroms = [c for c in chrom_seqs.keys() if is_main_chrom(c)]
    print(f"[info] using chromosomes: {chroms}")

    random.seed(42)

    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for chrom in chroms:
            seq = chrom_seqs[chrom]
            seq_len = len(seq)
            print(f"[synthetic] {chrom}, len={seq_len}")

            # 🔴 너무 짧은 contig은 WINDOW_SIZE를 만들 수 없으니 스킵
            if seq_len <= 2 * MARGIN:
                print(
                    f"[skip] {chrom} is too short for "
                    f"WINDOW_SIZE={WINDOW_SIZE}, MARGIN={MARGIN}"
                )
                continue

            n_created = 0
            pbar = tqdm(total=SYN_PER_CHROM, desc=f"{chrom}")
            while n_created < SYN_PER_CHROM:
                # 1-based 위치에서 양 끝 MARGIN 만큼은 피해서 샘플링
                pos = random.randint(MARGIN, seq_len - MARGIN)

                ref_base = seq[pos - 1]
                if ref_base not in BASES:
                    # N, 소문자 등은 건너뛰기
                    continue

                alt_base = pick_alt(ref_base)

                ref_seq, win_start, center_idx = get_window(
                    seq, pos, WINDOW_SIZE
                )

                # sanity check
                assert ref_seq[center_idx] == ref_base

                alt_seq_list = list(ref_seq)
                alt_seq_list[center_idx] = alt_base
                alt_seq = "".join(alt_seq_list)

                # 🔹 metric 학습용 필드
                n_mut = count_mismatch(ref_seq, alt_seq)  # synthetic SNV라 보통 1
                group_id = f"{chrom}:{win_start}-{win_start + WINDOW_SIZE - 1}"

                rec = {
                    "id": f"syn_{chrom}_{pos}_{ref_base}>{alt_base}",
                    "source": "synthetic",
                    "chrom": chrom,
                    "pos": pos,
                    "ref_allele": ref_base,
                    "alt_allele": alt_base,
                    "ref_seq": ref_seq,
                    "alt_seq": alt_seq,
                    "var_seq": alt_seq,            # (옵션) alias, 나중에 코드에서 var_seq만 써도 됨
                    "window_size": WINDOW_SIZE,
                    "center_index": center_idx,

                    # 🔹 metric loss v1용 필드
                    "label_str": None,             # synthetic은 ClinSig 없음 → 나중에 필터링해서 빼면 됨
                    "n_mut": n_mut,
                    "group_id": group_id,

                    "labels": {
                        "pathogenic": None,
                        "benign": None,
                        "uncertain": None,
                    },
                    "extra": {"db": "synthetic"},
                }

                f_out.write(json.dumps(rec) + "\n")
                n_created += 1
                pbar.update(1)

            pbar.close()

    print(f"[done] wrote synthetic pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
