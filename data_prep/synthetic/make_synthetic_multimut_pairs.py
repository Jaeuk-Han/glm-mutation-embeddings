#!/usr/bin/env python
import json
import random
from pathlib import Path

from tqdm import tqdm

from data_prep.common.fasta import load_fasta_as_dict

# ===== 설정 =====
REF_FASTA = "data/reference/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"

OUT_PATH = "data/processed/synthetic_multimut_pairs.jsonl"

WINDOW_SIZE = 1024
MARGIN = 1024   # 크로모좀 양 끝에서 이만큼은 안 씀
GROUPS_PER_CHROM = 8000   # 크로모좀당 ref 윈도우 그룹 수 (x MUT_LEVELS 개수만큼 var 생성)
MUT_LEVELS = [0, 1, 2, 3, 5, 8]  # 각 ref 윈도우에서 만들 변이 개수 후보
BASES = ["A", "C", "G", "T"]
SEED = 42


def is_main_chrom(name: str) -> bool:
    """
    짧은 스캐폴드(KI..., GL..., chrUn...) 같은 거 제외하고
    메인 크로모좀(1~22, X, Y, MT)만 사용.
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


def get_window(seq: str, center_pos_1based: int, window_size: int):
    """
    center_pos_1based 를 중심으로 길이 window_size 짜리 subseq를 잘라온다.
    - center_pos_1based: 크로모좀 상 1-based 위치
    반환: (subseq, window_start_1based, center_idx_in_window)
    """
    half = window_size // 2
    start = center_pos_1based - half
    end = start + window_size  # 1-based exclusive

    # 인덱스 범위 보정
    if start < 1:
        start = 1
        end = start + window_size
    if end > len(seq) + 1:
        end = len(seq) + 1
        start = end - window_size

    # python 슬라이스는 0-based / end-exclusive
    subseq = seq[start - 1 : end - 1]
    center_idx = center_pos_1based - start  # window 내 0-based

    assert len(subseq) == window_size, f"subseq length != {window_size}"
    assert 0 <= center_idx < window_size, "center_idx out of range"
    return subseq, start, center_idx


def pick_alt(base: str) -> str:
    """ref base와 다른 임의의 염기를 하나 고름."""
    base = base.upper()
    cand = [b for b in BASES if b != base]
    if not cand:
        return base
    return random.choice(cand)


def count_mismatch(a: str, b: str) -> int:
    """ref_seq vs var_seq mismatch 개수."""
    assert len(a) == len(b)
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    random.seed(SEED)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print(f"[info] loading FASTA from {REF_FASTA} ...")
    chrom_seqs = load_fasta_as_dict(REF_FASTA)
    print(f"[info] loaded chroms: {list(chrom_seqs.keys())[:10]} ...")

    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for chrom, seq in chrom_seqs.items():
            if not is_main_chrom(chrom):
                continue

            seq_len = len(seq)
            print(f"[chrom] {chrom} (len={seq_len})")

            n_groups = 0
            pbar = tqdm(total=GROUPS_PER_CHROM, desc=f"{chrom}")

            while n_groups < GROUPS_PER_CHROM:
                # 1-based 위치에서 양 끝 MARGIN 만큼은 피해서 샘플링
                pos = random.randint(MARGIN, seq_len - MARGIN)

                # 윈도우 잘라오기
                ref_seq, win_start, center_idx = get_window(
                    seq, pos, WINDOW_SIZE
                )

                # N 비율이 너무 많으면 건너뛰기
                if "N" in ref_seq:
                    continue

                group_id = f"{chrom}:{win_start}-{win_start + WINDOW_SIZE - 1}"

                # 한 ref_seq에 대해 MUT_LEVELS 만큼 여러 var 생성
                for k_idx, k in enumerate(MUT_LEVELS):
                    var_list = list(ref_seq)
                    mut_positions = []

                    if k > 0:
                        # 윈도우 내에서 k개 위치 선택 (중복 없이)
                        # 중앙 근처만 쓰고 싶으면 범위를 좀 줄여도 됨
                        candidate_positions = list(range(WINDOW_SIZE))
                        random.shuffle(candidate_positions)
                        for p in candidate_positions:
                            base = var_list[p]
                            if base not in BASES:
                                continue
                            alt = pick_alt(base)
                            var_list[p] = alt
                            mut_positions.append(p)
                            if len(mut_positions) >= k:
                                break

                    var_seq = "".join(var_list)
                    n_mut = count_mismatch(ref_seq, var_seq)

                    rec = {
                        "id": f"synv2_{chrom}_{pos}_k{k}_idx{k_idx}",
                        "source": "synthetic_v2",
                        "chrom": chrom,
                        "window_start": win_start,
                        "window_end": win_start + WINDOW_SIZE - 1,
                        "ref_seq": ref_seq,
                        "var_seq": var_seq,
                        "n_mut": n_mut,
                        "n_snv": n_mut,   # 지금은 SNV만 생성
                        "n_indel": 0,
                        "mut_positions": mut_positions,
                        "group_id": group_id,
                        "extra": {"db": "synthetic_v2"},
                    }

                    f_out.write(json.dumps(rec) + "\n")

                n_groups += 1
                pbar.update(1)

            pbar.close()

    print(f"[done] wrote synthetic multimut pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
