#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL path containing a 'group_id' field per record.",
    )
    p.add_argument(
        "--out_dir",
        default=Path("data/processed"),
        type=Path,
        help="Output directory (default: data/processed).",
    )
    p.add_argument(
        "--prefix",
        default=None,
        type=str,
        help=(
            "Output filename prefix. If omitted, uses input file stem (e.g., 'clinvar_pairs_v4')."
        ),
    )
    p.add_argument(
        "--ratios",
        default=(0.8, 0.1, 0.1),
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "DEV"),
        help="Split ratios for train/val/dev (default: 0.8 0.1 0.1).",
    )
    p.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for group_id shuffling (default: 42).",
    )
    p.add_argument(
        "--dev_suffix",
        default="dev_eval",
        type=str,
        help="Suffix for the third split file (default: dev_eval).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any record is missing 'group_id' (default: drop those records).",
    )
    return p.parse_args()


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at {path}:{i}: {e}") from e


def _validate_ratios(ratios: Tuple[float, float, float]) -> None:
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/dev ratios must sum to 1.0, got {ratios} (sum={total})")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")


def _split_groups(group_ids: list[str], ratios: Tuple[float, float, float], seed: int) -> Tuple[set[str], set[str], set[str]]:
    random.seed(seed)
    random.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # remainder -> dev
    train = set(group_ids[:n_train])
    val = set(group_ids[n_train : n_train + n_val])
    dev = set(group_ids[n_train + n_val :])
    return train, val, dev


def main() -> None:
    args = _parse_args()
    in_path: Path = args.input
    out_dir: Path = args.out_dir
    ratios = tuple(args.ratios)
    seed: int = args.seed
    dev_suffix: str = args.dev_suffix
    strict: bool = args.strict

    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    _validate_ratios(ratios)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or in_path.stem

    train_path = out_dir / f"{prefix}_train.jsonl"
    val_path = out_dir / f"{prefix}_val.jsonl"
    dev_path = out_dir / f"{prefix}_{dev_suffix}.jsonl"

    # 1) collect unique group_ids
    group_set: set[str] = set()
    missing_gid = 0
    total = 0

    print(f"[info] collecting group_ids from: {in_path}")
    for d in _iter_jsonl(in_path):
        total += 1
        gid = d.get("group_id")
        if gid is None:
            missing_gid += 1
            if strict:
                raise KeyError(f"missing group_id (strict mode) in record id={d.get('id')} from {in_path}")
            continue
        group_set.add(str(gid))

    group_ids = sorted(group_set)  # sort for determinism before shuffle
    print(f"[info] total records scanned: {total}")
    print(f"[info] unique group_id: {len(group_ids)}")
    if missing_gid:
        print(f"[warn] records missing group_id: {missing_gid} (dropped)" + ("" if not strict else ""))

    # 2) split group_ids
    train_groups, val_groups, dev_groups = _split_groups(group_ids, ratios, seed)
    print(f"[info] train groups: {len(train_groups)}")
    print(f"[info] val groups  : {len(val_groups)}")
    print(f"[info] dev groups  : {len(dev_groups)}")

    # 3) stream again & write
    n_train = n_val = n_dev = n_drop = 0

    print(f"[info] writing -> {train_path}")
    print(f"[info] writing -> {val_path}")
    print(f"[info] writing -> {dev_path}")

    with (
        in_path.open("r", encoding="utf-8") as fin,
        train_path.open("w", encoding="utf-8") as ftr,
        val_path.open("w", encoding="utf-8") as fva,
        dev_path.open("w", encoding="utf-8") as fdv,
    ):
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            gid = d.get("group_id")
            if gid is None:
                if strict:
                    raise KeyError(f"missing group_id (strict mode) at line {line_no}")
                n_drop += 1
                continue
            gid = str(gid)

            if gid in train_groups:
                ftr.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_train += 1
            elif gid in val_groups:
                fva.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_val += 1
            elif gid in dev_groups:
                fdv.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_dev += 1
            else:
                # should not happen
                n_drop += 1

    print(f"[done] train records: {n_train}")
    print(f"[done] val records  : {n_val}")
    print(f"[done] dev records  : {n_dev}")
    if n_drop:
        print(f"[done] dropped records: {n_drop}")


if __name__ == "__main__":
    main()
