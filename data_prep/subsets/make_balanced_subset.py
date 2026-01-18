#!/usr/bin/env python

"""Create balanced train/val subsets by sampling records by `source`.

Default behavior (same intent as the original script):
  - Keep **all** synthetic examples
  - Randomly sample N examples from clinvar
  - Shuffle and write JSONL

This script is intended for *data preparation* only.

Key features:
  - No hard-coded paths (argparse CLI)
  - Streaming + reservoir sampling (works for huge JSONL without loading everything)
  - Optional caps for synthetic/other sources
  - Deterministic sampling via seed

Examples
--------
Keep all synthetic, sample 200k clinvar for train and 20k for val:

  python -m data_prep.subsets.make_balanced_subset \
    --train_input data/processed/train.jsonl \
    --val_input data/processed/val.jsonl \
    --train_clinvar_n 200000 \
    --val_clinvar_n 20000 \
    --train_output data/processed/train_balanced_250k.jsonl \
    --val_output data/processed/val_balanced_25k.jsonl

Cap synthetic too (reservoir sample), and keep up to 10k records from other sources:

  python -m data_prep.subsets.make_balanced_subset \
    --train_synthetic_n 50000 --val_synthetic_n 5000 \
    --keep_other_sources --train_other_n 10000 --val_other_n 1000
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


try:
    from tqdm import tqdm  # type: ignore

    def _tqdm(it: Iterable[Any], **kwargs):
        return tqdm(it, **kwargs)

except Exception:  # pragma: no cover

    def _tqdm(it: Iterable[Any], **kwargs):
        return it


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield dict records from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def reservoir_step(sample: List[Dict[str, Any]], item: Dict[str, Any], k: int, seen: int, rng: random.Random) -> None:
    """Reservoir sampling update.

    Args:
        sample: current reservoir
        item: new item
        k: reservoir size
        seen: how many items of this stream have been seen so far (1-indexed)
        rng: random generator
    """
    if k <= 0:
        return
    if len(sample) < k:
        sample.append(item)
        return
    j = rng.randrange(seen)
    if j < k:
        sample[j] = item


@dataclass
class SourceSplit:
    synthetic: List[Dict[str, Any]]
    clinvar: List[Dict[str, Any]]
    other: List[Dict[str, Any]]
    n_syn_total: int
    n_clin_total: int
    n_other_total: int


def sample_by_source(
    path: str,
    *,
    source_field: str,
    synthetic_value: str,
    clinvar_value: str,
    synthetic_k: Optional[int],
    clinvar_k: int,
    keep_other_sources: bool,
    other_k: Optional[int],
    rng: random.Random,
) -> SourceSplit:
    """Stream a JSONL file and sample records by source."""
    syn: List[Dict[str, Any]] = []
    clin: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []

    n_syn_total = 0
    n_clin_total = 0
    n_other_total = 0

    # Use tqdm if available
    for rec in _tqdm(iter_jsonl(path), desc=f"load {path}"):
        src = rec.get(source_field)
        if src == synthetic_value:
            n_syn_total += 1
            if synthetic_k is None:
                syn.append(rec)
            else:
                reservoir_step(syn, rec, synthetic_k, n_syn_total, rng)
        elif src == clinvar_value:
            n_clin_total += 1
            reservoir_step(clin, rec, clinvar_k, n_clin_total, rng)
        else:
            n_other_total += 1
            if keep_other_sources:
                if other_k is None:
                    other.append(rec)
                else:
                    reservoir_step(other, rec, other_k, n_other_total, rng)

    return SourceSplit(
        synthetic=syn,
        clinvar=clin,
        other=other,
        n_syn_total=n_syn_total,
        n_clin_total=n_clin_total,
        n_other_total=n_other_total,
    )


def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in _tqdm(records, desc=f"write {out_path}"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def _nonneg_int_or_none(value: str) -> Optional[int]:
    """Parse an optional cap.

    Convention:
      -1 => None (no cap, keep all)
       0 => 0 (keep none)
      >0 => reservoir size
    """
    n = int(value)
    if n < -1:
        raise argparse.ArgumentTypeError("must be -1, 0, or a positive integer")
    if n == -1:
        return None
    return n


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--train_input", required=True, help="Path to train.jsonl")
    p.add_argument("--val_input", required=True, help="Path to val.jsonl")
    p.add_argument("--train_output", required=True, help="Output JSONL for balanced train")
    p.add_argument("--val_output", required=True, help="Output JSONL for balanced val")

    p.add_argument("--train_clinvar_n", type=_positive_int, required=True, help="# clinvar examples to sample for train")
    p.add_argument("--val_clinvar_n", type=_positive_int, required=True, help="# clinvar examples to sample for val")

    p.add_argument(
        "--train_synthetic_n",
        type=_nonneg_int_or_none,
        default=None,
        help="Cap synthetic examples for train (reservoir). Use -1 to keep all (default).",
    )
    p.add_argument(
        "--val_synthetic_n",
        type=_nonneg_int_or_none,
        default=None,
        help="Cap synthetic examples for val (reservoir). Use -1 to keep all (default).",
    )

    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    p.add_argument(
        "--val_seed",
        type=int,
        default=None,
        help="Optional separate seed for val (defaults to seed+1)",
    )

    p.add_argument("--source_field", default="source", help="Field name that stores source label")
    p.add_argument("--synthetic_value", default="synthetic", help="Value that denotes synthetic")
    p.add_argument("--clinvar_value", default="clinvar", help="Value that denotes clinvar")

    p.add_argument(
        "--keep_other_sources",
        action="store_true",
        help="Keep examples whose source is neither synthetic nor clinvar",
    )
    p.add_argument(
        "--train_other_n",
        type=_nonneg_int_or_none,
        default=None,
        help="Cap other-source examples for train (requires --keep_other_sources). -1 keeps all.",
    )
    p.add_argument(
        "--val_other_n",
        type=_nonneg_int_or_none,
        default=None,
        help="Cap other-source examples for val (requires --keep_other_sources). -1 keeps all.",
    )

    p.add_argument("--no_shuffle", action="store_true", help="Do not shuffle output")
    p.add_argument("--allow_smaller", action="store_true", help="Allow sampling less than requested if data is insufficient")
    p.add_argument("--dry_run", action="store_true", help="Print stats only; do not write files")
    return p


def _assert_enough(total: int, need: int, *, name: str, allow_smaller: bool) -> None:
    if total < need and not allow_smaller:
        raise ValueError(f"Requested {need} {name} examples, but only found {total}.")


def _make_output(
    split: SourceSplit,
    *,
    requested_syn_k: Optional[int],
    requested_clin_k: int,
    requested_other_k: Optional[int],
    keep_other: bool,
    allow_smaller: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Validate counts after streaming
    _assert_enough(split.n_clin_total, requested_clin_k, name="clinvar", allow_smaller=allow_smaller)
    if requested_syn_k is not None:
        _assert_enough(split.n_syn_total, requested_syn_k, name="synthetic", allow_smaller=allow_smaller)
    if keep_other and requested_other_k is not None:
        _assert_enough(split.n_other_total, requested_other_k, name="other", allow_smaller=allow_smaller)

    out = list(split.synthetic) + list(split.clinvar)
    if keep_other:
        out += list(split.other)

    stats = {
        "syn_total": split.n_syn_total,
        "clin_total": split.n_clin_total,
        "other_total": split.n_other_total,
        "syn_kept": len(split.synthetic),
        "clin_kept": len(split.clinvar),
        "other_kept": len(split.other) if keep_other else 0,
        "out_total": len(out),
    }
    return out, stats


def main() -> None:
    args = build_argparser().parse_args()

    if (args.train_other_n is not None or args.val_other_n is not None) and not args.keep_other_sources:
        raise ValueError("--train_other_n/--val_other_n require --keep_other_sources")

    train_rng = random.Random(args.seed)
    val_seed = args.val_seed if args.val_seed is not None else (args.seed + 1)
    val_rng = random.Random(val_seed)

    # Train
    train_split = sample_by_source(
        args.train_input,
        source_field=args.source_field,
        synthetic_value=args.synthetic_value,
        clinvar_value=args.clinvar_value,
        synthetic_k=args.train_synthetic_n,
        clinvar_k=args.train_clinvar_n,
        keep_other_sources=args.keep_other_sources,
        other_k=args.train_other_n,
        rng=train_rng,
    )

    train_out, train_stats = _make_output(
        train_split,
        requested_syn_k=args.train_synthetic_n,
        requested_clin_k=args.train_clinvar_n,
        requested_other_k=args.train_other_n,
        keep_other=args.keep_other_sources,
        allow_smaller=args.allow_smaller,
    )

    if not args.no_shuffle:
        train_rng.shuffle(train_out)

    print("[train]", train_stats)

    # Val
    val_split = sample_by_source(
        args.val_input,
        source_field=args.source_field,
        synthetic_value=args.synthetic_value,
        clinvar_value=args.clinvar_value,
        synthetic_k=args.val_synthetic_n,
        clinvar_k=args.val_clinvar_n,
        keep_other_sources=args.keep_other_sources,
        other_k=args.val_other_n,
        rng=val_rng,
    )

    val_out, val_stats = _make_output(
        val_split,
        requested_syn_k=args.val_synthetic_n,
        requested_clin_k=args.val_clinvar_n,
        requested_other_k=args.val_other_n,
        keep_other=args.keep_other_sources,
        allow_smaller=args.allow_smaller,
    )

    if not args.no_shuffle:
        val_rng.shuffle(val_out)

    print("[val]", val_stats)

    if args.dry_run:
        print("[dry-run] not writing outputs")
        return

    write_jsonl(train_out, args.train_output)
    write_jsonl(val_out, args.val_output)
    print("[done] wrote balanced train/val subsets")


if __name__ == "__main__":
    main()
