#!/usr/bin/env python
"""Convert ClinVar pairs JSONL from v3 schema to a v4 training-ready schema.

What this does
- Reads a JSONL file where each record contains a `labels` dict like:
    {"pathogenic":0/1, "benign":0/1, "uncertain":0/1}
- Aggregates a single `label_str` using a priority order (default:
    pathogenic > benign > uncertain > unknown)
- Keeps only the labels you want for training (default: benign,pathogenic)
- Optionally rewrites the `labels` dict into a clean one-hot form

Defaults match your original script:
  input : data/processed/clinvar_pairs_v3.jsonl
  output: data/processed/clinvar_pairs_v4.jsonl
  keep  : benign,pathogenic

Typical usage
  python -m data_prep.clinvar.to_v4 \
    --input data/processed/clinvar_pairs_v3.jsonl \
    --output data/processed/clinvar_pairs_v4.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


DEFAULT_IN = Path("data/processed/clinvar_pairs_v3.jsonl")
DEFAULT_OUT = Path("data/processed/clinvar_pairs_v4.jsonl")


def _parse_csv_list(s: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("list must not be empty")
    return tuple(parts)


def aggregate_label_from_labels_dict(labels: object, *, priority: Iterable[str]) -> str:
    """Aggregate a single label_str from a labels dict.

    Args:
        labels: expected to be a dict-like object with 0/1 values.
        priority: ordered list like ("pathogenic", "benign", "uncertain").

    Returns:
        One of the priority labels, or "unknown" if none match.
    """
    if not isinstance(labels, dict):
        return "unknown"

    for k in priority:
        try:
            if bool(labels.get(k, 0)):
                return k
        except Exception:
            # defensive: if values are weird types
            if labels.get(k) in (1, "1", True, "true", "True"):
                return k

    return "unknown"


def convert_v3_to_v4(
    *,
    in_path: Path,
    out_path: Path,
    keep: Tuple[str, ...] = ("benign", "pathogenic"),
    priority: Tuple[str, ...] = ("pathogenic", "benign", "uncertain"),
    rewrite_labels: bool = True,
    drop_if_missing_labels: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Run conversion and return stats."""

    if not in_path.exists():
        raise FileNotFoundError(f"input not found: {in_path}")

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    cnt_in = 0
    cnt_out = 0

    old_label_counter: Counter[str] = Counter()
    new_label_counter: Counter[str] = Counter()
    dropped_counter: Counter[str] = Counter()

    fout = None
    try:
        fin = in_path.open("r", encoding="utf-8")
        if not dry_run:
            fout = out_path.open("w", encoding="utf-8")

        with fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                cnt_in += 1
                d = json.loads(line)

                old_label = d.get("label_str", "unknown")
                old_label_counter[str(old_label)] += 1

                labels_dict = d.get("labels")
                if labels_dict is None and drop_if_missing_labels:
                    dropped_counter["missing_labels"] += 1
                    continue

                new_label = aggregate_label_from_labels_dict(labels_dict, priority=priority)
                new_label_counter[new_label] += 1

                if new_label not in keep:
                    dropped_counter[new_label] += 1
                    continue

                d["label_str"] = new_label

                if rewrite_labels:
                    # Clean one-hot. Unknowns are not expected here since we filtered.
                    clean = {k: (1 if k == new_label else 0) for k in set(priority) | set(keep)}
                    # Ensure common keys exist.
                    for k in ("pathogenic", "benign", "uncertain"):
                        clean.setdefault(k, 0)
                    # If we are not keeping uncertain, force it to 0 (matches your original intent)
                    if "uncertain" not in keep:
                        clean["uncertain"] = 0
                    d["labels"] = clean

                if not dry_run:
                    assert fout is not None
                    fout.write(json.dumps(d, ensure_ascii=False) + "\n")

                cnt_out += 1

    finally:
        if fout is not None:
            fout.close()

    stats: Dict[str, object] = {
        "input_records": cnt_in,
        "output_records": cnt_out,
        "old_label_str": dict(old_label_counter),
        "new_aggregated_label": dict(new_label_counter),
        "dropped": dict(dropped_counter),
        "input": str(in_path),
        "output": str(out_path),
        "keep": list(keep),
        "priority": list(priority),
        "rewrite_labels": rewrite_labels,
        "dry_run": dry_run,
    }
    return stats


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_IN, help="Input v3 JSONL path")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSONL path")
    p.add_argument(
        "--keep",
        type=_parse_csv_list,
        default=("benign", "pathogenic"),
        help="Comma-separated label_str values to keep (default: benign,pathogenic)",
    )
    p.add_argument(
        "--priority",
        type=_parse_csv_list,
        default=("pathogenic", "benign", "uncertain"),
        help="Comma-separated priority order for aggregation (default: pathogenic,benign,uncertain)",
    )
    p.add_argument(
        "--no-rewrite-labels",
        action="store_true",
        help="Do not rewrite labels dict to a clean one-hot form",
    )
    p.add_argument(
        "--drop-if-missing-labels",
        action="store_true",
        help="Drop rows that do not contain a labels dict (instead of treating as unknown)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute stats; do not write output",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    stats = convert_v3_to_v4(
        in_path=args.input,
        out_path=args.output,
        keep=tuple(args.keep),
        priority=tuple(args.priority),
        rewrite_labels=not args.no_rewrite_labels,
        drop_if_missing_labels=args.drop_if_missing_labels,
        dry_run=args.dry_run,
    )

    print(f"[stats] input records              : {stats['input_records']}")
    print(f"[stats] output records             : {stats['output_records']}")
    print(f"[stats] old label_str distribution : {stats['old_label_str']}")
    print(f"[stats] new aggregated distribution: {stats['new_aggregated_label']}")
    print(f"[stats] dropped counts             : {stats['dropped']}")
    if stats["dry_run"]:
        print("[done] dry-run (no file written)")
    else:
        print(f"[done] wrote: {stats['output']}")


if __name__ == "__main__":
    main()
