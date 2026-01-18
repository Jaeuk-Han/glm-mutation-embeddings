from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class ClinVarPairsDataset(Dataset):
    """JSONL dataset for (ref_seq, var_seq) pairs.

    Expected fields (a subset is fine):
      - ref_seq (str)
      - var_seq or alt_seq (str)
      - n_mut (int/float)  : number of mutations (supervision)
      - label_str (str)    : benign/pathogenic/uncertain (optional)
      - group_id (Any)     : group key for PCC ranking (optional)
    """

    def __init__(self, path: str | Path, max_samples: Optional[int] = None):
        self.path = str(path)
        self.records: List[Dict[str, Any]] = []

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)

                ref_seq = d.get("ref_seq")
                var_seq = d.get("var_seq", d.get("alt_seq"))
                n_mut = d.get("n_mut")
                if ref_seq is None or var_seq is None or n_mut is None:
                    continue

                label_str = d.get("label_str")
                if label_str not in ("benign", "pathogenic", "uncertain"):
                    label_str = None

                self.records.append(
                    {
                        "ref_seq": ref_seq,
                        "var_seq": var_seq,
                        "n_mut": float(n_mut),
                        "label_str": label_str,
                        "group_id": d.get("group_id"),
                    }
                )

                if max_samples is not None and len(self.records) >= max_samples:
                    break

        print(f"[Dataset] loaded {len(self.records)} records from {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


def make_pairs_collate(tokenizer, max_length: int):
    """Collate for pair datasets.

    Tokenizes [ref_seq ...] + [var_seq ...] in one call to reduce overhead.

    Returns:
      input_ids: [2B, L]
      attention_mask: [2B, L]
      n_muts: [B]
      labels: List[str|None]
      group_ids: List[Any]
      batch_size: B
    """

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ref_seqs = [b["ref_seq"] for b in batch]
        var_seqs = [b["var_seq"] for b in batch]
        n_muts = torch.tensor([b["n_mut"] for b in batch], dtype=torch.float)
        labels = [b.get("label_str") for b in batch]
        group_ids = [b.get("group_id") for b in batch]

        enc = tokenizer(
            ref_seqs + var_seqs,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            # best effort
            attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "n_muts": n_muts,
            "labels": labels,
            "group_ids": group_ids,
            "batch_size": len(ref_seqs),
        }

    return collate


class TestSeqDataset(Dataset):
    """test.csv (ID, seq) dataset for submission generation."""

    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path)
        if "ID" not in df.columns or "seq" not in df.columns:
            raise ValueError("test.csv must have columns: ID, seq")
        self.ids = df["ID"].tolist()
        self.seqs = df["seq"].tolist()
        print(f"[Dataset] loaded {len(self.ids)} records from {csv_path}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"ID": self.ids[idx], "seq": self.seqs[idx]}


def make_test_collate(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [b["ID"] for b in batch]
        seqs = [b["seq"] for b in batch]

        enc = tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            if getattr(tokenizer, "pad_token_id", None) is not None:
                attention_mask = (input_ids != tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids)

        return {"ID": ids, "input_ids": input_ids, "attention_mask": attention_mask}

    return collate
