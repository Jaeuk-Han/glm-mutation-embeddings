#!/usr/bin/env python
"""Create a competition submission csv.

Embeddings are generated using:
  - dna2vec backbone (mean pooled, L2 normalized)
  - optional dual-head projection (z)

Modes:
  - z      : use only dual-head z (best score)
  - concat : use [base || z] then L2 normalize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dna2vec.backbone import DNA2VecBackbone
from glm_dna2vec.data import TestSeqDataset, make_test_collate
from glm_dna2vec.dualhead import load_dualhead_checkpoint
from glm_dna2vec.utils import set_seed, l2_normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer embeddings for test.csv")

    p.add_argument("--test_path", type=str, required=True, help="test.csv path (ID, seq)")
    p.add_argument("--output_path", type=str, required=True, help="output submission csv")

    p.add_argument("--head_ckpt", type=str, required=True, help="dual-head checkpoint")
    p.add_argument("--backbone_id", type=str, default="roychowdhuryresearch/dna2vec")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--mode", type=str, default="z", choices=["z", "concat"])
    p.add_argument("--max_dim", type=int, default=2048, help="max submission dim")

    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] device = {device}")

    backbone = DNA2VecBackbone(model_id=args.backbone_id, max_length=args.max_length)
    backbone.to(device)
    backbone.eval()

    in_dim = backbone.hidden_size

    head, cfg = load_dualhead_checkpoint(args.head_ckpt, in_dim=in_dim, device=device, strict=False)
    head.to(device)
    head.eval()

    test_ds = TestSeqDataset(args.test_path)
    collate_fn = make_test_collate(backbone.tokenizer, args.max_length)
    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    all_ids: List[str] = []
    all_embeds: List[np.ndarray] = []

    total_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        base = backbone.forward_embeds(input_ids, attention_mask)  # [B, H]
        z = head(base)  # [B, D]

        if args.mode == "z":
            final = z
        else:
            concat = torch.cat([base, z], dim=-1)
            if concat.shape[1] > args.max_dim:
                concat = concat[:, : args.max_dim]
            final = l2_normalize(concat)

        all_ids.extend(batch["ID"])
        all_embeds.append(final.detach().cpu().numpy())

        if step % max(1, total_batches // 10) == 0 or step == total_batches:
            print(f"[progress] {step}/{total_batches} batches")

    embeds = np.concatenate(all_embeds, axis=0)
    dim = embeds.shape[1]
    print(f"[info] final embeddings shape: {embeds.shape}")

    cols = [f"emb_{i:04d}" for i in range(dim)]
    df = pd.DataFrame(embeds, columns=cols)
    df.insert(0, "ID", all_ids)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[done] saved submission to {out_path}")


if __name__ == "__main__":
    main()
