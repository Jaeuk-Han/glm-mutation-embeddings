#!/usr/bin/env python
"""Evaluate CD / CDD / PCC on a dev JSONL with dna2vec + dual-head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glm_dna2vec.backbone import DNA2VecBackbone
from glm_dna2vec.data import ClinVarPairsDataset, make_pairs_collate
from glm_dna2vec.dualhead import load_dualhead_checkpoint
from glm_dna2vec.metrics import compute_cd_cdd_pcc
from glm_dna2vec.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval dna2vec + dual-head (CD/CDD/PCC)")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--head_ckpt", type=str, required=True)

    p.add_argument("--backbone_id", type=str, default="roychowdhuryresearch/dna2vec")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_json", type=str, default=None)
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

    ds = ClinVarPairsDataset(args.data_path)

    backbone = DNA2VecBackbone(model_id=args.backbone_id, max_length=args.max_length)
    backbone.to(device)
    backbone.eval()

    collate_fn = make_pairs_collate(backbone.tokenizer, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    in_dim = backbone.hidden_size
    head, cfg = load_dualhead_checkpoint(args.head_ckpt, in_dim=in_dim, device=device, strict=False)

    dists: List[float] = []
    label_strs: List[Optional[str]] = []
    n_muts_list: List[Optional[float]] = []
    group_ids_list: List[Optional[Any]] = []

    total_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B = int(batch["batch_size"])

        base = backbone.forward_embeds(input_ids, attention_mask)  # [2B, H]
        ref_base = base[:B]
        var_base = base[B:]

        ref_z = head(ref_base)
        var_z = head(var_base)

        dist = 1.0 - F.cosine_similarity(ref_z, var_z, dim=-1)
        d_np = dist.detach().cpu().numpy().astype(np.float64).tolist()

        dists.extend([float(x) for x in d_np])
        label_strs.extend(batch.get("labels", [None] * B))
        group_ids_list.extend(batch.get("group_ids", [None] * B))

        nm = batch.get("n_muts")
        if isinstance(nm, torch.Tensor):
            n_muts_list.extend([float(x) for x in nm.detach().cpu().numpy().tolist()])
        else:
            n_muts_list.extend([None] * B)

        if step % max(1, total_batches // 10) == 0 or step == total_batches:
            print(f"[progress] {step}/{total_batches} batches")

    metrics = compute_cd_cdd_pcc(dists, label_strs, n_muts_list, group_ids_list)

    print("========== dna2vec + DualHead Metrics ==========")
    print(f"backbone : {args.backbone_id}")
    print(f"head_ckpt: {args.head_ckpt}")
    print(f"data     : {args.data_path}")
    print(f"CD  = {metrics['cd']:.6f}")
    print(f"CDD = {metrics['cdd']:.6f}  (mean_pat={metrics['mean_pat']:.6f}, mean_ben={metrics['mean_ben']:.6f})")
    print(f"PCC = {metrics['pcc']:.6f}  (groups={int(metrics['num_groups_pcc'])})")
    print("===============================================")

    out = {
        "backbone_id": args.backbone_id,
        "head_ckpt": args.head_ckpt,
        "data_path": args.data_path,
        **metrics,
        "ckpt_config": cfg,
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
