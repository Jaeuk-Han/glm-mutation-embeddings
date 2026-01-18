#!/usr/bin/env python
"""Train dna2vec + dual-head projection.

This is the *best-score* style training script you provided,
refactored to use src/glm_dna2vec/* modules.

Example:
  python scripts/train_dualhead.py \
    --train_path data/processed/clinvar_pairs_v3_train.jsonl \
    --out_dir runs/dna2vec_dualhead_best \
    --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from glm_dna2vec.backbone import DNA2VecBackbone
from glm_dna2vec.data import ClinVarPairsDataset, make_pairs_collate
from glm_dna2vec.dualhead import Dna2VecDualHead, save_dualhead_checkpoint
from glm_dna2vec.losses import dist_reg_loss, pcc_rank_loss, cd_scale_loss
from glm_dna2vec.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dna2vec + dual-head projection")

    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # backbone
    p.add_argument("--backbone_id", type=str, default="roychowdhuryresearch/dna2vec")
    p.add_argument("--max_length", type=int, default=512)

    # train
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")

    # loss
    p.add_argument("--max_n_mut", type=float, default=20.0)
    p.add_argument("--target_cd", type=float, default=0.13)
    p.add_argument("--lambda_pcc", type=float, default=0.02)
    p.add_argument("--lambda_cd_scale", type=float, default=0.25)
    p.add_argument("--pcc_margin", type=float, default=1e-3)

    # head dims (best config)
    p.add_argument("--cd_dim", type=int, default=768)
    p.add_argument("--pcc_dim", type=int, default=256)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) dataset
    ds = ClinVarPairsDataset(args.train_path)

    # 2) backbone + freeze
    backbone = DNA2VecBackbone(model_id=args.backbone_id, max_length=args.max_length)
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    in_dim = backbone.hidden_size
    print(f"[info] dna2vec hidden_size = {in_dim}")

    collate_fn = make_pairs_collate(backbone.tokenizer, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # 3) head
    head = Dna2VecDualHead(in_dim=in_dim, cd_dim=args.cd_dim, pcc_dim=args.pcc_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)

    use_fp16 = bool(args.fp16) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # 4) training
    for epoch in range(args.epochs):
        head.train()
        losses = []
        reg_losses = []
        pcc_losses = []
        scale_losses = []

        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)           # [2B, L]
            attention_mask = batch["attention_mask"].to(device)
            n_muts = batch["n_muts"].to(device)                 # [B]
            group_ids = batch["group_ids"]
            B = int(batch["batch_size"])

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                with torch.no_grad():
                    base_emb = backbone.forward_embeds(input_ids, attention_mask)  # [2B, H]

                ref_base = base_emb[:B]
                var_base = base_emb[B:]

                ref_z, ref_cd, ref_pcc = head(ref_base, return_branches=True)
                var_z, var_cd, var_pcc = head(var_base, return_branches=True)

                loss_reg = dist_reg_loss(ref_z, var_z, n_muts, max_n_mut=args.max_n_mut)
                loss_pcc = pcc_rank_loss(ref_pcc, var_pcc, n_muts, group_ids, margin_scale=args.pcc_margin)
                loss_scale = cd_scale_loss(ref_z, var_z, target_cd=args.target_cd)

                loss = loss_reg + args.lambda_pcc * loss_pcc + args.lambda_cd_scale * loss_scale

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.item()))
            reg_losses.append(float(loss_reg.item()))
            pcc_losses.append(float(loss_pcc.item()))
            scale_losses.append(float(loss_scale.item()))

            if step % max(1, len(loader) // 10) == 0:
                print(
                    f"[epoch {epoch+1}] step {step}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"(reg={loss_reg.item():.4f}, pcc={loss_pcc.item():.4f}, scale={loss_scale.item():.4f})"
                )

        print(
            f"[epoch {epoch+1}] mean loss={np.mean(losses):.4f} "
            f"(reg={np.mean(reg_losses):.4f}, pcc={np.mean(pcc_losses):.4f}, scale={np.mean(scale_losses):.4f})"
        )

        ckpt_path = out_dir / f"head_epoch{epoch+1}.pt"
        save_dualhead_checkpoint(
            str(ckpt_path),
            head,
            backbone_id=args.backbone_id,
            max_length=args.max_length,
            in_dim=in_dim,
        )
        print(f"[info] saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
