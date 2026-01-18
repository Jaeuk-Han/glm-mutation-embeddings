from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn

from .utils import l2_normalize


class Dna2VecDualHead(nn.Module):
    """Dual projection head used for the best score.

    - CD/CDD branch: larger MLP (heavier)
    - PCC branch: smaller MLP

    forward(x) -> z (L2-normalized)
    """

    def __init__(self, in_dim: int, cd_dim: int = 768, pcc_dim: int = 256):
        super().__init__()
        hidden_cd = in_dim * 2
        self.cd_head = nn.Sequential(
            nn.Linear(in_dim, hidden_cd),
            nn.LayerNorm(hidden_cd),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_cd, cd_dim),
        )

        hidden_pcc = in_dim
        self.pcc_head = nn.Sequential(
            nn.Linear(in_dim, hidden_pcc),
            nn.GELU(),
            nn.Linear(hidden_pcc, pcc_dim),
        )

        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        self.cd_dim = int(cd_dim)
        self.pcc_dim = int(pcc_dim)
        self.out_dim = self.cd_dim + self.pcc_dim

    def forward(self, x: torch.Tensor, return_branches: bool = False):
        h_cd = l2_normalize(self.cd_head(x))
        h_pcc = l2_normalize(self.pcc_head(x))

        z = torch.cat([self.alpha * h_cd, self.beta * h_pcc], dim=-1)
        z = l2_normalize(z)

        if return_branches:
            return z, h_cd, h_pcc
        return z


def save_dualhead_checkpoint(
    path: str,
    head: Dna2VecDualHead,
    *,
    backbone_id: str,
    max_length: int,
    in_dim: int,
) -> None:
    """Save checkpoint in the same format expected by eval/infer scripts."""
    torch.save(
        {
            "config": {
                "backbone_id": backbone_id,
                "max_length": int(max_length),
                "in_dim": int(in_dim),
                "cd_dim": int(head.cd_dim),
                "pcc_dim": int(head.pcc_dim),
                "out_dim": int(head.out_dim),
                "dual_head": True,
            },
            "state_dict": head.state_dict(),
        },
        path,
    )


def load_dualhead_checkpoint(
    path: str,
    *,
    in_dim: int,
    device: torch.device,
    strict: bool = False,
) -> Tuple[Dna2VecDualHead, Dict[str, Any]]:
    """Load checkpoint and return (head, config)."""
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("checkpoint format unexpected (need {'config', 'state_dict', ...})")

    cfg = ckpt.get("config", {})
    cd_dim = int(cfg.get("cd_dim", 768))
    pcc_dim = int(cfg.get("pcc_dim", 256))

    head = Dna2VecDualHead(in_dim=in_dim, cd_dim=cd_dim, pcc_dim=pcc_dim).to(device)
    missing, unexpected = head.load_state_dict(ckpt["state_dict"], strict=strict)
    if missing or unexpected:
        # Keep warning only; often ok when strict=False
        print(f"[warn] load_state_dict missing={missing}, unexpected={unexpected}")

    head.eval()
    return head, cfg
