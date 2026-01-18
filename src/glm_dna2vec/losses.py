from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def dist_reg_loss(
    ref_emb: torch.Tensor,
    var_emb: torch.Tensor,
    n_muts: torch.Tensor,
    *,
    max_n_mut: float = 20.0,
) -> torch.Tensor:
    """Distance regression for n_mut.

    d = 1 - cos(ref, var) ~ n_mut / max_n_mut
    """
    dists = 1.0 - F.cosine_similarity(ref_emb, var_emb, dim=-1)
    targets = torch.clamp(n_muts / float(max_n_mut), max=1.0)
    return F.mse_loss(dists, targets)


def cd_scale_loss(
    ref_emb: torch.Tensor,
    var_emb: torch.Tensor,
    *,
    target_cd: float = 0.13,
) -> torch.Tensor:
    """Keep the mean distance close to target_cd (CD anchoring)."""
    dists = 1.0 - F.cosine_similarity(ref_emb, var_emb, dim=-1)
    mean_d = dists.mean()
    return (mean_d - float(target_cd)) ** 2


def pcc_rank_loss(
    ref_emb: torch.Tensor,
    var_emb: torch.Tensor,
    n_muts: torch.Tensor,
    group_ids: List[Any],
    *,
    margin_scale: float = 1e-3,
) -> torch.Tensor:
    """Ranking loss for PCC.

    Within each group_id, encourage larger n_mut -> larger distance.
    Uses adjacent pairs in the n_mut sorted order for lower complexity.
    """
    device = ref_emb.device
    dists = 1.0 - F.cosine_similarity(ref_emb, var_emb, dim=-1)

    group_to_indices: Dict[Any, List[int]] = {}
    for i, g in enumerate(group_ids):
        if g is None:
            continue
        group_to_indices.setdefault(g, []).append(i)

    loss_terms: List[torch.Tensor] = []

    for idx_list in group_to_indices.values():
        if len(idx_list) < 2:
            continue
        idx_tensor = torch.tensor(idx_list, device=device, dtype=torch.long)

        nm = n_muts[idx_tensor]
        ds = dists[idx_tensor]

        order = torch.argsort(nm)
        nm_sorted = nm[order]
        ds_sorted = ds[order]

        diff_nm = nm_sorted[1:] - nm_sorted[:-1]
        diff_ds = ds_sorted[1:] - ds_sorted[:-1]

        valid = diff_nm > 0
        if valid.any():
            margin = float(margin_scale) * diff_nm[valid]
            loss_vec = F.relu(margin - diff_ds[valid])
            if loss_vec.numel() > 0:
                loss_terms.append(loss_vec)

    if not loss_terms:
        return torch.tensor(0.0, device=device)

    return torch.cat(loss_terms).mean()
