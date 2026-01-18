from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def pearsonr_np(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return 0.0
    r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return r


def compute_cd_cdd_pcc(
    dists: List[float],
    label_strs: List[Optional[str]],
    n_muts: List[Optional[float]],
    group_ids: List[Optional[Any]],
) -> Dict[str, float]:
    """Compute CD / CDD / PCC from pairwise distances.

    - CD  : mean(distance)
    - CDD : mean(distance|pathogenic) - mean(distance|benign)
    - PCC : mean over groups of pearson(n_mut, distance)
    """
    d_arr = np.asarray(dists, dtype=np.float64)
    cd = float(np.mean(d_arr)) if d_arr.size else 0.0

    ben = [d for d, lab in zip(dists, label_strs) if lab == "benign"]
    pat = [d for d, lab in zip(dists, label_strs) if lab == "pathogenic"]
    mean_ben = float(np.mean(ben)) if ben else 0.0
    mean_pat = float(np.mean(pat)) if pat else 0.0
    cdd = float(mean_pat - mean_ben) if (ben and pat) else 0.0

    gid_to_n: Dict[Any, List[float]] = defaultdict(list)
    gid_to_d: Dict[Any, List[float]] = defaultdict(list)
    for d, n, gid in zip(dists, n_muts, group_ids):
        if gid is None or n is None or (isinstance(n, float) and np.isnan(n)):
            continue
        gid_to_n[gid].append(float(n))
        gid_to_d[gid].append(float(d))

    pccs: List[float] = []
    for gid, xs in gid_to_n.items():
        ys = gid_to_d[gid]
        pccs.append(pearsonr_np(xs, ys))
    pcc = float(np.mean(pccs)) if pccs else 0.0

    return {
        "cd": cd,
        "cdd": cdd,
        "mean_ben": mean_ben,
        "mean_pat": mean_pat,
        "pcc": pcc,
        "num_records": float(len(dists)),
        "num_groups_pcc": float(len(pccs)),
    }
