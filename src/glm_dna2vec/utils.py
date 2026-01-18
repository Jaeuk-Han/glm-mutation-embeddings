import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Best-effort reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize over last dimension."""
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
