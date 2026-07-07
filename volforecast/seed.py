"""Deterministic seeding across python / numpy / torch (torch optional)."""
from __future__ import annotations

import os
import random
import numpy as np


def set_seed(seed: int = 7) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
