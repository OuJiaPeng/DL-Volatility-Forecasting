"""Quantile output head with monotonicity enforced by sorting."""
from __future__ import annotations

import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    """(B, D) -> (B, H, Q) with q-dim sorted so q10 <= q50 <= q90 always holds."""

    def __init__(self, emb_dim: int, n_horizons: int, n_quantiles: int, dropout: float = 0.1):
        super().__init__()
        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_horizons * n_quantiles),
        )
        # zero-init the output layer: the residual starts at exactly ZERO, i.e. the
        # hybrid begins as its classical prior (HAR) and must be pulled away from it
        # by real gradient signal. Standard residual-corrector trick; without it the
        # untrained model starts as prior + noise and must first learn its way back.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z).view(-1, self.n_horizons, self.n_quantiles)
        return torch.sort(out, dim=-1).values
