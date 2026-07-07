"""Residual transformer: iTransformer-style channel mixing + IV-surface cross-attention.

Design (why this and not plug-and-play PatchTST): the thesis is that the signal lives
ACROSS series (RV <-> IV interaction), which channel-independent PatchTST structurally
cannot model. So each *variable's* lookback series becomes one token (iTransformer) and
self-attention mixes variables; a cross-attention block then lets those tokens query the
IV-surface state at the decision time. Output is a (H, Q) grid of RESIDUAL log-vol
quantiles, added to the classical prior by the wrapper in ``hybrid.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import QuantileHead


class ResidualTransformer(nn.Module):
    def __init__(
        self,
        n_vars: int,
        lookback: int,
        n_surface: int,
        n_horizons: int,
        n_quantiles: int,
        emb_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.var_embed = nn.Linear(lookback, emb_dim)   # one token per variable
        self.var_norm = nn.LayerNorm(emb_dim)
        enc = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.surf_embed = nn.Linear(1, emb_dim)          # one token per surface bucket
        self.cross = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(emb_dim)
        self.head = QuantileHead(emb_dim, n_horizons, n_quantiles, dropout)

    def forward(self, x_feat: torch.Tensor, x_surf: torch.Tensor,
                x_intra: torch.Tensor | None = None) -> torch.Tensor:
        """x_feat (B, L, V); x_surf (B, S). Returns residual log-vol quantiles (B, H, Q).

        ``x_intra`` is accepted for interface compatibility and ignored — the raw
        intraday pathway belongs to the E3 patch trunk.
        """
        tok = self.var_norm(self.var_embed(x_feat.transpose(1, 2)))  # (B, V, D)
        tok = self.encoder(tok)                                      # channel mixing
        surf = self.surf_embed(x_surf.unsqueeze(-1))                 # (B, S, D)
        att, _ = self.cross(tok, surf, surf)                         # query vars, kv surface
        tok = self.cross_norm(tok + att)
        z = tok.mean(dim=1)                                          # (B, D)
        return self.head(z)
