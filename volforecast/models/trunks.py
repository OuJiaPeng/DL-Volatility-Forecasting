"""Trunk registry — the E2 bake-off arms, all in one file, all sharing one contract.

Contract: ``trunk(x_feat (B,L,V), x_surf (B,S)) -> residual log-vol quantiles (B,H,Q)``,
ending in the zero-init QuantileHead so EVERY arm starts exactly at the HAR prior.
Arms differ from the chassis by their trunk alone — that's what makes the bake-off a
controlled comparison instead of a bowl-mix.

Registered trunks:
  * itransformer — the incumbent (channel-mixing + IV cross-attention), in
    residual_transformer.py
  * lstm         — recurrent trunk; the "attention must beat this" sequence arm
  * tcn          — dilated causal convolutions over the lookback; strong small-data bias
  * linear       — plain linear map on the flattened window; the DLinear-class control
(state-dependent HAR and xgboost are separate Forecaster arms, not torch trunks.)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import QuantileHead
from .residual_transformer import ResidualTransformer


class _SurfMix(nn.Module):
    """Shared surface-conditioning block for non-attention trunks: z + proj(surf)."""

    def __init__(self, n_surface: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(n_surface, emb_dim)

    def forward(self, z: torch.Tensor, x_surf: torch.Tensor) -> torch.Tensor:
        return z + self.proj(x_surf)


class LSTMTrunk(nn.Module):
    def __init__(self, n_vars, lookback, n_surface, n_horizons, n_quantiles,
                 emb_dim=32, n_layers=1, dropout=0.1, **_):
        super().__init__()
        self.rnn = nn.LSTM(n_vars, emb_dim, num_layers=n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.mix = _SurfMix(n_surface, emb_dim)
        self.head = QuantileHead(emb_dim, n_horizons, n_quantiles, dropout)

    def forward(self, x_feat, x_surf, x_intra=None):
        out, _ = self.rnn(x_feat)              # (B, L, D)
        z = self.mix(out[:, -1], x_surf)       # last hidden state, surface-conditioned
        return self.head(z)


class TCNTrunk(nn.Module):
    """Causal dilated Conv1d stack (receptive field ~ 2^n_layers * kernel)."""

    def __init__(self, n_vars, lookback, n_surface, n_horizons, n_quantiles,
                 emb_dim=32, n_layers=3, kernel=3, dropout=0.1, **_):
        super().__init__()
        layers, ch = [], n_vars
        for i in range(n_layers):
            d = 2**i
            layers += [
                nn.ConstantPad1d(((kernel - 1) * d, 0), 0.0),   # left-pad: causal
                nn.Conv1d(ch, emb_dim, kernel, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch = emb_dim
        self.net = nn.Sequential(*layers)
        self.mix = _SurfMix(n_surface, emb_dim)
        self.head = QuantileHead(emb_dim, n_horizons, n_quantiles, dropout)

    def forward(self, x_feat, x_surf, x_intra=None):
        h = self.net(x_feat.transpose(1, 2))   # (B, D, L)
        z = self.mix(h[:, :, -1], x_surf)      # last causal step
        return self.head(z)


class LinearTrunk(nn.Module):
    """Flattened-window linear control: if attention can't beat this, it isn't earning."""

    def __init__(self, n_vars, lookback, n_surface, n_horizons, n_quantiles,
                 emb_dim=32, dropout=0.1, **_):
        super().__init__()
        self.flat = nn.Sequential(nn.Flatten(), nn.Linear(lookback * n_vars, emb_dim))
        self.mix = _SurfMix(n_surface, emb_dim)
        self.head = QuantileHead(emb_dim, n_horizons, n_quantiles, dropout)

    def forward(self, x_feat, x_surf, x_intra=None):
        z = self.mix(self.flat(x_feat), x_surf)
        return self.head(z)


class IntradayPatchTransformer(nn.Module):
    """E3 input arm: variable tokens + PATCH TOKENS from the raw 5-min cube.

    x_intra (B, K*78, 3) — (r, |r|, r^2) channels — is patched by a strided Conv1d
    (patch = 26 bars = ~2h), giving ~3 tokens/day x K days. The r^2/|r| channels hand
    the encoder the nonlinearity a linear patch embed of raw returns cannot produce.
    Patch tokens join the variable tokens in one encoder (full self-attention), then
    the IV-surface cross-attention and zero-init quantile head as in the incumbent.
    """

    def __init__(self, n_vars, lookback, n_surface, n_horizons, n_quantiles,
                 emb_dim=32, n_heads=4, n_layers=2, dropout=0.1, patch_bars=26, **_):
        super().__init__()
        self.var_embed = nn.Linear(lookback, emb_dim)
        self.var_norm = nn.LayerNorm(emb_dim)
        self.patch = nn.Conv1d(3, emb_dim, kernel_size=patch_bars, stride=patch_bars)
        self.patch_norm = nn.LayerNorm(emb_dim)
        enc = nn.TransformerEncoderLayer(emb_dim, n_heads, emb_dim * 4, dropout,
                                         activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.surf_embed = nn.Linear(1, emb_dim)
        self.cross = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(emb_dim)
        self.head = QuantileHead(emb_dim, n_horizons, n_quantiles, dropout)

    def forward(self, x_feat, x_surf, x_intra=None):
        tok = self.var_norm(self.var_embed(x_feat.transpose(1, 2)))     # (B, V, D)
        if x_intra is not None and x_intra.shape[1] > 1:
            pt = self.patch(x_intra.transpose(1, 2)).transpose(1, 2)    # (B, P, D)
            tok = torch.cat([tok, self.patch_norm(pt)], dim=1)
        tok = self.encoder(tok)
        surf = self.surf_embed(x_surf.unsqueeze(-1))
        att, _ = self.cross(tok, surf, surf)
        tok = self.cross_norm(tok + att)
        return self.head(tok.mean(dim=1))


TRUNKS = {
    "itransformer": ResidualTransformer,
    "itransformer_intra": IntradayPatchTransformer,
    "lstm": LSTMTrunk,
    "tcn": TCNTrunk,
    "linear": LinearTrunk,
}


def build_trunk(name: str, *, n_vars, lookback, n_surface, n_horizons, n_quantiles,
                emb_dim=32, n_heads=4, n_layers=2, dropout=0.1) -> nn.Module:
    if name not in TRUNKS:
        raise ValueError(f"unknown trunk {name!r}; registered: {sorted(TRUNKS)}")
    return TRUNKS[name](n_vars=n_vars, lookback=lookback, n_surface=n_surface,
                        n_horizons=n_horizons, n_quantiles=n_quantiles, emb_dim=emb_dim,
                        n_heads=n_heads, n_layers=n_layers, dropout=dropout)
