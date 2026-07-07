"""Torch windowing OVER the panel — never re-aligns anything.

Each sample is keyed by an origin ``t0`` that is a panel row: the feature window is the
L panel rows ending AT ``t0`` (all causal by panel construction), the surface vector is
the row at ``t0``, and the label is that same row's ``tgt_*`` columns. Origins without
L rows of history are dropped (reported via ``skipped``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .models.prior import VOL_FLOOR

SURFACE_COLS = ["feat_atm_iv", "feat_skew", "feat_term_slope", "feat_vix"]
TENOR_SURFACE_COLS = SURFACE_COLS + [f"feat_atm_iv_{h}" for h in (1, 5, 10, 21)]

# fixed, leakage-free channel scales for the raw intraday cube (r, |r|, r^2):
# deterministic constants rather than fitted stats, so no split can leak through them
INTRA_SCALE = (1e3, 1e3, 1e6)
BARS_PER_DAY = 78  # 5-min RTH grid


def build_intraday_cube(bars, cal, panel_index) -> np.ndarray:
    """(n_sessions, BARS_PER_DAY, 3) raw intraday tensor aligned row-for-row with the panel.

    Channels: (r, |r|, r^2) per 5-min bar, fixed-scaled. This is the E3 input arm —
    the information HAR's daily aggregates (and our 4 summary features) compress away.
    """
    from .features.realized import daily_realized_variance  # noqa: F401 (doc anchor)

    cube = np.zeros((len(panel_index), BARS_PER_DAY, 3), dtype=np.float32)
    close_to_day = {cal.session_close(d): d for d in
                    pd.Index(bars.index.normalize().unique())}
    for i, t0 in enumerate(panel_index):
        day = close_to_day.get(pd.Timestamp(t0))
        if day is None:
            continue
        px = bars["close"].reindex(cal.rv_grid(day)).dropna()
        if len(px) < 3:
            continue
        r = np.diff(np.log(px.values)).astype(np.float32)
        r = r[:BARS_PER_DAY]
        cube[i, : len(r), 0] = r * INTRA_SCALE[0]
        cube[i, : len(r), 1] = np.abs(r) * INTRA_SCALE[1]
        cube[i, : len(r), 2] = (r**2) * INTRA_SCALE[2]
    return cube


class PanelWindowDataset(Dataset):
    def __init__(
        self,
        panel: pd.DataFrame,
        origins: pd.DatetimeIndex,
        split,
        lookback: int,
        horizons,
        prior_log: np.ndarray,
        surface_cols=None,
        cube: np.ndarray | None = None,
        intra_days: int = 5,
    ):
        self.lookback = lookback
        self.intra_days = intra_days
        self._cube = cube  # (n, BARS_PER_DAY, 3) aligned with panel rows, or None
        feat_norm = split.normalize(panel)
        self._X = feat_norm.values.astype(np.float32)                       # (n, V)
        surf_cols = [c for c in (surface_cols or SURFACE_COLS) if c in feat_norm.columns]
        self._S = feat_norm[surf_cols].values.astype(np.float32)            # (n, S)
        tgt_cols = [f"tgt_rv_{h}" for h in horizons]
        self._Y = np.log(np.clip(panel[tgt_cols].values, VOL_FLOOR, None)).astype(np.float32)

        pos_of = {t: i for i, t in enumerate(panel.index)}
        self.positions, kept, keep_mask = [], [], []
        for t in origins:
            p = pos_of[t]
            ok = p >= lookback - 1
            keep_mask.append(ok)
            if ok:
                self.positions.append(p)
                kept.append(t)
        self.origins = pd.DatetimeIndex(kept)
        self.skipped = pd.DatetimeIndex([t for t, ok in zip(origins, keep_mask) if not ok])

        prior_all = np.asarray(prior_log, dtype=np.float32)
        if len(prior_all) != len(origins):
            raise ValueError("prior_log must be aligned with `origins` row-for-row")
        self._prior = prior_all[np.asarray(keep_mask)]

        self.n_vars = self._X.shape[1]
        self.n_surface = self._S.shape[1]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        p = self.positions[i]
        x = torch.from_numpy(self._X[p - self.lookback + 1 : p + 1])  # (L, V)
        s = torch.from_numpy(self._S[p])                              # (S,)
        if self._cube is not None:
            intra = torch.from_numpy(
                self._cube[p - self.intra_days + 1 : p + 1].reshape(-1, 3)
            )                                                          # (K*78, 3)
        else:
            intra = torch.zeros(1, 3)                                  # dummy, ignored
        prior = torch.from_numpy(self._prior[i])                      # (H,)
        y = torch.from_numpy(self._Y[p])                              # (H,)
        return x, s, intra, prior, y
