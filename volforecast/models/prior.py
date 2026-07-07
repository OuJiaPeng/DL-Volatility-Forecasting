"""Classical prior for the hybrid model: HAR-RV in log-vol space.

The prior is fit on TRAIN origins only and produces a causal per-origin forecast for
every panel row. The transformer learns only the residual around it, so the hybrid
starts at the strong classical baseline by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.har_rv import HARRV

VOL_FLOOR = 1e-6


class _ConstPrior:
    """Train-mean vol per horizon — the 'no real prior' falsification arm."""

    def __init__(self, horizons):
        self.horizons = list(horizons)

    def fit(self, panel, train_idx):
        self._mu = np.array([panel.loc[train_idx, f"tgt_rv_{h}"].mean()
                             for h in self.horizons])
        return self

    def predict(self, panel, origins):
        return np.tile(self._mu, (len(origins), 1))


class HARPrior:
    """Classical prior wrapper; ``kind`` selects the nesting floor: 'har' or 'har_iv'."""

    def __init__(self, horizons, kind: str = "har"):
        self.horizons = list(horizons)
        if kind == "har":
            self._base = HARRV(horizons)
        elif kind == "har_iv":
            from .classical_arms import HARIV

            self._base = HARIV(horizons)
        elif kind == "const":
            # falsification arm: train-mean level only ("free-running" network)
            self._base = _ConstPrior(horizons)
        else:
            raise ValueError(f"unknown prior kind {kind!r} ('har', 'har_iv', 'const')")

    def fit(self, panel: pd.DataFrame, train_idx: pd.DatetimeIndex) -> "HARPrior":
        self._base.fit(panel, train_idx)
        return self

    def prior_log(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        """Log-vol prior forecasts, shape (len(origins), n_horizons)."""
        vol = self._base.predict(panel, origins)
        return np.log(np.clip(vol, VOL_FLOOR, None))
