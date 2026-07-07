"""HAR-RV (Corsi 2009): OLS of forward vol on trailing daily/weekly/monthly vol.

One OLS per horizon, fit on TRAIN only, applied out-of-sample. Coefficients absorb scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class HARRV(Forecaster):
    name = "har_rv"

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            np.ones(len(df)),
            df["feat_rv_d"].values,
            df["feat_rv_w"].values,
            df["feat_rv_m"].values,
        ]).astype(float)

    def fit(self, panel, train_idx):
        X = self._design(panel.loc[train_idx])
        self.betas = {}
        for h in self.horizons:
            y = panel.loc[train_idx, f"tgt_rv_{h}"].values.astype(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.betas[h] = beta
        return self

    def predict(self, panel, origins):
        X = self._design(panel.loc[origins])
        cols = [np.clip(X @ self.betas[h], 0.0, None) for h in self.horizons]
        return np.column_stack(cols)
