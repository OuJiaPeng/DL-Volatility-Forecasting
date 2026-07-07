"""Naive-IV: use the market's ATM implied vol as the realized-vol forecast.

This is THE bar to beat. If the custom model can't beat "just use the market's number",
that is reported as a finding, not hidden.
"""
from __future__ import annotations

import pandas as pd

from .base import Forecaster


class NaiveIV(Forecaster):
    name = "naive_iv"

    def fit(self, panel, train_idx):
        return self

    def predict(self, panel, origins):
        # tenor-matched bar when per-horizon ATM IV columns exist (real-data path);
        # otherwise a single ATM tenor broadcast flat across horizons.
        tenor_cols = [f"feat_atm_iv_{h}" for h in self.horizons]
        if all(c in panel.columns for c in tenor_cols):
            return panel.loc[origins, tenor_cols].values.astype(float)
        return self._flat(panel.loc[origins, "feat_atm_iv"].values)
