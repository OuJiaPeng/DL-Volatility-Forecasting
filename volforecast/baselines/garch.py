"""GARCH(1,1) baseline: refit on daily returns up to each origin, forecast h-step variance.

Returns are scaled to percent before fitting (rescale=False on ~1e-2 returns is poorly
conditioned — the audit lesson) and the forecast variance is scaled back. If history is
short or the fit fails, falls back to the trailing realized vol (so it never errors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class GARCH(Forecaster):
    name = "garch"

    def __init__(self, horizons, min_obs: int = 250):
        super().__init__(horizons)
        self.min_obs = min_obs

    def fit(self, panel, train_idx):
        self._ret = panel["meta_ret_1d"]
        return self

    def predict(self, panel, origins):
        from arch import arch_model

        ret = self._ret
        max_h = max(self.horizons)
        out = []
        for t0 in origins:
            r = ret.loc[ret.index <= t0].dropna().values.astype(float)
            fallback = float(panel.loc[t0, "feat_rv_d"])
            if len(r) < self.min_obs:
                out.append([fallback] * len(self.horizons))
                continue
            try:
                am = arch_model(r * 100.0, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
                res = am.fit(disp="off")
                var_steps = res.forecast(horizon=max_h, reindex=False).variance.values[-1] / 1e4
                row = [float(np.sqrt(np.mean(var_steps[:h]))) for h in self.horizons]
            except Exception:
                row = [fallback] * len(self.horizons)
            out.append(row)
        return np.asarray(out, dtype=float)
