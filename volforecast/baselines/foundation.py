"""Foundation-model zero-shot baseline (optional; needs the [foundation] extra).

Feeds the trailing daily realized-vol series (a fair, causal, univariate context) to a
pretrained forecaster (Chronos by default) and reads off the median forecast per horizon.
Not in the default registry so core install/tests stay light; add via build_baselines(..,
names=[..,"foundation"]) once chronos-forecasting is installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class FoundationZeroShot(Forecaster):
    name = "foundation"

    def __init__(self, horizons, model_name: str = "amazon/chronos-t5-small", context: int = 256):
        super().__init__(horizons)
        self.model_name = model_name
        self.context = context
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:  # pragma: no cover - heavy optional dep
            from chronos import ChronosPipeline
            import torch

            self._pipe = ChronosPipeline.from_pretrained(self.model_name, torch_dtype=torch.float32)

    def fit(self, panel, train_idx):
        # zero-shot: just remember the causal RV series (feat_rv_d as a vol level proxy)
        self._series = panel["feat_rv_d"]
        return self

    def predict(self, panel, origins):  # pragma: no cover - heavy optional dep
        self._ensure()
        import torch

        max_h = max(self.horizons)
        out = []
        for t0 in origins:
            ctx = self._series.loc[self._series.index <= t0].dropna().values[-self.context:]
            fc = self._pipe.predict(torch.tensor(ctx, dtype=torch.float32), prediction_length=max_h)
            median = np.median(fc[0].numpy(), axis=0)  # (max_h,)
            out.append([float(np.mean(median[:h])) for h in self.horizons])
        return np.asarray(out, dtype=float)
