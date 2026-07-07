"""Persistence / random walk: forward vol == most recent trailing realized vol."""
from __future__ import annotations

import pandas as pd

from .base import Forecaster


class Persistence(Forecaster):
    name = "persistence"

    def fit(self, panel, train_idx):
        return self

    def predict(self, panel, origins):
        return self._flat(panel.loc[origins, "feat_rv_d"].values)
