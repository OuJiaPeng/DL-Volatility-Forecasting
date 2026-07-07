"""``Forecaster`` ABC — the shared contract so every model predicts on the SAME origins.

predict() returns point forecasts of forward realized vol, shape (len(origins), n_horizons),
in the same daily-vol units as the tgt_rv_* columns. (The v2 hybrid emits quantiles; its
q50 plugs in here.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = "base"

    def __init__(self, horizons):
        self.horizons = list(horizons)

    @abstractmethod
    def fit(self, panel: pd.DataFrame, train_idx: pd.DatetimeIndex) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        ...

    def _flat(self, values: np.ndarray) -> np.ndarray:
        """Broadcast a per-origin scalar forecast across all horizons."""
        return np.repeat(np.asarray(values, dtype=float)[:, None], len(self.horizons), axis=1)
