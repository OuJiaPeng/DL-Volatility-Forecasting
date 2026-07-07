"""Deterministic calendar features known at t0."""
from __future__ import annotations

import pandas as pd

from ..timeutil import pit_guard


@pit_guard("causal")
def calendar_features(*, t0):
    t0 = pd.Timestamp(t0)
    feats = {
        "feat_dow": float(t0.dayofweek),
        "feat_dom": float(t0.day),
        "feat_month": float(t0.month),
    }
    return feats, pd.DatetimeIndex([t0])
