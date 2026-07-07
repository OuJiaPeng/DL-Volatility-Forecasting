"""Intraday-structure features: the last session's stats knowable at t0 (PIT-guarded)."""
from __future__ import annotations

import pandas as pd

from ..timeutil import pit_guard


@pit_guard("causal")
def intraday_features(stats: pd.DataFrame, *, t0):
    hist = stats.loc[stats.index <= pd.Timestamp(t0)]
    if len(hist) == 0:
        return None, pd.DatetimeIndex([])
    row = hist.iloc[-1]
    feats = {f"feat_{c}": float(row[c]) for c in hist.columns}
    return feats, hist.index[-1:]
