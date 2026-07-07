"""IV-surface summary features: the last snapshot knowable at t0."""
from __future__ import annotations

import pandas as pd

from ..timeutil import pit_guard


@pit_guard("causal")
def surface_features(iv: pd.DataFrame, *, t0):
    hist = iv.loc[iv.index <= pd.Timestamp(t0)]
    if len(hist) == 0:
        return None, pd.DatetimeIndex([])
    row = hist.iloc[-1]
    # every surface column becomes a feature — includes tenor-matched atm_iv_{h}
    # columns when the adapter provides them (Databento path), on top of the base
    # atm_iv / skew / term_slope / vix that every adapter guarantees.
    feats = {f"feat_{c}": float(row[c]) for c in hist.columns}
    return feats, hist.index[-1:]
