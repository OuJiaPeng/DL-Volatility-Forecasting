"""HAR-RV feature components (trailing daily / weekly / monthly realized vol).

All in *vol* units (sqrt of averaged realized variance), using only sessions with
close <= t0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import pit_guard


@pit_guard("causal")
def har_components(rv_var: pd.Series, *, t0):
    hist = rv_var.loc[rv_var.index <= pd.Timestamp(t0)]
    if len(hist) < 1:
        return None, pd.DatetimeIndex([])
    d = hist.iloc[-1]
    w = hist.iloc[-5:].mean()
    m = hist.iloc[-22:].mean()
    feats = {
        "feat_rv_d": float(np.sqrt(d)),
        "feat_rv_w": float(np.sqrt(w)),
        "feat_rv_m": float(np.sqrt(m)),
    }
    used_ts = hist.index[-22:]
    return feats, used_ts
