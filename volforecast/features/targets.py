"""Forward realized-vol targets — strictly after t0 (the only place targets are built).

tgt_rv_{h} = sqrt(mean daily realized variance over the next h sessions), i.e. the
forward realized vol in daily units. Rows without a full h-session future are NaN and
dropped downstream. These are a DIFFERENT object from the trailing feat_rv_* features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutil import pit_guard


@pit_guard("forward")
def forward_targets(rv_var: pd.Series, horizons, *, t0):
    fut = rv_var.loc[rv_var.index > pd.Timestamp(t0)]
    out = {}
    used = []
    for h in horizons:
        seg = fut.iloc[:h]
        if len(seg) < h:
            out[f"tgt_rv_{h}"] = np.nan
        else:
            out[f"tgt_rv_{h}"] = float(np.sqrt(seg.mean()))
            used.extend(seg.index.tolist())
    used_ts = pd.DatetimeIndex(used) if used else pd.DatetimeIndex([])
    return out, used_ts
