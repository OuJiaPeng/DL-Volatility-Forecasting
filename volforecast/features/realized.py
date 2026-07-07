"""Realized variance on a 5-minute grid, plus the daily close/return series.

Intraday RV uses only within-session 5-minute log returns (overnight excluded), which
is the standard noise-robust realized-variance construction. Each value is stamped at
the session close, so it is knowable at that day's decision time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import Calendar


def daily_realized_variance(bars: pd.DataFrame, cal: Calendar) -> pd.Series:
    """Per-session realized variance (daily units) indexed by session close ts."""
    out = {}
    dates = pd.Index(bars.index.normalize().unique())
    for d in dates:
        grid = cal.rv_grid(d)
        px = bars["close"].reindex(grid).dropna()
        if len(px) < 3:
            continue
        r = np.diff(np.log(px.values))
        out[cal.session_close(d)] = float(np.sum(r**2))
    s = pd.Series(out, dtype=float).sort_index()
    s.index.name = "ts"
    s.name = "rv_var"
    return s


def daily_close(bars: pd.DataFrame, cal: Calendar) -> pd.Series:
    """Per-session closing price indexed by session close ts."""
    out = {}
    dates = pd.Index(bars.index.normalize().unique())
    for d in dates:
        px = bars["close"].reindex(cal.minute_grid(d)).dropna()
        if len(px) == 0:
            continue
        out[cal.session_close(d)] = float(px.iloc[-1])
    s = pd.Series(out, dtype=float).sort_index()
    s.index.name = "ts"
    s.name = "close"
    return s


def daily_return(bars: pd.DataFrame, cal: Calendar) -> pd.Series:
    """Close-to-close daily log return indexed by session close ts."""
    c = daily_close(bars, cal)
    r = np.log(c).diff()
    r.name = "ret_1d"
    return r


def intraday_stats(bars: pd.DataFrame, cal: Calendar) -> pd.DataFrame:
    """Per-session intraday-structure stats HAR's daily aggregates cannot represent.

    All computed from the same 5-min grid as RV, stamped at session close (causal):
      * semi_neg_share  — downside semivariance share of RV (asymmetry of the day)
      * jump_share      — max(RV - bipower, 0)/RV (jump vs diffusive composition)
      * lasthour_share  — share of RV realized in the final hour (late-day stress)
      * overnight_gap   — |log(open / prior close)| (gap risk the RTH RV misses)
    """
    rows, index = [], []
    prev_close = None
    for d in pd.Index(bars.index.normalize().unique()):
        grid = cal.rv_grid(d)
        px = bars["close"].reindex(grid).dropna()
        if len(px) < 6:
            continue
        r = np.diff(np.log(px.values))
        rv = float(np.sum(r**2))
        if rv <= 0:
            continue
        semi_neg = float(np.sum(r[r < 0] ** 2)) / rv
        bipower = float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        jump = max(rv - bipower, 0.0) / rv
        n_hour = max(int(round(60 / cal.rv_minutes)), 1)
        lasthour = float(np.sum(r[-n_hour:] ** 2)) / rv
        open_px = float(px.iloc[0])
        gap = abs(np.log(open_px / prev_close)) if prev_close else 0.0
        prev_close = float(px.iloc[-1])
        rows.append({"semi_neg_share": semi_neg, "jump_share": jump,
                     "lasthour_share": lasthour, "overnight_gap": gap})
        index.append(cal.session_close(d))
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="ts"))
    return out
