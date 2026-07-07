"""Canonical column names + lightweight PIT/dtype validation (no pandera dep in v1)."""
from __future__ import annotations

import pandas as pd

BAR_COLS = ["open", "high", "low", "close", "volume"]
IV_COLS = ["atm_iv", "skew", "term_slope", "vix"]


def _validate_index(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: index must be a DatetimeIndex, got {type(df.index)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{name}: index must be sorted ascending")
    if not df.index.is_unique:
        raise ValueError(f"{name}: index has duplicate timestamps")


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    _validate_index(df, "minute_bars")
    missing = [c for c in BAR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"minute_bars missing columns: {missing}")
    if df["close"].isna().any():
        raise ValueError("minute_bars: NaN in close")
    return df


def validate_iv(df: pd.DataFrame) -> pd.DataFrame:
    _validate_index(df, "iv_surface")
    missing = [c for c in IV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"iv_surface missing columns: {missing}")
    return df
