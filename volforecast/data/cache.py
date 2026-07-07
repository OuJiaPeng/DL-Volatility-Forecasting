"""Tiny parquet cache helpers (pyarrow)."""
from __future__ import annotations

import os
import pandas as pd


def save_parquet(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_parquet(path)
    return path


def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached file at {path}. Build it first.")
    return pd.read_parquet(path)
