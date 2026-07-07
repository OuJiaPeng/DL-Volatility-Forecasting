"""``MarketDataAdapter`` — the one interface every data vendor implements.

Keeping this surface tiny is what makes the Massive -> Databento swap a one-file
change. Everything downstream (features, panel, baselines) depends only on this ABC,
never on a concrete vendor.

All returned frames are indexed by ``ts`` (a tz-naive DatetimeIndex of *knowable-at*
timestamps), sorted ascending and unique.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class MarketDataAdapter(ABC):
    @abstractmethod
    def minute_bars(self, start, end) -> pd.DataFrame:
        """Underlying minute OHLCV. Index ``ts`` = bar-end. Cols: open/high/low/close/volume."""

    @abstractmethod
    def iv_surface(self, start, end) -> pd.DataFrame:
        """Per-session IV-surface summary knowable at each session close.

        Index ``ts`` = session close. Cols: atm_iv, skew, term_slope, vix
        (all in daily-vol units except vix, which is annualized %).
        """


def get_adapter(cfg) -> MarketDataAdapter:
    """Factory: build the adapter named by ``cfg.vendor``."""
    vendor = getattr(cfg, "vendor", "synthetic")
    if vendor == "synthetic":
        from .synthetic import SyntheticAdapter

        return SyntheticAdapter(
            start=cfg.start,
            end=cfg.end,
            seed=getattr(cfg, "seed", 7),
        )
    if vendor == "databento":
        from .databento import DatabentoAdapter

        return DatabentoAdapter(cfg)
    if vendor == "massive":
        from .massive import MassiveAdapter

        return MassiveAdapter(cfg)
    raise ValueError(
        f"Unknown vendor: {vendor!r} (expected 'synthetic', 'databento', or 'massive')"
    )
