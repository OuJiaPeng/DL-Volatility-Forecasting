"""Time-ordered train/val/test splits with purge + embargo for overlapping labels.

A multi-session target at origin t0 covers sessions t0+1..t0+max_h, so origins within
``max_h + embargo`` of a split boundary are dropped (purged) to prevent their forward
windows from leaking across the boundary. Normalization stats are fit on TRAIN only.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class Split:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex
    feat_cols: list
    mean: pd.Series
    std: pd.Series
    purge: int = 0

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df[self.feat_cols] - self.mean) / self.std


def make_walkforward(panel: pd.DataFrame, cfg, n_folds: int = 4,
                     test_span_frac: float = 0.4, val_frac: float = 0.10) -> list:
    """Walk-forward folds: expanding train, purged val block, purged test block.

    The last ``test_span_frac`` of origins is tiled into ``n_folds`` equal test
    blocks; each fold's val block (length ``val_frac`` of all origins) immediately
    precedes its test block; train is everything earlier. Purge gaps (max horizon +
    embargo) are removed at BOTH boundaries so no overlapping target window crosses
    a boundary. Fold geometry must stay fixed for a whole experiment campaign.
    """
    horizons = list(getattr(cfg, "horizons", [1, 5, 10, 21]))
    purge = max(horizons) + int(getattr(cfg, "embargo", 5))
    t0 = panel.index
    n = len(t0)
    feat_cols = [c for c in panel.columns if c.startswith("feat_")]

    test_start0 = int(n * (1.0 - test_span_frac))
    block = (n - test_start0) // n_folds
    val_len = max(int(n * val_frac), purge + 5)

    folds = []
    for i in range(n_folds):
        te_a = test_start0 + i * block
        te_b = te_a + block if i < n_folds - 1 else n
        va_b = te_a - purge
        va_a = va_b - val_len
        tr_b = va_a - purge
        if tr_b < purge + 10:
            raise ValueError(f"fold {i}: not enough history for a train slice")
        train, val, test = t0[:tr_b], t0[va_a:va_b], t0[te_a:te_b]
        mean = panel.loc[train, feat_cols].mean()
        std = panel.loc[train, feat_cols].std().replace(0.0, 1.0)
        folds.append(Split(train, val, test, feat_cols, mean, std, purge))
    return folds


def make_splits(panel: pd.DataFrame, cfg) -> Split:
    horizons = list(getattr(cfg, "horizons", [1, 5, 10, 21]))
    max_h = max(horizons)
    embargo = int(getattr(cfg, "embargo", 5))
    purge = max_h + embargo

    train_frac = float(getattr(cfg, "train_frac", 0.6))
    val_frac = float(getattr(cfg, "val_frac", 0.2))

    t0 = panel.index
    n = len(t0)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))

    train = t0[:i_tr]
    val = t0[i_tr:i_va]
    test = t0[i_va:]

    # purge the tail of train and val so their target windows don't cross the boundary
    train = train[:-purge] if len(train) > purge else train[:0]
    val = val[:-purge] if len(val) > purge else val[:0]

    feat_cols = [c for c in panel.columns if c.startswith("feat_")]
    mean = panel.loc[train, feat_cols].mean()
    std = panel.loc[train, feat_cols].std().replace(0.0, 1.0)
    return Split(train, val, test, feat_cols, mean, std, purge)
