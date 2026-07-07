"""Shared intraday-arena utilities: augmented feature set, day-folds, OOS helpers.

The augmented linear champion (diurnal-HAR + IV + event-clock interactions) was
discovered via scripts/intraday_oracle.py + the gate-chase documented in
experiments/README.md. Feature construction lives here so the distributional and
hedging chapters can't drift from the oracle's definitions.
"""
import numpy as np
import pandas as pd

EPS = 1e-5

# scheduled FOMC statement days (14:00 ET), 2021-2026H1
FOMC_DATES = pd.to_datetime([
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
])


def base_features(df):
    L = lambda s: np.log(df[s].values + EPS)
    tod = pd.get_dummies(df.block).values.astype(float)
    dow = pd.get_dummies(df.dow).values.astype(float)
    return np.column_stack([
        tod, dow, L("rv30"), L("rv60"), L("rvop"), L("rv_d"), L("rv_w"), L("rv_m"),
        L("iv1"), np.log(df.vix.values), df[["ret30", "trend", "gap"]].values,
        np.abs(df.gap.values)[:, None], L("rmax"), df[["v30"]].values,
    ])


def aug_features(df):
    """Champion feature set: base + event-clock interactions (clock x IV-premium,
    clock x IV level, clock x |ret|, FOMC x clock, FOMC x IV-premium)."""
    L = lambda s: np.log(df[s].values + EPS)
    tod = pd.get_dummies(df.block).values.astype(float)
    fomc = df.day.isin(FOMC_DATES).values.astype(float)[:, None]
    ivprem = (L("iv1") - L("rv_w"))[:, None]
    return np.column_stack([
        base_features(df),
        tod * ivprem, tod * L("iv1")[:, None],
        tod * np.abs(df.ret30.values)[:, None],
        fomc, fomc * tod, fomc * ivprem,
    ])


def day_folds(df, n_folds=4, purge_days=1):
    """Contiguous day-block folds. Returns (fold_id_per_row, train_mask(k))."""
    uniq = np.array(sorted(df.day.unique()))
    dpos = {d: j for j, d in enumerate(uniq)}
    dfold = np.minimum(np.arange(len(uniq)) * n_folds // len(uniq), n_folds - 1)
    fold = df.day.map(lambda d: dfold[dpos[d]]).values
    pos = df.day.map(dpos).values

    def train_mask(k):
        kpos = np.where(dfold == k)[0]
        return (fold != k) & ((pos < kpos.min() - purge_days) | (pos > kpos.max() + purge_days))

    return fold, train_mask


def fit_oos(X, y, fold, train_mask, model_fn, n_folds=4):
    """Fold-wise OOS predictions with train-only standardization."""
    p = np.full(len(y), np.nan)
    for k in range(n_folds):
        tr, te = train_mask(k), fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        p[te] = model_fn().fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
    return p
