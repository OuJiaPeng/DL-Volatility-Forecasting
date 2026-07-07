"""Walk-forward evaluation: per-fold fits, val-first reporting, milestone-gated test.

``run_walkforward`` fits each arm fresh per fold (factories receive the fold's Split so
normalization stats and any inner validation stay fold-local), scores val always, and
scores test ONLY when ``milestone=True`` — enforcing the evaluation contract at the API
level rather than by good intentions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import qlike_per_origin, dm_test


def _score(panel, origins, forecaster, tgt_cols):
    y_true = panel.loc[origins, tgt_cols].values
    y_pred = forecaster.predict(panel, origins)
    return qlike_per_origin(y_true, y_pred)


def run_walkforward(panel: pd.DataFrame, folds, factories: dict, horizons,
                    dm_ref: str = "har_rv", milestone: bool = False) -> pd.DataFrame:
    """factories: {arm_name: fn(split) -> fitted-ready Forecaster}. Returns one row/arm."""
    tgt_cols = [f"tgt_rv_{h}" for h in horizons]
    hac_lag = max(horizons) - 1

    val_losses = {name: [] for name in factories}
    test_losses = {name: [] for name in factories}
    coverage = {name: [] for name in factories}
    for split in folds:
        for name, make in factories.items():
            f = make(split)
            f.fit(panel, split.train)
            val_losses[name].append(_score(panel, split.val, f, tgt_cols))
            if hasattr(f, "predict_quantiles"):
                q = f.predict_quantiles(panel, split.val)          # (n, H, Q) vol units
                y = panel.loc[split.val, tgt_cols].values
                coverage[name].append(((y >= q[..., 0]) & (y <= q[..., -1])).ravel())
            if milestone:
                test_losses[name].append(_score(panel, split.test, f, tgt_cols))

    rows = {}
    for name in factories:
        v = np.concatenate(val_losses[name])
        row = {"folds": len(folds), "VAL_QLIKE": float(np.mean(v))}
        # empirical coverage of the [q10, q90] band (nominal 0.80) — quantile arms only
        row["VAL_COV80"] = (float(np.mean(np.concatenate(coverage[name])))
                            if coverage[name] else np.nan)
        if dm_ref in factories and name != dm_ref:
            t, p = dm_test(v, np.concatenate(val_losses[dm_ref]), hac_lag=hac_lag)
            row["VAL_DM_t"], row["VAL_DM_p"] = t, p
        else:
            row["VAL_DM_t"], row["VAL_DM_p"] = np.nan, np.nan
        if milestone:
            te = np.concatenate(test_losses[name])
            row["TEST_QLIKE"] = float(np.mean(te))
            if dm_ref in factories and name != dm_ref:
                t, p = dm_test(te, np.concatenate(test_losses[dm_ref]), hac_lag=hac_lag)
                row["TEST_DM_t"], row["TEST_DM_p"] = t, p
            else:
                row["TEST_DM_t"], row["TEST_DM_p"] = np.nan, np.nan
        else:
            row["TEST_QLIKE"] = row["TEST_DM_t"] = row["TEST_DM_p"] = np.nan
        rows[name] = row

    cols = ["folds", "VAL_QLIKE", "VAL_COV80", "VAL_DM_t", "VAL_DM_p",
            "TEST_QLIKE", "TEST_DM_t", "TEST_DM_p"]
    return pd.DataFrame(rows).T[cols].sort_values("VAL_QLIKE")
