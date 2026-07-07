"""Comparison harness: every forecaster fit on train, scored on the SAME test origins."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import mse, mae, qlike, qlike_per_origin, dm_test


def compare(panel: pd.DataFrame, split, forecasters, horizons,
            dm_ref: str = "har_rv") -> pd.DataFrame:
    test = split.test
    tgt_cols = [f"tgt_rv_{h}" for h in horizons]
    y_true = panel.loc[test, tgt_cols].values
    hac_lag = max(horizons) - 1

    rows, losses = {}, {}
    for f in forecasters:
        f.fit(panel, split.train)
        y_pred = f.predict(panel, test)
        rows[f.name] = {
            "MSE": mse(y_true, y_pred),
            "MAE": mae(y_true, y_pred),
            "QLIKE": qlike(y_true, y_pred),
        }
        losses[f.name] = qlike_per_origin(y_true, y_pred)

    # Diebold-Mariano vs the reference model on per-origin QLIKE (HAC variance,
    # lag = max horizon - 1, since overlapping targets serially correlate losses).
    # stat < 0: model beats the reference.
    for name in rows:
        if name == dm_ref or dm_ref not in losses:
            rows[name]["DM_t"], rows[name]["DM_p"] = np.nan, np.nan
            continue
        t, p = dm_test(losses[name], losses[dm_ref], hac_lag=hac_lag)
        rows[name]["DM_t"], rows[name]["DM_p"] = t, p

    table = pd.DataFrame(rows).T[["MSE", "MAE", "QLIKE", "DM_t", "DM_p"]]
    return table.sort_values("QLIKE")
