"""Forecast metrics. Inputs are forward-vol arrays shaped (n_origins, n_horizons)."""
from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mse(y_true, y_pred) -> float:
    return float(np.mean((_arr(y_true) - _arr(y_pred)) ** 2))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(_arr(y_true) - _arr(y_pred))))


def qlike(y_true_vol, y_pred_vol, eps: float = 1e-12) -> float:
    """QLIKE on variance (scale-robust, penalizes under-prediction correctly)."""
    vt = _arr(y_true_vol) ** 2
    vp = np.clip(_arr(y_pred_vol) ** 2, eps, None)
    return float(np.mean(np.log(vp) + vt / vp))


def pinball(y_true, q_pred, quantiles) -> float:
    """Quantile (pinball) loss. y_true (n,H); q_pred (n,H,Q); quantiles (Q,)."""
    yt = _arr(y_true)[..., None]
    qp = _arr(q_pred)
    qs = _arr(quantiles)
    e = yt - qp
    return float(np.mean(np.maximum(qs * e, (qs - 1.0) * e)))


def coverage(y_true, lo, hi) -> float:
    """Empirical coverage of a [lo, hi] prediction interval."""
    yt = _arr(y_true)
    return float(np.mean((yt >= _arr(lo)) & (yt <= _arr(hi))))


def dm_test(loss_a, loss_b, hac_lag: int = 0):
    """Diebold-Mariano with Bartlett-kernel HAC variance.

    ``hac_lag`` should be >= max_horizon - 1 when losses come from overlapping
    multi-session targets (their differentials are serially correlated by
    construction, and a plain variance overstates significance).
    Returns (stat, two-sided p); stat < 0 means loss_a < loss_b (a is better).
    """
    from scipy import stats

    d = _arr(loss_a) - _arr(loss_b)
    n = len(d)
    if n < 2:
        return 0.0, 1.0
    dc = d - d.mean()
    gamma0 = float(np.mean(dc * dc))
    var_lr = gamma0
    L = min(int(hac_lag), n - 1)
    for k in range(1, L + 1):
        gamma_k = float(np.mean(dc[k:] * dc[:-k]))
        var_lr += 2.0 * (1.0 - k / (L + 1.0)) * gamma_k
    var_lr = max(var_lr, 1e-18) / n
    stat = float(np.mean(d) / np.sqrt(var_lr))
    p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return stat, float(p)


def qlike_per_origin(y_true_vol, y_pred_vol, eps: float = 1e-12) -> np.ndarray:
    """Per-origin QLIKE (mean over horizons), for loss-differential tests. (n,)"""
    vt = _arr(y_true_vol) ** 2
    vp = np.clip(_arr(y_pred_vol) ** 2, eps, None)
    return np.mean(np.log(vp) + vt / vp, axis=-1)
