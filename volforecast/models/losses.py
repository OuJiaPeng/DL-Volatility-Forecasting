"""Training losses in log-vol space.

The model predicts log-vol quantiles. Pinball gives the distributional fit; QLIKE on
the median adds the vol-appropriate asymmetric penalty (under-predicting variance is
punished harder than over-predicting, matching how vol forecasts are consumed).
"""
from __future__ import annotations

import torch


def pinball_loss(y_log: torch.Tensor, q_log: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """y_log (B,H); q_log (B,H,Q); quantiles (Q,). Mean pinball over B,H,Q."""
    e = y_log.unsqueeze(-1) - q_log
    q = quantiles.view(1, 1, -1)
    return torch.mean(torch.maximum(q * e, (q - 1.0) * e))


def qlike_loss(y_log: torch.Tensor, yhat_log: torch.Tensor) -> torch.Tensor:
    """QLIKE with variance = exp(2*log_vol): log(vp) + vt/vp = 2*yhat + exp(2*(y-yhat)).

    Minimized at yhat == y; inputs are log-vols shaped (B,H).
    """
    z = 2.0 * (y_log - yhat_log)
    return torch.mean(2.0 * yhat_log + torch.exp(torch.clamp(z, max=30.0)))


def combined_loss(y_log, q_log, quantiles, median_idx: int, lambda_qlike: float) -> torch.Tensor:
    loss = pinball_loss(y_log, q_log, quantiles)
    if lambda_qlike > 0:
        loss = loss + lambda_qlike * qlike_loss(y_log, q_log[..., median_idx])
    return loss
