"""Metric correctness on known inputs."""
import numpy as np

from volforecast.eval.metrics import mse, mae, qlike, pinball


def test_perfect_forecast():
    y = np.array([[0.01, 0.02], [0.015, 0.03]])
    assert mse(y, y) == 0.0
    assert mae(y, y) == 0.0
    # QLIKE of a perfect forecast = mean(log(var) + 1)
    expected = float(np.mean(np.log(y**2) + 1.0))
    assert np.isclose(qlike(y, y), expected)


def test_mae_simple():
    y = np.array([[1.0, 2.0]])
    p = np.array([[1.5, 1.0]])
    assert np.isclose(mae(y, p), 0.75)


def test_pinball_median_is_half_mae():
    y = np.array([[1.0], [2.0]])
    q_pred = np.array([[[1.5]], [[1.0]]])  # (n=2, H=1, Q=1)
    assert np.isclose(pinball(y, q_pred, [0.5]), 0.5 * mae(y, q_pred[..., 0]))


def test_qlike_penalizes_underprediction_more():
    y = np.array([[0.02]])
    under = qlike(y, np.array([[0.01]]))
    over = qlike(y, np.array([[0.04]]))  # over by same factor of 2
    assert under > over
