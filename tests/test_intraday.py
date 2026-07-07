"""Intraday-structure features: shapes, bounds, causality; zero-init head; HAC DM test."""
import numpy as np
import pandas as pd
import pytest

from volforecast.features.realized import intraday_stats
from volforecast.features.intraday import intraday_features
from volforecast.eval.metrics import dm_test


def test_intraday_stats_shape_and_bounds(adapter, calendar):
    bars = adapter.minute_bars(adapter.start, adapter.end)
    stats = intraday_stats(bars, calendar)
    assert set(stats.columns) == {"semi_neg_share", "jump_share", "lasthour_share", "overnight_gap"}
    assert len(stats) > 100
    for c in ("semi_neg_share", "jump_share", "lasthour_share"):
        assert stats[c].between(0.0, 1.0).all(), c
    assert (stats["overnight_gap"] >= 0).all()


def test_intraday_features_causal(adapter, calendar):
    bars = adapter.minute_bars(adapter.start, adapter.end)
    stats = intraday_stats(bars, calendar)
    t0 = stats.index[len(stats) // 2]
    full = intraday_features(stats, t0=t0)
    truncated = intraday_features(stats.loc[stats.index <= t0], t0=t0)
    assert full == truncated  # future rows must not influence the feature


def test_panel_contains_intraday_features(panel):
    for c in ("feat_semi_neg_share", "feat_jump_share", "feat_lasthour_share", "feat_overnight_gap"):
        assert c in panel.columns
        assert panel[c].notna().all()


def test_dm_hac_reduces_significance_on_autocorrelated_losses():
    rng = np.random.default_rng(0)
    # build serially-correlated loss differentials (like overlapping horizons create)
    e = rng.standard_normal(300)
    d = np.convolve(e, np.ones(20) / 20, mode="same") + 0.05
    a = np.zeros(300)
    t_plain, p_plain = dm_test(a + d, a, hac_lag=0)
    t_hac, p_hac = dm_test(a + d, a, hac_lag=20)
    assert abs(t_hac) < abs(t_plain)  # HAC widens the variance -> smaller |t|
    t0, p0 = dm_test(a, a)
    assert t0 == 0.0 and p0 == 1.0


def test_quantile_head_zero_init_starts_at_prior():
    torch = pytest.importorskip("torch")
    from volforecast.models.heads import QuantileHead

    head = QuantileHead(emb_dim=16, n_horizons=4, n_quantiles=3)
    z = torch.randn(8, 16)
    out = head(z)
    assert torch.allclose(out, torch.zeros_like(out))  # residual == 0 at init
