"""The centerpiece: prove features cannot see the future and targets cannot see the past."""
import numpy as np
import pandas as pd
import pytest

from volforecast.timeutil import assert_causal, assert_forward, LookaheadError
from volforecast.features.har import har_components
from volforecast.features.targets import forward_targets


def test_guards_raise():
    t0 = pd.Timestamp("2020-06-01 16:00")
    assert_causal([t0 - pd.Timedelta(days=1), t0], t0)          # ok
    assert_forward([t0 + pd.Timedelta(days=1)], t0)             # ok
    with pytest.raises(LookaheadError):
        assert_causal([t0 + pd.Timedelta(days=1)], t0)         # feature peeks ahead
    with pytest.raises(LookaheadError):
        assert_forward([t0], t0)                                # target uses the present


def test_features_truncation_invariant(rv_var):
    """A feature at t0 must be identical whether or not future data exists."""
    t0 = rv_var.index[len(rv_var) // 2]
    full = har_components(rv_var, t0=t0)
    truncated = har_components(rv_var.loc[rv_var.index <= t0], t0=t0)
    assert full == truncated


def test_future_spike_does_not_touch_features_but_does_touch_targets(rv_var):
    t0 = rv_var.index[len(rv_var) // 2]
    spiked = rv_var.copy()
    fut_ts = spiked.index[spiked.index > t0][3]
    spiked.loc[fut_ts] = spiked.max() * 1000  # enormous future anomaly

    # features unchanged
    assert har_components(rv_var, t0=t0) == har_components(spiked, t0=t0)
    # targets DO change (they're supposed to see the future)
    base = forward_targets(rv_var, [1, 5, 10, 21], t0=t0)
    bumped = forward_targets(spiked, [1, 5, 10, 21], t0=t0)
    assert bumped["tgt_rv_5"] != base["tgt_rv_5"]


def test_leaky_feature_is_caught_by_guard(rv_var):
    """A deliberately-leaky builder must trip the PIT guard."""
    from volforecast.timeutil import pit_guard

    @pit_guard("causal")
    def leaky(series, *, t0):
        fut = series.loc[series.index > t0]
        return {"x": float(fut.iloc[0])}, fut.index[:1]  # uses a future ts -> illegal

    t0 = rv_var.index[len(rv_var) // 2]
    with pytest.raises(LookaheadError):
        leaky(rv_var, t0=t0)


def test_panel_pit_invariant_holds(panel):
    feat = [c for c in panel.columns if c.startswith("feat_")]
    tgt = [c for c in panel.columns if c.startswith("tgt_")]
    assert feat and tgt
    assert set(feat).isdisjoint(tgt)
    assert panel[tgt].notna().all().all()  # PanelBuilder drops incomplete-future rows
