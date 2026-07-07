"""Walk-forward splitter invariants + trunk-registry contract."""
import numpy as np
import pandas as pd
import pytest

from volforecast.splits import make_walkforward


def test_walkforward_fold_invariants(panel, cfg):
    folds = make_walkforward(panel, cfg, n_folds=3)
    assert len(folds) == 3
    pos = {t: i for i, t in enumerate(panel.index)}
    max_h = max(cfg.horizons)
    prev_test_end = -1
    for s in folds:
        assert len(s.train) and len(s.val) and len(s.test)
        # disjoint and ordered within the fold
        assert s.train.max() < s.val.min() < s.test.min()
        # purge gaps at both boundaries exceed the target horizon
        assert pos[s.val.min()] - pos[s.train.max()] > max_h
        assert pos[s.test.min()] - pos[s.val.max()] > max_h
        # test blocks tile forward without overlap across folds
        assert pos[s.test.min()] > prev_test_end
        prev_test_end = pos[s.test.max()]
    # expanding train: each fold sees at least as much history as the previous
    trains = [len(s.train) for s in folds]
    assert trains == sorted(trains)


def test_walkforward_norm_stats_are_fold_local(panel, cfg):
    folds = make_walkforward(panel, cfg, n_folds=3)
    a, b = folds[0], folds[-1]
    assert not a.mean.equals(b.mean)  # stats from different (expanding) train slices


def test_trunk_registry_contract():
    torch = pytest.importorskip("torch")
    from volforecast.models.trunks import TRUNKS, build_trunk

    B, L, V, S, H, Q = 4, 10, 7, 4, 4, 3
    x = torch.randn(B, L, V)
    surf = torch.randn(B, S)
    intra = torch.randn(B, 5 * 78, 3)
    for name in TRUNKS:
        trunk = build_trunk(name, n_vars=V, lookback=L, n_surface=S,
                            n_horizons=H, n_quantiles=Q, emb_dim=16, n_heads=2, n_layers=1)
        for xi in (None, intra):
            out = trunk(x, surf, x_intra=xi)
            assert out.shape == (B, H, Q), name
            # nesting: every trunk starts at exactly zero residual (== the HAR prior)
            assert torch.allclose(out, torch.zeros_like(out)), f"{name} not zero-init"
            assert (out.sort(dim=-1).values == out).all(), f"{name} quantiles not monotone"


def test_intraday_cube_alignment(adapter, calendar, panel):
    from volforecast.datasets import build_intraday_cube, BARS_PER_DAY

    bars = adapter.minute_bars(adapter.start, adapter.end)
    cube = build_intraday_cube(bars, calendar, panel.index)
    assert cube.shape == (len(panel), BARS_PER_DAY, 3)
    mid = cube[len(panel) // 2]
    assert np.abs(mid[:, 0]).sum() > 0                       # returns present
    assert np.allclose(mid[:, 1], np.abs(mid[:, 0]))         # ch1 = |ch0|
    assert (mid[:, 2] >= 0).all()                            # ch2 = squared


def test_classical_arms_shapes_and_finite(panel, cfg):
    from volforecast.models.classical_arms import StateDependentHAR, GradientBoostedTrees

    folds = make_walkforward(panel, cfg, n_folds=3)
    s = folds[0]
    for cls in (StateDependentHAR, GradientBoostedTrees):
        f = cls(cfg.horizons).fit(panel, s.train)
        pred = f.predict(panel, s.val)
        assert pred.shape == (len(s.val), len(cfg.horizons))
        assert np.isfinite(pred).all() and (pred > 0).all(), cls.__name__


def test_statehar_stays_in_har_family(panel, cfg):
    """statehar is HAR + state interactions: predictions should track plain HAR closely."""
    from volforecast.models.classical_arms import StateDependentHAR
    from volforecast.baselines import HARRV

    folds = make_walkforward(panel, cfg, n_folds=3)
    s = folds[0]
    sh = StateDependentHAR(cfg.horizons).fit(panel, s.train)
    har = HARRV(cfg.horizons).fit(panel, s.train)
    p1, p2 = sh.predict(panel, s.val), har.predict(panel, s.val)
    assert np.corrcoef(p1[:, 1], p2[:, 1])[0, 1] > 0.8


def test_har_iv_arm_and_prior(panel, cfg):
    from volforecast.models.classical_arms import HARIV
    from volforecast.models.prior import HARPrior

    folds = make_walkforward(panel, cfg, n_folds=3)
    s = folds[0]
    f = HARIV(cfg.horizons).fit(panel, s.train)
    pred = f.predict(panel, s.val)
    assert pred.shape == (len(s.val), len(cfg.horizons))
    assert np.isfinite(pred).all() and (pred > 0).all()
    # prior wrapper: har_iv prior differs from plain har (IV carries information)
    p_har = HARPrior(cfg.horizons, kind="har").fit(panel, s.train).prior_log(panel, s.val)
    p_iv = HARPrior(cfg.horizons, kind="har_iv").fit(panel, s.train).prior_log(panel, s.val)
    assert p_har.shape == p_iv.shape
    assert not np.allclose(p_har, p_iv)
    with pytest.raises(ValueError):
        HARPrior(cfg.horizons, kind="kalman")


def test_unknown_trunk_raises():
    pytest.importorskip("torch")
    from volforecast.models.trunks import build_trunk

    with pytest.raises(ValueError):
        build_trunk("resnet50", n_vars=3, lookback=5, n_surface=2, n_horizons=2, n_quantiles=3)
