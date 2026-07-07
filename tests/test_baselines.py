"""Baselines have the right shapes, are finite, and match their definitions."""
import numpy as np

from volforecast.splits import make_splits
from volforecast.baselines import build_baselines, Persistence, NaiveIV
from volforecast.eval import compare


def test_all_baselines_shape_and_finite(panel, cfg):
    s = make_splits(panel, cfg)
    for f in build_baselines(cfg.horizons):
        f.fit(panel, s.train)
        pred = f.predict(panel, s.test)
        assert pred.shape == (len(s.test), len(cfg.horizons))
        assert np.isfinite(pred).all(), f"{f.name} produced non-finite forecasts"


def test_naive_iv_equals_atm_iv(panel, cfg):
    s = make_splits(panel, cfg)
    f = NaiveIV(cfg.horizons).fit(panel, s.train)
    pred = f.predict(panel, s.test)
    expected = panel.loc[s.test, "feat_atm_iv"].values
    assert np.allclose(pred[:, 0], expected)
    assert np.allclose(pred[:, -1], expected)  # flat across horizons


def test_persistence_equals_trailing_rv(panel, cfg):
    s = make_splits(panel, cfg)
    f = Persistence(cfg.horizons).fit(panel, s.train)
    pred = f.predict(panel, s.test)
    assert np.allclose(pred[:, 0], panel.loc[s.test, "feat_rv_d"].values)


def test_compare_table(panel, cfg):
    s = make_splits(panel, cfg)
    table = compare(panel, s, build_baselines(cfg.horizons), cfg.horizons)
    assert set(table.index) == {"persistence", "har_rv", "naive_iv", "garch"}
    assert list(table.columns) == ["MSE", "MAE", "QLIKE", "DM_t", "DM_p"]
    assert table["MSE"].notna().all()
    # DM columns: NaN for the reference (har_rv), finite for everyone else
    assert np.isnan(table.loc["har_rv", "DM_t"])
    others = table.drop("har_rv")
    assert np.isfinite(others["DM_t"]).all()
