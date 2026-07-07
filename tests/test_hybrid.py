"""v2 hybrid model: smoke training, quantile sanity, determinism, compare integration.

Kept tiny (small dims, few epochs) so the whole file runs in well under a minute on CPU.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from types import SimpleNamespace

from volforecast.splits import make_splits
from volforecast.baselines import build_baselines
from volforecast.eval import compare
from volforecast.models.hybrid import HybridResidualForecaster


def tiny_mcfg(**over):
    base = dict(
        lookback=10,
        emb_dim=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        quantiles=[0.1, 0.5, 0.9],
        lr=1e-3,
        epochs=3,
        patience=5,
        batch_size=64,
        lambda_qlike=0.1,
        n_runs=1,
        seed=7,
        device="cpu",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(scope="module")
def fitted(panel_module, cfg_module):
    split = make_splits(panel_module, cfg_module)
    f = HybridResidualForecaster(cfg_module.horizons, tiny_mcfg()).bind_split(split)
    f.fit(panel_module, split.train)
    return f, split


@pytest.fixture(scope="module")
def panel_module(cfg_module):
    from volforecast.data.synthetic import SyntheticAdapter
    from volforecast.panel import PanelBuilder

    adapter = SyntheticAdapter(start=cfg_module.start, end=cfg_module.end, seed=cfg_module.seed)
    return PanelBuilder(cfg_module, adapter).build()


@pytest.fixture(scope="module")
def cfg_module():
    from conftest import make_cfg

    return make_cfg()


def test_smoke_shapes_and_finite(fitted, panel_module, cfg_module):
    f, split = fitted
    pred = f.predict(panel_module, split.test)
    assert pred.shape == (len(split.test), len(cfg_module.horizons))
    assert np.isfinite(pred).all()
    assert (pred > 0).all()  # vol is positive


def test_quantiles_monotone_and_bracket_median(fitted, panel_module):
    f, split = fitted
    q = f.predict_quantiles(panel_module, split.test)  # (n, H, Q)
    assert q.shape[-1] == 3
    assert (np.diff(q, axis=-1) >= 0).all()  # q10 <= q50 <= q90 everywhere


def test_deterministic_same_seed(panel_module, cfg_module):
    split = make_splits(panel_module, cfg_module)
    preds = []
    for _ in range(2):
        f = HybridResidualForecaster(cfg_module.horizons, tiny_mcfg(epochs=2)).bind_split(split)
        f.fit(panel_module, split.train)
        preds.append(f.predict(panel_module, split.test))
    assert np.allclose(preds[0], preds[1])


def test_hybrid_in_compare_table(fitted, panel_module, cfg_module):
    f, split = fitted
    forecasters = build_baselines(cfg_module.horizons) + [f]
    table = compare(panel_module, split, forecasters, cfg_module.horizons)
    assert "hybrid" in table.index
    assert np.isfinite(table.loc["hybrid"].values).all()


def test_losses_match_numpy():
    from volforecast.models.losses import pinball_loss, qlike_loss
    from volforecast.eval.metrics import pinball, qlike

    rng = np.random.default_rng(0)
    y_log = rng.normal(-4, 0.3, (8, 4)).astype(np.float32)
    q_log = np.sort(rng.normal(-4, 0.3, (8, 4, 3)), axis=-1).astype(np.float32)
    qs = [0.1, 0.5, 0.9]

    t_pin = pinball_loss(torch.tensor(y_log), torch.tensor(q_log), torch.tensor(qs)).item()
    assert np.isclose(t_pin, pinball(y_log, q_log, qs), atol=1e-6)

    yhat_log = q_log[..., 1]
    t_ql = qlike_loss(torch.tensor(y_log), torch.tensor(yhat_log)).item()
    n_ql = qlike(np.exp(y_log), np.exp(yhat_log))
    assert np.isclose(t_ql, n_ql, atol=1e-4)
