"""Shared fixtures: a small, deterministic synthetic panel (no paid data, fast)."""
from types import SimpleNamespace

import pytest

from volforecast.calendar import Calendar
from volforecast.data.synthetic import SyntheticAdapter
from volforecast.features.realized import daily_realized_variance
from volforecast.panel import PanelBuilder


def make_cfg(**over):
    base = dict(
        vendor="synthetic",
        start="2020-01-01",
        end="2021-06-30",
        seed=7,
        open_time="09:30",
        close_time="16:00",
        rv_minutes=5,
        horizons=[1, 5, 10, 21],
        train_frac=0.6,
        val_frac=0.2,
        embargo=3,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def cfg():
    return make_cfg()


@pytest.fixture
def calendar():
    return Calendar()


@pytest.fixture
def adapter(cfg):
    return SyntheticAdapter(start=cfg.start, end=cfg.end, seed=cfg.seed)


@pytest.fixture
def rv_var(adapter, calendar):
    return daily_realized_variance(adapter.minute_bars(adapter.start, adapter.end), calendar)


@pytest.fixture
def panel(cfg, adapter):
    return PanelBuilder(cfg, adapter).build()
