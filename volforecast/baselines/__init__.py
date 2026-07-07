"""Baseline forecasters. The default registry is the bar the custom model must clear."""
from .base import Forecaster
from .persistence import Persistence
from .naive_iv import NaiveIV
from .har_rv import HARRV
from .garch import GARCH

DEFAULT_BASELINES = {
    "persistence": Persistence,
    "har_rv": HARRV,
    "naive_iv": NaiveIV,
    "garch": GARCH,
}


def build_baselines(horizons, names=None):
    names = names or list(DEFAULT_BASELINES)
    return [DEFAULT_BASELINES[n](horizons) for n in names]


__all__ = ["Forecaster", "Persistence", "NaiveIV", "HARRV", "GARCH",
           "DEFAULT_BASELINES", "build_baselines"]
