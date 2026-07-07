"""Same seed -> identical synthetic data; different seed -> different."""
import numpy as np

from volforecast.data.synthetic import SyntheticAdapter


def test_same_seed_identical():
    a = SyntheticAdapter(start="2020-01-01", end="2020-06-30", seed=7)
    b = SyntheticAdapter(start="2020-01-01", end="2020-06-30", seed=7)
    assert a._bars["close"].equals(b._bars["close"])
    assert a._iv["atm_iv"].equals(b._iv["atm_iv"])


def test_different_seed_differs():
    a = SyntheticAdapter(start="2020-01-01", end="2020-06-30", seed=7)
    c = SyntheticAdapter(start="2020-01-01", end="2020-06-30", seed=8)
    assert not np.allclose(a._bars["close"].values, c._bars["close"].values)
