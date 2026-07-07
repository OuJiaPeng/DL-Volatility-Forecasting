"""Point-in-time (PIT) guards — the enforcement layer for the no-lookahead invariant.

Two clocks:
  * ``ts``  — "knowable-at" time: when a datum became available to a real observer.
  * ``t0``  — decision time / forecast origin.

The invariant, asserted everywhere features and targets are built:
    every feature input has  ts <= t0   (causal)
    every target  input has  ts >  t0   (strictly forward)
"""
from __future__ import annotations

import functools
import pandas as pd


class LookaheadError(AssertionError):
    """Raised when a feature peeks at ``ts > t0`` or a target uses ``ts <= t0``."""


def _to_index(used_ts) -> pd.DatetimeIndex:
    if isinstance(used_ts, pd.DatetimeIndex):
        return used_ts
    if isinstance(used_ts, (pd.Series, pd.Index)):
        return pd.DatetimeIndex(pd.to_datetime(used_ts))
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(list(_as_iter(used_ts)))))


def _as_iter(x):
    if isinstance(x, (pd.Timestamp, str)) or not hasattr(x, "__iter__"):
        return [x]
    return x


def assert_causal(used_ts, t0, name: str = "feature") -> None:
    """Assert all ``used_ts`` are <= ``t0`` (knowable at decision time)."""
    idx = _to_index(used_ts)
    if len(idx) == 0:
        return
    t0 = pd.Timestamp(t0)
    bad = idx[idx > t0]
    if len(bad) > 0:
        raise LookaheadError(
            f"{name}: {len(bad)} input timestamp(s) are AFTER t0={t0} "
            f"(e.g. {bad[:3].tolist()}). A feature is peeking at the future."
        )


def assert_forward(used_ts, t0, name: str = "target") -> None:
    """Assert all ``used_ts`` are > ``t0`` (strictly after decision time)."""
    idx = _to_index(used_ts)
    if len(idx) == 0:
        return
    t0 = pd.Timestamp(t0)
    bad = idx[idx <= t0]
    if len(bad) > 0:
        raise LookaheadError(
            f"{name}: {len(bad)} target input timestamp(s) are AT/BEFORE t0={t0} "
            f"(e.g. {bad[:3].tolist()}). A target is leaking the present/past."
        )


def as_of(obj, t0):
    """Rows of a ts-indexed Series/DataFrame knowable at ``t0`` (index <= t0)."""
    t0 = pd.Timestamp(t0)
    return obj.loc[obj.index <= t0]


def after(obj, t0):
    """Rows of a ts-indexed Series/DataFrame strictly after ``t0`` (index > t0)."""
    t0 = pd.Timestamp(t0)
    return obj.loc[obj.index > t0]


def pit_guard(kind: str):
    """Decorator for builders returning ``(value, used_ts)`` to auto-check PIT.

    The wrapped fn must accept ``t0`` as a kwarg or first positional after self and
    return a tuple ``(value, used_ts)``. ``kind`` is "causal" or "forward".
    """
    check = {"causal": assert_causal, "forward": assert_forward}[kind]

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, t0=None, **kwargs):
            value, used_ts = fn(*args, t0=t0, **kwargs)
            check(used_ts, t0, name=fn.__name__)
            return value

        return wrapper

    return deco
