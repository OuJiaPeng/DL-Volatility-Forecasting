"""Deterministic synthetic vendor — lets the whole pipeline + tests run with no paid data.

Generates minute bars from an AR(1)-log-vol (stochastic-volatility) process so that
realized vol has genuine persistence (HAR/persistence are meaningful), and an IV
series that is a noisy, premium-inflated view of *future* realized vol so that:
  * naive-IV is a genuinely skillful baseline, and
  * the variance risk premium (IV > subsequent RV) is positive and time-varying.

NOTE on the forward-looking IV construction: the *generator* uses the future latent
vol path to set IV — this simulates an efficient market pricing expected forward vol.
It is NOT pipeline leakage: each IV value is stamped at ts = session close (t0) and is
therefore knowable at the decision time that consumes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .adapter import MarketDataAdapter
from .schema import validate_bars, validate_iv
from ..calendar import Calendar


class SyntheticAdapter(MarketDataAdapter):
    def __init__(
        self,
        start,
        end,
        calendar: Calendar | None = None,
        seed: int = 7,
        phi: float = 0.95,
        vol_of_vol: float = 0.22,
        base_daily_vol: float = 0.011,
        vrp: float = 0.15,
        iv_horizon: int = 21,
    ):
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.cal = calendar or Calendar()
        self.seed = seed
        self.phi = phi
        self.vov = vol_of_vol
        self.base = base_daily_vol
        self.vrp = vrp
        self.iv_horizon = iv_horizon
        self._build()

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        days = self.cal.trading_days(self.start, self.end)
        if len(days) == 0:
            raise ValueError("SyntheticAdapter: no trading days in range")
        n = len(days)

        # AR(1) latent daily log-vol
        mu = np.log(self.base)
        logsig = np.empty(n)
        logsig[0] = mu
        for t in range(1, n):
            logsig[t] = mu + self.phi * (logsig[t - 1] - mu) + self.vov * rng.standard_normal()
        sigma_d = np.exp(logsig)  # true daily vol per session

        # Minute bars from per-minute gaussian returns scaled to the session's daily vol
        frames = []
        price = 100.0
        for i, day in enumerate(days):
            grid = self.cal.minute_grid(day)
            m = len(grid)
            sig_min = sigma_d[i] / np.sqrt(m)
            rets = sig_min * rng.standard_normal(m)
            closes = price * np.exp(np.cumsum(rets))
            opens = np.empty(m)
            opens[0] = price
            opens[1:] = closes[:-1]
            wiggle = np.abs(rng.standard_normal(m)) * sig_min * 0.1
            highs = np.maximum(opens, closes) * (1.0 + wiggle)
            lows = np.minimum(opens, closes) * (1.0 - wiggle)
            vols = rng.integers(100, 1000, m).astype(float)
            frames.append(
                pd.DataFrame(
                    {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
                    index=grid,
                )
            )
            price = closes[-1]
        bars = pd.concat(frames)
        bars.index.name = "ts"
        self._bars = validate_bars(bars)

        # IV surface stamped at each session close
        closes_ts = pd.DatetimeIndex([self.cal.session_close(d) for d in days])
        H = self.iv_horizon
        atm = np.empty(n)
        for i in range(n):
            fut = sigma_d[i + 1 : i + 1 + H]
            exp_vol = fut.mean() if len(fut) > 0 else sigma_d[i]
            atm[i] = exp_vol * (1.0 + self.vrp) * np.exp(0.05 * rng.standard_normal())
        iv = pd.DataFrame(
            {
                "atm_iv": atm,
                "skew": -0.5 * atm + 0.02 * rng.standard_normal(n),
                "term_slope": 0.1 * atm + 0.01 * rng.standard_normal(n),
                "vix": atm * np.sqrt(252.0) * 100.0,
            },
            index=closes_ts,
        )
        iv.index.name = "ts"
        self._iv = validate_iv(iv)
        self._sigma_d = pd.Series(sigma_d, index=closes_ts, name="sigma_d")

    def minute_bars(self, start, end) -> pd.DataFrame:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self._bars.loc[(self._bars.index >= s) & (self._bars.index <= e)]

    def iv_surface(self, start, end) -> pd.DataFrame:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return self._iv.loc[(self._iv.index >= s) & (self._iv.index <= e)]
