"""Solve a daily IV-surface summary from raw option-chain quotes (no vendor analytics).

SPX options are European, so the pipeline is exact and self-contained:

  1. Per expiry, put-call parity regression across strikes,  C - P = D (F - K),
     jointly recovers the discount factor D (slope = -D) and forward F
     (intercept / D) — no external rates or dividend inputs needed.
  2. Black-76 inversion of OTM mid quotes gives per-contract implied vol.
  3. Linear interpolation of the OTM smile at K = F gives ATM IV per expiry;
     the 25-delta put/call wings give the skew.
  4. Linear interpolation in TOTAL VARIANCE (sigma^2 * tau) across expiries gives
     tenor-matched ATM IV at each forecast horizon.

Units: everything here is ANNUALIZED Black vol; the adapter converts to the
panel's daily-vol convention (divide by sqrt(252)). Year fractions use ACT/365
for expiries and h/252 for session horizons — a ~2% convention mismatch that is
acceptable at daily cadence and documented here rather than hidden.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252.0
MIN_QUOTES_PER_EXPIRY = 5


# --- Black-76 ----------------------------------------------------------------------
def black76_price(F, K, tau, sigma, is_call: bool, D: float = 1.0) -> float:
    if tau <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return D * intrinsic
    st = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    d2 = d1 - st
    if is_call:
        return D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_delta(F, K, tau, sigma, is_call: bool) -> float:
    """Forward (undiscounted) delta."""
    if tau <= 0 or sigma <= 0:
        step = 1.0 if F > K else 0.0
        return step if is_call else step - 1.0
    st = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def black76_iv(price, F, K, tau, is_call: bool, D: float = 1.0) -> float:
    """Implied Black vol via bracketed root-find; NaN if the quote is out of bounds."""
    intrinsic = black76_price(F, K, tau, 1e-12, is_call, D)
    upper = black76_price(F, K, tau, 5.0, is_call, D)
    if not (intrinsic < price < upper):
        return float("nan")
    f = lambda s: black76_price(F, K, tau, s, is_call, D) - price
    try:
        return float(brentq(f, 1e-4, 5.0, xtol=1e-8))
    except ValueError:
        return float("nan")


# --- Parity ------------------------------------------------------------------------
def parity_forward(strikes: np.ndarray, call_mids: np.ndarray, put_mids: np.ndarray):
    """Regress (C - P) on K to recover (discount D, forward F). Needs >= 2 strikes."""
    K = np.asarray(strikes, dtype=float)
    y = np.asarray(call_mids, dtype=float) - np.asarray(put_mids, dtype=float)
    if len(K) < 2:
        raise ValueError("parity_forward needs at least 2 strikes with both C and P quotes")
    slope, intercept = np.polyfit(K, y, 1)
    D = -slope
    if D <= 0 or D > 1.2:
        raise ValueError(f"parity regression gave implausible discount D={D:.4f}")
    F = intercept / D
    return float(D), float(F)


# --- Per-expiry solve --------------------------------------------------------------
def solve_expiry(chain: pd.DataFrame, tau: float):
    """Solve one expiry's slice of the chain.

    ``chain`` columns: strike, right ('C'/'P'), bid, ask. Returns a dict with
    forward, discount, atm_iv (annualized), skew_25d, n_used — or None if the
    slice is too thin/degenerate to trust.
    """
    df = chain.copy()
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]
    if len(df) < MIN_QUOTES_PER_EXPIRY:
        return None
    df["mid"] = 0.5 * (df["bid"] + df["ask"])

    # real chains carry duplicate strikes per expiry date (SPX AM-settled and SPXW
    # PM-settled roots share calendar expiries) — keep the tightest quote per
    # (right, strike) so the parity join below is unique
    df["_spread"] = df["ask"] - df["bid"]
    df = df.sort_values("_spread").drop_duplicates(subset=["right", "strike"], keep="first")

    calls = df[df["right"] == "C"].set_index("strike")["mid"]
    puts = df[df["right"] == "P"].set_index("strike")["mid"]
    both = calls.index.intersection(puts.index).sort_values()
    if len(both) < 2:
        return None

    # restrict the parity regression to strikes near the forward: deep wings quote
    # with huge spreads and junk mids that would otherwise dominate the fit.
    # F0 = strike where |C - P| is smallest (the classic ATM locator).
    gap = (calls.loc[both] - puts.loc[both]).abs()
    F0 = float(gap.idxmin())
    for band in (0.10, 0.20, None):
        sel = both if band is None else both[(both >= F0 * (1 - band)) & (both <= F0 * (1 + band))]
        if len(sel) >= 2:
            break
    try:
        D, F = parity_forward(sel.values, calls.loc[sel].values, puts.loc[sel].values)
    except ValueError:
        return None

    # invert OTM quotes only (puts below the forward, calls above) — the liquid side
    otm = df[((df["right"] == "P") & (df["strike"] <= F)) |
             ((df["right"] == "C") & (df["strike"] > F))].copy()
    otm["iv"] = [
        black76_iv(r.mid, F, r.strike, tau, r.right == "C", D) for r in otm.itertuples()
    ]
    otm = otm.dropna(subset=["iv"]).sort_values("strike")
    if len(otm) < 3:
        return None
    # the invertible strikes must BRACKET the forward — otherwise np.interp would
    # silently flat-extrapolate a wing IV and report it as "ATM" (verified failure
    # mode: one-sided chains after a gap move mis-mark ATM by several vol points)
    if not (otm["strike"].min() <= F <= otm["strike"].max()):
        return None

    # ATM: linear interpolation of the OTM smile at K = F
    atm_iv = float(np.interp(F, otm["strike"].values, otm["iv"].values))

    # 25-delta wings
    otm["delta"] = [
        black76_delta(F, r.strike, tau, r.iv, r.right == "C") for r in otm.itertuples()
    ]
    puts25 = otm[otm["right"] == "P"]
    calls25 = otm[otm["right"] == "C"]
    skew = float("nan")
    if len(puts25) and len(calls25):
        p = puts25.iloc[(puts25["delta"] + 0.25).abs().argmin()]
        c = calls25.iloc[(calls25["delta"] - 0.25).abs().argmin()]
        # only call it "25-delta" if the nearest quoted wings are actually near 25d;
        # sparse chains would otherwise substitute near-ATM strikes and understate skew
        if 0.15 <= abs(p["delta"]) <= 0.35 and 0.15 <= abs(c["delta"]) <= 0.35:
            skew = float(p["iv"] - c["iv"])

    return {"forward": F, "discount": D, "atm_iv": atm_iv, "skew_25d": skew,
            "n_used": int(len(otm))}


# --- Tenor interpolation -----------------------------------------------------------
def tenor_interpolate(taus: np.ndarray, atm_ivs: np.ndarray, target_taus) -> np.ndarray:
    """Interpolate ATM IV at target year-fractions, linear in total variance.

    Clamps outside the quoted range (flat extrapolation of variance slope is
    deliberately avoided — short-dated vol is where extrapolation lies to you).
    """
    taus = np.asarray(taus, dtype=float)
    ivs = np.asarray(atm_ivs, dtype=float)
    order = np.argsort(taus)
    taus, ivs = taus[order], ivs[order]
    w = ivs**2 * taus  # total variance
    out = []
    for t in np.atleast_1d(target_taus):
        t_c = float(np.clip(t, taus[0], taus[-1]))
        w_t = float(np.interp(t_c, taus, w))
        out.append(math.sqrt(max(w_t, 0.0) / t_c))
    return np.asarray(out)


# --- Session-level summary ---------------------------------------------------------
def surface_summary(chain: pd.DataFrame, session_date, horizons_sessions) -> dict | None:
    """One session's chain -> tenor-matched surface summary (annualized units).

    ``chain`` columns: expiry (datetime-like), strike, right, bid, ask.
    Returns atm_iv_{h} per horizon, atm_iv (longest-horizon default), skew
    (nearest-monthly 25d), term_slope (long minus short tenor ATM) — or None.
    """
    session_date = pd.Timestamp(session_date).normalize()
    per_expiry = []
    for expiry, grp in chain.groupby("expiry"):
        cal_days = (pd.Timestamp(expiry).normalize() - session_date).days
        if cal_days < 1:
            continue
        tau = cal_days / 365.0
        solved = solve_expiry(grp, tau)
        if solved is not None:
            solved["tau"] = tau
            per_expiry.append(solved)
    if len(per_expiry) < 2:
        return None

    # dedupe identical taus (SPX AM- and SPXW PM-settled series share expiry dates):
    # keep the slice solved from more quotes
    per_expiry.sort(key=lambda s: (s["tau"], -s["n_used"]))
    per_expiry = [s for i, s in enumerate(per_expiry)
                  if i == 0 or s["tau"] != per_expiry[i - 1]["tau"]]
    if len(per_expiry) < 2:
        return None
    taus = np.array([s["tau"] for s in per_expiry])
    ivs = np.array([s["atm_iv"] for s in per_expiry])

    horizons = sorted(horizons_sessions)
    target_taus = [h / TRADING_DAYS for h in horizons]
    tenor_ivs = tenor_interpolate(taus, ivs, target_taus)

    # skew from the expiry nearest ~21 sessions among those with a FINITE 25d
    # estimate; a session with no valid skew anywhere is a bad-data day — skip it
    # rather than emit a silently-zero skew feature
    finite = [s for s in per_expiry if np.isfinite(s["skew_25d"])]
    if not finite:
        return None
    monthly = min(finite, key=lambda s: abs(s["tau"] - 21.0 / TRADING_DAYS))

    out = {f"atm_iv_{h}": float(v) for h, v in zip(horizons, tenor_ivs)}
    out["atm_iv"] = out[f"atm_iv_{max(horizons)}"]
    out["skew"] = float(monthly["skew_25d"])
    out["term_slope"] = out[f"atm_iv_{max(horizons)}"] - out[f"atm_iv_{min(horizons)}"]
    return out
