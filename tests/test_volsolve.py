"""IV solver: generate Black-76 chains with KNOWN vol/forward/discount, recover them."""
import numpy as np
import pandas as pd
import pytest

from volforecast.data import volsolve as vs


def make_chain(F=5000.0, D=0.998, tau=30 / 365, atm=0.18, slope=-0.10, spread=0.02):
    """Synthetic SPX-like expiry slice with a linear-in-log-moneyness smile."""
    strikes = np.arange(0.85, 1.16, 0.025) * F
    rows = []
    for K in strikes:
        sigma = atm + slope * np.log(K / F)  # negative slope: puts rich, calls cheap
        for right in ("C", "P"):
            mid = vs.black76_price(F, K, tau, sigma, right == "C", D)
            half = spread / 2
            rows.append({"strike": K, "right": right,
                         "bid": max(mid - half, 0.01), "ask": mid + half})
    return pd.DataFrame(rows)


def test_black76_iv_round_trip():
    F, K, tau, D = 5000.0, 5100.0, 45 / 365, 0.997
    for sigma in (0.08, 0.18, 0.45):
        for is_call in (True, False):
            price = vs.black76_price(F, K, tau, sigma, is_call, D)
            assert np.isclose(vs.black76_iv(price, F, K, tau, is_call, D), sigma, atol=1e-6)


def test_black76_iv_out_of_bounds_is_nan():
    assert np.isnan(vs.black76_iv(1e-9, 5000, 5000, 0.1, True, 1.0))   # below intrinsic
    assert np.isnan(vs.black76_iv(1e9, 5000, 5000, 0.1, True, 1.0))    # above max


def test_parity_recovers_forward_and_discount():
    F, D, tau, sigma = 5000.0, 0.995, 60 / 365, 0.2
    K = np.arange(4500, 5500, 100.0)
    C = np.array([vs.black76_price(F, k, tau, sigma, True, D) for k in K])
    P = np.array([vs.black76_price(F, k, tau, sigma, False, D) for k in K])
    D_hat, F_hat = vs.parity_forward(K, C, P)
    assert np.isclose(D_hat, D, atol=1e-6)
    assert np.isclose(F_hat, F, rtol=1e-6)


def test_solve_expiry_recovers_atm_and_skew_sign():
    F, atm, tau = 5000.0, 0.18, 30 / 365
    chain = make_chain(F=F, atm=atm, tau=tau, slope=-0.10)
    out = vs.solve_expiry(chain, tau)
    assert out is not None
    assert np.isclose(out["forward"], F, rtol=2e-3)
    assert np.isclose(out["atm_iv"], atm, atol=0.005)
    assert out["skew_25d"] > 0  # negative smile slope => 25d put IV above 25d call IV


def test_solve_expiry_rejects_thin_chain():
    chain = make_chain().head(3)
    assert vs.solve_expiry(chain, 30 / 365) is None


def test_tenor_interpolation_flat_and_monotone():
    taus = np.array([5, 21, 63]) / 252
    flat = vs.tenor_interpolate(taus, [0.2, 0.2, 0.2], [1 / 252, 10 / 252, 21 / 252])
    assert np.allclose(flat, 0.2, atol=1e-9)
    rising = vs.tenor_interpolate(taus, [0.15, 0.18, 0.22], [5 / 252, 21 / 252, 63 / 252])
    assert rising[0] < rising[1] < rising[2]
    # clamped outside the quoted range
    clamped = vs.tenor_interpolate(taus, [0.15, 0.18, 0.22], [1 / 252])
    assert np.isclose(clamped[0], 0.15, atol=1e-9)


def test_surface_summary_end_to_end():
    session = pd.Timestamp("2024-03-01")
    frames = []
    for cal_days, atm in ((7, 0.15), (30, 0.18), (90, 0.21)):
        c = make_chain(atm=atm, tau=cal_days / 365)
        c["expiry"] = session + pd.Timedelta(days=cal_days)
        frames.append(c)
    chain = pd.concat(frames, ignore_index=True)
    out = vs.surface_summary(chain, session, [1, 5, 10, 21])
    assert out is not None
    for h in (1, 5, 10, 21):
        assert 0.10 < out[f"atm_iv_{h}"] < 0.25
    assert out["atm_iv_1"] < out["atm_iv_21"]     # upward term structure recovered
    assert out["term_slope"] > 0
    assert out["skew"] > 0


def test_one_sided_chain_rejected_not_extrapolated():
    """Strikes that don't bracket the forward must return None, not a wing IV as 'ATM'."""
    F, tau = 5000.0, 30 / 365
    chain = make_chain(F=F, tau=tau)
    chain = chain[chain["strike"] <= 0.95 * F]  # one-sided after a gap move
    assert vs.solve_expiry(chain, tau) is None


def test_duplicate_strikes_from_two_roots():
    """SPX + SPXW list the same expiry/strikes; solver must dedupe, not crash."""
    F, atm, tau = 5000.0, 0.18, 30 / 365
    a = make_chain(F=F, atm=atm, tau=tau, spread=0.02)
    b = make_chain(F=F, atm=atm, tau=tau, spread=0.10)  # same contracts, wider quotes
    out = vs.solve_expiry(pd.concat([a, b], ignore_index=True), tau)
    assert out is not None
    assert np.isclose(out["atm_iv"], atm, atol=0.005)  # tightest quotes win


def test_narrow_chain_gives_nan_skew_not_fake_25d():
    """Wings only near ATM (~41-43 delta) must not be labelled 25-delta skew."""
    F, tau, atm, D = 5000.0, 30 / 365, 0.18, 0.998
    rows = []
    for m in (0.99, 1.00, 1.01):
        K = m * F
        for right in ("C", "P"):
            mid = vs.black76_price(F, K, tau, atm, right == "C", D)
            rows.append({"strike": K, "right": right, "bid": mid - 0.01, "ask": mid + 0.01})
    out = vs.solve_expiry(pd.DataFrame(rows), tau)
    assert out is not None
    assert np.isnan(out["skew_25d"])


def test_surface_summary_horizon_order_invariant():
    session = pd.Timestamp("2024-03-01")
    frames = []
    for cal_days, atm in ((7, 0.15), (30, 0.18), (90, 0.21)):
        c = make_chain(atm=atm, tau=cal_days / 365)
        c["expiry"] = session + pd.Timedelta(days=cal_days)
        frames.append(c)
    chain = pd.concat(frames, ignore_index=True)
    a = vs.surface_summary(chain, session, [1, 5, 10, 21])
    b = vs.surface_summary(chain, session, [21, 1, 5, 10])
    assert np.isclose(a["atm_iv"], b["atm_iv"])
    assert np.isclose(a["term_slope"], b["term_slope"])
    assert a["term_slope"] > 0


def test_naive_iv_uses_tenor_columns_when_present():
    from volforecast.baselines import NaiveIV

    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="t0")
    panel = pd.DataFrame({
        "feat_atm_iv": [0.010, 0.011],
        "feat_atm_iv_1": [0.008, 0.009],
        "feat_atm_iv_5": [0.009, 0.010],
        "feat_atm_iv_10": [0.010, 0.011],
        "feat_atm_iv_21": [0.011, 0.012],
    }, index=idx)
    f = NaiveIV([1, 5, 10, 21]).fit(panel, idx)
    pred = f.predict(panel, idx)
    assert np.allclose(pred[:, 0], panel["feat_atm_iv_1"].values)   # tenor-matched
    assert np.allclose(pred[:, 3], panel["feat_atm_iv_21"].values)
    # fallback: no tenor columns -> flat broadcast of the single ATM column
    flat_panel = panel[["feat_atm_iv"]]
    pred_flat = NaiveIV([1, 5, 10, 21]).fit(flat_panel, idx).predict(flat_panel, idx)
    assert np.allclose(pred_flat[:, 0], pred_flat[:, 3])
