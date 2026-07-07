"""Build real hedging episodes: short the 1-day SPXW ATM straddle at D-1 close,
hedge with ES through day D, cash-settle |S_T - K| at the close.

Everything from data already on disk ($0): entry premium = real straddle mid from
the cached D-1 close-window chain (SPXW = PM-settled only; AM-settled SPX monthlies
would expire at the OPEN and corrupt V0); strike anchored at the chain's own parity
forward; path = ES minute bars mapped to a 5-min grid, overnight gap included as the
first (hedgeable-at-entry) step. Half-days skipped.

Output: artifacts/hedge_episodes.npz + hedge_meta.parquet
Usage:  python scripts/build_hedge_episodes.py --config configs/spx.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.data.databento import DatabentoAdapter

GRID_MIN = 5
STEPS = 78  # 09:35 .. 16:00


def entry_straddle(cache_dir, tag, expiry_day, es_ref):
    """Parity forward, ATM strike, straddle mid from the D-1 close chain (SPXW only)."""
    defs_p = os.path.join(cache_dir, f"spx_defs_{tag}.parquet")
    cbbo_p = os.path.join(cache_dir, f"spx_cbbo_{tag}.parquet")
    if not (os.path.exists(defs_p) and os.path.exists(cbbo_p)):
        return None
    defs = pd.read_parquet(defs_p).drop_duplicates(subset=["instrument_id"], keep="last")
    exp = pd.to_datetime(defs["expiration"])
    if exp.dt.tz is not None:
        exp = exp.dt.tz_convert("America/New_York").dt.tz_localize(None)
    defs = defs[(exp.dt.normalize() == expiry_day)
                & defs["raw_symbol"].str.startswith("SPXW")]
    if defs.empty:
        return None
    m = pd.read_parquet(cbbo_p).merge(
        defs[["instrument_id", "strike_price", "instrument_class"]], on="instrument_id")
    m = m[(m.bid_px_00 > 0) & np.isfinite(m.ask_px_00)]
    m["mid"] = (m.bid_px_00 + m.ask_px_00) / 2
    m["K"] = m.strike_price.astype(float)
    m = m[(m.K > es_ref * 0.98) & (m.K < es_ref * 1.02)]
    c = m[m.instrument_class == "C"].set_index("K")
    p = m[m.instrument_class == "P"].set_index("K")
    both = c.index.intersection(p.index)
    if len(both) < 3:
        return None
    F = float(np.median(c.loc[both, "mid"] - p.loc[both, "mid"] + both.values))
    K = float(both[np.argmin(np.abs(both.values - F))])
    V0 = float(c.loc[K, "mid"] + p.loc[K, "mid"])
    sprd = float((c.loc[K, "ask_px_00"] - c.loc[K, "bid_px_00"])
                 + (p.loc[K, "ask_px_00"] - p.loc[K, "bid_px_00"]))
    if not (np.isfinite(V0) and V0 > 1 and sprd / V0 < 0.5):
        return None
    return F, K, V0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    args = ap.parse_args()

    cfg = get_config(args.config)
    adapter = DatabentoAdapter(cfg)
    bars = adapter.minute_bars(cfg.start, cfg.end)
    day_groups = dict(tuple(bars.groupby(bars.index.normalize())))
    days = sorted(day_groups)

    tau0 = 1.0 / 252.0
    grid_times = [pd.Timedelta(hours=9, minutes=35) + pd.Timedelta(minutes=GRID_MIN * k)
                  for k in range(STEPS)]

    paths, meta = [], []
    for i in range(1, len(days)):
        d_entry, d_exp = days[i - 1], days[i]
        if adapter.cal.session_close(d_exp) != d_exp + pd.Timedelta(hours=16):
            continue  # half-day
        db = day_groups[d_exp]
        if len(db) < 300:
            continue
        es_close_prev = float(day_groups[d_entry].close.iloc[-1])
        ent = entry_straddle(adapter.cache_dir, d_entry.strftime("%Y%m%d"),
                             d_exp, es_close_prev)
        if ent is None:
            continue
        F, K, V0 = ent
        sigma0 = V0 / (np.sqrt(2 / np.pi) * F * np.sqrt(tau0))
        if not (0.03 < sigma0 < 2.0):
            continue
        # path: parity forward anchored at entry, moved by ES returns (asof 5-min grid)
        closes = db.close
        S = [F]
        ok = True
        for gt in grid_times:
            j = closes.index.searchsorted(d_exp + gt, side="right") - 1
            if j < 0:
                ok = False
                break
            S.append(F * float(closes.iloc[j]) / es_close_prev)
        if not ok:
            continue
        paths.append(S)
        meta.append({"date": d_exp, "K": K, "V0": V0, "S0": F, "sigma0": sigma0,
                     "year": d_exp.year})

    S = np.array(paths)
    md = pd.DataFrame(meta)
    rem = np.array([1.0] + [(390 - GRID_MIN * (k + 1)) / 390 for k in range(STEPS)])
    tau = np.where(np.arange(STEPS + 1) == 0, tau0, 0.75 * tau0 * rem)  # 25% overnight
    np.savez("artifacts/hedge_episodes.npz", S=S, tau=tau,
             K=md.K.values, V0=md.V0.values, S0=md.S0.values,
             sigma0=md.sigma0.values, year=md.year.values,
             date=md.date.values.astype("datetime64[D]").astype(str))
    md.to_parquet("artifacts/hedge_meta.parquet")
    print(f"episodes: {len(md)} | by year: {md.groupby('year').size().to_dict()}")
    print(f"V0/S0 median {(md.V0 / md.S0).median():.3%} | sigma0 (ann) median "
          f"{md.sigma0.median():.3f} | unhedged edge V0-|S_T-K| mean "
          f"{(md.V0.values - np.abs(S[:, -1] - md.K.values)).mean():+.2f} pts")


if __name__ == "__main__":
    main()
