"""0DTE terminal study: session table -> premium level -> predictability -> exhaustion.

Per session: parity-implied spot from the strip itself (F = median(C - P + K), no ES
basis), ATM straddle mid at T-15/-10/-5, realized |move| = |F_last - K| from the final
snapshot. Ladder: unconditional -> seasonal -> Engle-Sokalska-lite (day vol x diurnal
share x recent innovation) -> straddle (naive) -> straddle MZ -> blend. Fold-wise;
targets DON'T overlap, so inference is clean. LOFO exhaustion gate last.

Usage: python scripts/study_0dte.py --config configs/spx.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.data.databento import DatabentoAdapter

FLOOR = 0.5  # index points; |m| floor for log targets


def parity_spot(snap):
    """Median C - P + K across strikes with two-sided quotes."""
    c = snap[snap.instrument_class == "C"].set_index("strike_price")
    p = snap[snap.instrument_class == "P"].set_index("strike_price")
    both = c.index.intersection(p.index)
    if len(both) < 3:
        return np.nan
    cm = (c.loc[both, "bid_px_00"] + c.loc[both, "ask_px_00"]) / 2
    pm = (p.loc[both, "bid_px_00"] + p.loc[both, "ask_px_00"]) / 2
    return float(np.median(cm - pm + both.values))


def straddle_at(snap, K):
    r = {}
    for cls in ("C", "P"):
        row = snap[(snap.instrument_class == cls) & (snap.strike_price == K)]
        if row.empty or not (row.bid_px_00.iloc[0] > 0):
            return np.nan, np.nan
        r[cls] = (float(row.bid_px_00.iloc[0]) + float(row.ask_px_00.iloc[0])) / 2
        sp = float(row.ask_px_00.iloc[0]) - float(row.bid_px_00.iloc[0])
    return r["C"] + r["P"], sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    ap.add_argument("--delta", type=int, default=15, help="minutes before close")
    args = ap.parse_args()

    cfg = get_config(args.config)
    adapter = DatabentoAdapter(cfg)
    bars = adapter.minute_bars(cfg.start, cfg.end)
    panel = pd.read_parquet("artifacts/panel.parquet")

    rows = []
    days = adapter.cal.trading_days(cfg.start, cfg.end)
    for day in days:
        tag = day.strftime("%Y%m%d")
        f = os.path.join(adapter.cache_dir, f"{adapter.root_tag}0dte_{tag}.parquet")
        if not os.path.exists(f):
            continue
        strip = pd.read_parquet(f)
        if strip.empty:
            continue
        strip["ts_recv"] = pd.to_datetime(strip["ts_recv"], utc=True)
        close_utc = (adapter.cal.session_close(day).tz_localize("America/New_York")
                     .tz_convert("UTC"))
        t_entry = close_utc - pd.Timedelta(minutes=args.delta)

        def snap_at(t, width_min=2):
            w = strip[(strip.ts_recv <= t) & (strip.ts_recv > t - pd.Timedelta(minutes=width_min))]
            w = w.dropna(subset=["bid_px_00", "ask_px_00"])   # last COHERENT quote only
            return w.sort_values("ts_recv").groupby("instrument_id").tail(1)

        snap_e = snap_at(t_entry)
        snap_l = snap_at(close_utc + pd.Timedelta(minutes=1), width_min=3)
        if snap_e.empty or snap_l.empty or strip.ts_recv.max() < close_utc - pd.Timedelta(minutes=2):
            continue
        F0, F1 = parity_spot(snap_e), parity_spot(snap_l)
        if not (np.isfinite(F0) and np.isfinite(F1)):
            continue
        strikes = snap_e.strike_price.unique()
        K = strikes[np.argmin(np.abs(strikes - F0))]
        V, spread = straddle_at(snap_e, K)
        if not np.isfinite(V) or V <= 0 or spread / V > 0.5:
            continue

        # day context from ES bars (all strictly before t_entry, NY-naive)
        t_e_ny = t_entry.tz_convert("America/New_York").tz_localize(None)
        db = bars.loc[str(day.date())]
        db = db[db.index <= t_e_ny]
        if len(db) < 60:
            continue
        r_all = np.diff(np.log(db["close"].values))
        rv_day = np.sqrt(np.sum(r_all**2))
        rv_30 = np.sqrt(np.sum(r_all[-30:] ** 2))
        trend = np.log(db["close"].iloc[-1] / db["close"].iloc[0])

        t0 = adapter.cal.session_close(day)
        vix = float(panel.loc[t0, "feat_vix"]) if t0 in panel.index else np.nan
        rows.append({
            "date": day.normalize(), "K": K, "F0": F0, "offset": F0 - K,
            "V": V, "spread": spread, "m": F1 - F0, "m_abs_K": abs(F1 - K),
            "rv_day": rv_day, "rv_30": rv_30, "trend": trend, "vix": vix,
            "dow": day.dayofweek, "year": day.year,
        })

    df = pd.DataFrame(rows).set_index("date")
    df.to_parquet("artifacts/spx0dte_study.parquet")
    print(f"session table: {len(df)} rows -> artifacts/spx0dte_study.parquet")
    print(f"by year: {df.groupby('year').size().to_dict()}")
    print(f"median straddle spread/V: {(df.spread/df.V).median():.1%}")

    # ---- Q1: the premium ------------------------------------------------------------
    R = df["m_abs_K"] / df["V"]
    boot = [R.sample(len(R), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nQ1  E[|m|/V] = {R.mean():.3f}  (95% CI {lo:.3f}-{hi:.3f}; <1 = premium)")
    print("    by year: " + str({y: round(g.mean(), 3) for y, g in R.groupby(df.year)}))
    print("    by dow : " + str({d: round(g.mean(), 3) for d, g in R.groupby(df.dow)}))

    # ---- Q2: ladder (target log(|m|+floor)), 4 time folds ---------------------------
    from sklearn.linear_model import Ridge

    y = np.log(df["m_abs_K"].values + FLOOR)
    n = len(df)
    fold = np.minimum((np.arange(n) * 4) // n, 3)
    feats = {
        "seasonal":  pd.get_dummies(df["dow"]).values.astype(float),
        "es_lite":   np.column_stack([np.log(df.rv_day + 1e-6), np.log(df.rv_30 + 1e-6),
                                      np.abs(df.trend), pd.get_dummies(df.dow).values]),
        "straddle_mz": np.log(df[["V"]].values),
        "blend":     np.column_stack([np.log(df[["V"]].values),
                                      np.log(df.rv_30.values + 1e-6),
                                      np.log(df.rv_day.values + 1e-6),
                                      np.abs(df.trend.values)]),
    }
    print("\nQ2  OOS MSE of log(|m|+0.5), 4 folds (lower better):")
    mse = {}
    base_preds = {}
    for name, X in feats.items():
        p = np.full(n, np.nan)
        for k in range(4):
            m = Ridge(alpha=1.0).fit(X[fold != k], y[fold != k])
            p[fold == k] = m.predict(X[fold == k])
        mse[name] = np.mean((y - p) ** 2)
        base_preds[name] = p
    unc = np.full(n, np.nan)
    for k in range(4):
        unc[fold == k] = y[fold != k].mean()
    mse["unconditional"] = np.mean((y - unc) ** 2)
    naive = np.log(0.8 * df["V"].values + FLOOR)   # straddle scaled to E|m| (Gaussian)
    mse["straddle_naive"] = np.mean((y - naive) ** 2)
    for k, v in sorted(mse.items(), key=lambda t: t[1]):
        print(f"    {k:15s} {v:.4f}")

    # ---- Q3: LOFO exhaustion on the best rung ---------------------------------------
    best = min((k for k in feats), key=lambda k: mse[k])
    e = y - base_preds[best]
    Xall = np.column_stack([feats["blend"], pd.get_dummies(df.dow).values,
                            df[["vix"]].fillna(df.vix.median()).values,
                            df[["offset"]].abs().values])
    print(f"\nQ3  LOFO residual R^2 on best rung ({best}):")
    vals = []
    for k in range(4):
        m = Ridge(alpha=1.0).fit(Xall[fold != k], e[fold != k])
        base = np.mean((e[fold == k] - e[fold != k].mean()) ** 2)
        vals.append(1 - np.mean((e[fold == k] - m.predict(Xall[fold == k])) ** 2) / base)
    v = np.array(vals)
    print(f"    folds {np.round(v, 4).tolist()}  mean {v.mean():+.4f}"
          f"  -> {'GATE OPEN' if (v > 0).all() and v.mean() > 0.01 else 'gate closed'}")


if __name__ == "__main__":
    main()
