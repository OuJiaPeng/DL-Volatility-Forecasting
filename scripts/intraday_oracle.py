"""Intraday oracle: does ANYTHING predict next-30-min RV beyond diurnal x persistence?

Protocol (amended rule: oracle BEFORE any model):
  1. Build intraday panel from owned minute bars. PIT by construction: features from
     returns with ts <= t0, target = RV over (t0, t0+block]. Origins on a 30-min grid
     10:00..15:30; targets never cross a day.
  2. Ladder (fold-wise ridge): unconditional -> time-of-day -> +dow -> ES-lite
     (diurnal x persistence) -> diurnal-HAR + prior-close IV/VIX (full linear)
     -> GBT on the same features (nonlinearity certificate).
  3. LOFO exhaustion gate on the best linear rung's residuals, ridge AND GBT.
Folds are 4 contiguous DAY blocks with a 1-day purge at each boundary.

Usage: python scripts/intraday_oracle.py --config configs/spx.yaml [--block 30]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.data.databento import DatabentoAdapter

EPS = 1e-5  # log floor, well below a typical 30-min RV (~1e-3 in log-return units)


def build_panel(cfg, adapter, daily_panel, block_min):
    bars = adapter.minute_bars(cfg.start, cfg.end)
    day_groups = dict(tuple(bars.groupby(bars.index.normalize())))
    days = sorted(day_groups)

    drv = pd.Series({d: float(np.sqrt(np.nansum(np.diff(np.log(day_groups[d].close.values)) ** 2)))
                     for d in days})
    rv_d_prior = drv.shift(1)
    rv_w = drv.pow(2).rolling(5).mean().pow(0.5).shift(1)
    rv_m = drv.pow(2).rolling(21).mean().pow(0.5).shift(1)
    last_close = pd.Series({d: float(day_groups[d].close.iloc[-1]) for d in days})

    blk = pd.Timedelta(minutes=block_min)
    rows, n_noiv = [], 0
    for i, d in enumerate(days):
        if i < 21 or not np.isfinite(rv_m.loc[d]):
            continue
        # prior-session close IV/VIX: last daily-panel row strictly before today's open
        t_iv = daily_panel.index.asof(d + pd.Timedelta(hours=9, minutes=25))
        if pd.isna(t_iv) or (d - t_iv) > pd.Timedelta(days=5):
            n_noiv += 1
            continue
        iv1 = float(daily_panel.loc[t_iv, "feat_atm_iv_1"])
        vix = float(daily_panel.loc[t_iv, "feat_vix"])
        if not (np.isfinite(iv1) and np.isfinite(vix)):
            n_noiv += 1
            continue

        db = day_groups[d]
        c = np.log(db.close.values)
        r = np.diff(c)
        rts = db.index[1:]  # returns stamped at bar END
        vol = db.volume.values[1:]
        gap = c[0] - np.log(last_close.iloc[i - 1])
        close_d = adapter.cal.session_close(d)

        t0 = d + pd.Timedelta(hours=10)
        while t0 + blk <= close_d:
            h30 = (rts > t0 - blk) & (rts <= t0)
            h60 = (rts > t0 - 2 * blk) & (rts <= t0)
            hop = rts <= t0
            fut = (rts > t0) & (rts <= t0 + blk)
            if h30.sum() >= block_min - 10 and fut.sum() >= block_min - 10 and hop.sum() >= 25:
                rows.append({
                    "t0": t0, "day": d, "dow": d.dayofweek,
                    "block": int((t0 - d).total_seconds() // 60),
                    "rv30": np.sqrt(np.sum(r[h30] ** 2)), "rv60": np.sqrt(np.sum(r[h60] ** 2)),
                    "rvop": np.sqrt(np.sum(r[hop] ** 2)), "rmax": np.abs(r[h30]).max(),
                    "ret30": r[h30].sum(), "trend": r[hop].sum(), "gap": gap,
                    "v30": np.log1p(np.sum(vol[h30])),
                    "rv_d": rv_d_prior.loc[d], "rv_w": rv_w.loc[d], "rv_m": rv_m.loc[d],
                    "iv1": iv1, "vix": vix,
                    "y": np.log(np.sqrt(np.sum(r[fut] ** 2)) + EPS),
                })
            t0 += blk
    df = pd.DataFrame(rows).set_index("t0")
    print(f"panel: {len(df)} origins over {df.day.nunique()} days "
          f"({n_noiv} days dropped: no prior-close IV)")
    return df


def day_folds(df, n_folds=4, purge_days=1):
    """Contiguous day blocks; returns per-row fold id + train mask fn with purge."""
    uniq = np.array(sorted(df.day.unique()))
    dpos = {d: j for j, d in enumerate(uniq)}
    dfold = np.minimum(np.arange(len(uniq)) * n_folds // len(uniq), n_folds - 1)
    fold = df.day.map(lambda d: dfold[dpos[d]]).values
    pos = df.day.map(dpos).values

    def train_mask(k):
        kpos = np.where(dfold == k)[0]
        lo, hi = kpos.min(), kpos.max()
        return (fold != k) & ((pos < lo - purge_days) | (pos > hi + purge_days))

    return fold, train_mask


def fit_oos(X, y, fold, train_mask, model_fn):
    p = np.full(len(y), np.nan)
    for k in range(4):
        tr = train_mask(k)
        te = fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = model_fn().fit((X[tr] - mu) / sd, y[tr])
        p[te] = m.predict((X[te] - mu) / sd)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    ap.add_argument("--block", type=int, default=30)
    args = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor as GBT

    cfg = get_config(args.config)
    adapter = DatabentoAdapter(cfg)
    pname = getattr(cfg, "name", "panel")
    tag = "spx" if pname == "panel" else pname
    daily_panel = pd.read_parquet(f"artifacts/{pname}.parquet")

    df = build_panel(cfg, adapter, daily_panel, args.block)
    df.to_parquet(f"artifacts/intraday_{tag}.parquet")

    y = df["y"].values
    fold, train_mask = day_folds(df)
    L = lambda s: np.log(df[s].values + EPS)
    tod = pd.get_dummies(df.block).values.astype(float)
    dow = pd.get_dummies(df.dow).values.astype(float)
    rungs = {
        "tod":      tod,
        "tod_dow":  np.column_stack([tod, dow]),
        "es_lite":  np.column_stack([tod, L("rv30"), L("rvop"), L("rv_d")]),
        "dhar_iv":  np.column_stack([tod, dow, L("rv30"), L("rv60"), L("rvop"),
                                     L("rv_d"), L("rv_w"), L("rv_m"),
                                     L("iv1"), np.log(df.vix.values),
                                     df[["ret30", "trend", "gap"]].values,
                                     np.abs(df.gap.values)[:, None],
                                     L("rmax"), df[["v30"]].values]),
    }
    rungs["gbt_full"] = rungs["dhar_iv"]

    unc = np.full(len(y), np.nan)
    for k in range(4):
        unc[fold == k] = y[train_mask(k)].mean()
    mse_unc = np.mean((y - unc) ** 2)

    print(f"\nladder: OOS MSE of log(RV_{args.block}m + eps), 4 day-folds | R^2 vs unconditional")
    print(f"    {'unconditional':14s} {mse_unc:.4f}")
    preds, mse = {}, {}
    for name, X in rungs.items():
        model = (lambda: GBT(random_state=0, max_iter=400, learning_rate=0.06)) \
            if name == "gbt_full" else (lambda: Ridge(alpha=1.0))
        p = fit_oos(X, y, fold, train_mask, model)
        preds[name], mse[name] = p, np.mean((y - p) ** 2)
        print(f"    {name:14s} {mse[name]:.4f}   R^2 {1 - mse[name] / mse_unc:+.3f}")

    lin_best = min((k for k in rungs if k != "gbt_full"), key=lambda k: mse[k])
    fmse = lambda p, k: np.mean((y[fold == k] - p[fold == k]) ** 2)
    dlt = [1 - fmse(preds["gbt_full"], k) / fmse(preds[lin_best], k) for k in range(4)]
    print(f"\nnonlinearity certificate: GBT vs {lin_best}, per-fold MSE gain "
          f"{np.round(dlt, 4).tolist()}  -> {'NONLINEAR' if np.mean(dlt) > 0.01 and sum(d > 0 for d in dlt) >= 3 else 'linear suffices'}")

    e = y - preds[lin_best]
    Xall = rungs["dhar_iv"]
    print(f"\nLOFO exhaustion gate on {lin_best} residuals:")
    for gname, gfn in [("ridge", lambda: Ridge(alpha=1.0)),
                       ("gbt", lambda: GBT(random_state=0, max_iter=300, learning_rate=0.06))]:
        vals = []
        for k in range(4):
            tr, te = train_mask(k), fold == k
            mu, sd = Xall[tr].mean(0), Xall[tr].std(0) + 1e-9
            m = gfn().fit((Xall[tr] - mu) / sd, e[tr])
            base = np.mean((e[te] - e[tr].mean()) ** 2)
            vals.append(1 - np.mean((e[te] - m.predict((Xall[te] - mu) / sd)) ** 2) / base)
        v = np.array(vals)
        print(f"    {gname:6s} folds {np.round(v, 4).tolist()}  mean {v.mean():+.4f}"
              f"  -> {'GATE OPEN' if (v > 0).all() and v.mean() > 0.01 else 'gate closed'}")


if __name__ == "__main__":
    main()
