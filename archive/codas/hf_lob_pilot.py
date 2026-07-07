"""HF ES top-of-book volatility pilot: cost -> pull -> panel -> oracle.

This is the cheap gate before buying dense L2/L3 data. It uses Databento CME
Globex sampled schemas:

  * bbo-1s    top-of-book quote state once per second
  * ohlcv-1s  one-second trade bars

Target: next 5/15/30-minute realized volatility on 30-second origins.
Features: recent RV, volume, spread, top-of-book imbalance, microprice pressure,
and time-of-day. Evaluation is day-block OOS with a 1-day purge.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET = "GLBX.MDP3"
SYMBOL = "ES.c.0"
STYPE = "continuous"
SCHEMAS = ("bbo-1s", "ohlcv-1s")
NY = "America/New_York"
EPS = 1e-12


def load_key() -> None:
    """Populate DATABENTO_API_KEY from .env if needed."""
    if os.getenv("DATABENTO_API_KEY"):
        return
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DATABENTO_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ["DATABENTO_API_KEY"] = key
            return


def client():
    load_key()
    import databento as db

    return db.Historical()


def business_days(start: str, n_days: int) -> list[pd.Timestamp]:
    # Candidate weekdays. Exchange holidays come back as empty/zero-cost and are skipped.
    return list(pd.bdate_range(start, periods=int(n_days) + 20))[: int(n_days) + 20]


def rth_utc(day: pd.Timestamp) -> tuple[str, str]:
    d = pd.Timestamp(day).normalize()
    start = pd.Timestamp(
        year=d.year, month=d.month, day=d.day, hour=9, minute=30, tz=NY
    ).tz_convert("UTC")
    end = pd.Timestamp(
        year=d.year, month=d.month, day=d.day, hour=16, minute=0, tz=NY
    ).tz_convert("UTC")
    return start.isoformat(), end.isoformat()


def day_tag(day: pd.Timestamp) -> str:
    return pd.Timestamp(day).strftime("%Y%m%d")


def cache_path(schema: str, day: pd.Timestamp) -> Path:
    return ROOT / "data_cache" / f"hf_es_{schema}_{day_tag(day)}.parquet"


def cost_days(days: list[pd.Timestamp], schemas=SCHEMAS, verbose: bool = False) -> pd.DataFrame:
    c = client()
    rows = []
    for day in days:
        start, end = rth_utc(day)
        row = {"day": pd.Timestamp(day).date()}
        total = 0.0
        for schema in schemas:
            try:
                val = float(
                    c.metadata.get_cost(
                        dataset=DATASET,
                        symbols=[SYMBOL],
                        stype_in=STYPE,
                        schema=schema,
                        start=start,
                        end=end,
                    )
                )
            except Exception as e:
                print(f"[cost-error] {day.date()} {schema}: {type(e).__name__}: {e}", flush=True)
                val = np.nan
            row[schema] = val
            total += 0.0 if not np.isfinite(val) else val
        row["total"] = total
        rows.append(row)
        if verbose:
            bits = " ".join(
                f"{s}=${row[s]:.4f}" if np.isfinite(row[s]) else f"{s}=nan"
                for s in schemas
            )
            print(f"[cost] {day.date()} {bits} total=${total:.4f}", flush=True)
    return pd.DataFrame(rows)


def pull_days(
    days: list[pd.Timestamp],
    max_cost: float,
    refresh: bool = False,
    costs: pd.DataFrame | None = None,
) -> None:
    costs = cost_days(days, verbose=True) if costs is None else costs
    # Keep only sessions that appear to have data.
    live = costs[costs["total"].fillna(0) > 0].copy()
    total = float(live["total"].sum())
    print(f"estimated pull cost: ${total:.4f} across {len(live)} nonzero sessions", flush=True)
    if total > max_cost:
        raise SystemExit(f"abort: estimate ${total:.2f} exceeds --max-cost ${max_cost:.2f}")

    c = client()
    (ROOT / "data_cache").mkdir(exist_ok=True)
    pulled = 0
    for day in pd.to_datetime(live["day"]):
        start, end = rth_utc(day)
        for schema in SCHEMAS:
            out = cache_path(schema, day)
            if out.exists() and not refresh:
                continue
            print(f"[pull] {schema} {day.date()} -> {out.name}", flush=True)
            data = c.timeseries.get_range(
                dataset=DATASET,
                symbols=[SYMBOL],
                stype_in=STYPE,
                schema=schema,
                start=start,
                end=end,
            )
            df = data.to_df()
            if df.empty:
                print(f"[skip] empty {schema} {day.date()}", flush=True)
                continue
            df.index = pd.DatetimeIndex(df.index).tz_convert(NY).tz_localize(None)
            df.index.name = "ts"
            keep = [c for c in df.columns if c != "symbol"]
            df[keep].to_parquet(out)
            pulled += 1
    print(f"pull complete: wrote/updated {pulled} files", flush=True)


def _rolling_sum_future(x: pd.Series, seconds: int) -> pd.Series:
    # Sum x over (t, t+seconds], aligned at t.
    return x.shift(-1).iloc[::-1].rolling(seconds, min_periods=seconds).sum().iloc[::-1]


def build_panel(start: str, n_days: int, stride: int = 30) -> pd.DataFrame:
    rows = []
    for day in business_days(start, n_days):
        bbo_f = cache_path("bbo-1s", day)
        ohlc_f = cache_path("ohlcv-1s", day)
        if not (bbo_f.exists() and ohlc_f.exists()):
            continue
        bbo = pd.read_parquet(bbo_f)
        bar = pd.read_parquet(ohlc_f)
        if bbo.empty or bar.empty:
            continue

        idx = pd.date_range(
            pd.Timestamp(day.date()).tz_localize(NY).tz_localize(None)
            + pd.Timedelta(hours=9, minutes=30),
            pd.Timestamp(day.date()).tz_localize(NY).tz_localize(None)
            + pd.Timedelta(hours=16),
            freq="1s",
            inclusive="left",
        )
        bbo = bbo.reindex(idx).ffill()
        bar = bar.reindex(idx).ffill()
        px = bar["close"].astype(float).ffill()
        r = np.log(px).diff().fillna(0.0)
        r2 = r * r

        bid = bbo["bid_px_00"].astype(float)
        ask = bbo["ask_px_00"].astype(float)
        bsz = bbo["bid_sz_00"].astype(float)
        asz = bbo["ask_sz_00"].astype(float)
        mid = 0.5 * (bid + ask)
        depth = bsz + asz
        imb = (bsz - asz) / depth.replace(0.0, np.nan)
        micro = (ask * bsz + bid * asz) / depth.replace(0.0, np.nan)
        spread = ask - bid
        vol = bar["volume"].astype(float).fillna(0.0)
        hl = np.log(bar["high"].astype(float).clip(lower=EPS)) - np.log(
            bar["low"].astype(float).clip(lower=EPS)
        )
        bid_prev, ask_prev = bid.shift(1), ask.shift(1)
        bsz_prev, asz_prev = bsz.shift(1), asz.shift(1)
        bid_ofi = np.where(
            bid > bid_prev,
            bsz,
            np.where(bid == bid_prev, bsz - bsz_prev, -bsz_prev),
        )
        ask_ofi = np.where(
            ask < ask_prev,
            asz,
            np.where(ask == ask_prev, asz - asz_prev, -asz_prev),
        )
        ofi = pd.Series(bid_ofi - ask_ofi, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        feat = pd.DataFrame(index=idx)
        feat["feat_mid"] = mid
        feat["feat_log_spread"] = np.log(spread.clip(lower=0.25))
        feat["feat_rel_spread"] = spread / mid
        feat["feat_imbalance"] = imb
        feat["feat_log_depth"] = np.log1p(depth)
        feat["feat_micro_dev"] = (micro - mid) / mid
        feat["feat_ofi_1s"] = ofi
        feat["feat_ret_30s"] = r.rolling(30, min_periods=30).sum()
        feat["feat_ret_300s"] = r.rolling(300, min_periods=300).sum()
        for w in (30, 60, 300, 900):
            feat[f"feat_log_rv_{w}s"] = np.log(np.sqrt(r2.rolling(w, min_periods=w).sum()) + 1e-8)
            feat[f"feat_log_vol_{w}s"] = np.log1p(vol.rolling(w, min_periods=w).sum())
            feat[f"feat_hl_{w}s"] = hl.rolling(w, min_periods=w).mean()
            feat[f"feat_imb_{w}s"] = imb.rolling(w, min_periods=w).mean()
            feat[f"feat_ofi_{w}s"] = ofi.rolling(w, min_periods=w).sum()
            feat[f"feat_abs_ofi_{w}s"] = ofi.abs().rolling(w, min_periods=w).sum()
        seconds_from_open = np.arange(len(idx))
        angle = 2 * np.pi * seconds_from_open / len(idx)
        feat["feat_tod_sin"] = np.sin(angle)
        feat["feat_tod_cos"] = np.cos(angle)
        feat["feat_tod_frac"] = seconds_from_open / len(idx)

        pos_r2 = (r.clip(lower=0.0)) ** 2
        neg_r2 = (r.clip(upper=0.0)) ** 2
        absr = r.abs()
        bp_pair = (absr * absr.shift(1)).fillna(0.0)
        first_future_pair = (absr.shift(-1) * absr).fillna(0.0)
        for horizon_min in (1, 5, 15, 30):
            sec = horizon_min * 60
            fut = _rolling_sum_future(r2, sec)
            feat[f"tgt_logrv_{horizon_min}m"] = np.log(np.sqrt(fut) + 1e-8)
            up = _rolling_sum_future(pos_r2, sec)
            down = _rolling_sum_future(neg_r2, sec)
            total = up + down
            feat[f"tgt_semiasym_{horizon_min}m"] = (down - up) / (total + EPS)
            bp = (np.pi / 2.0) * (
                _rolling_sum_future(bp_pair, sec) - first_future_pair
            ).clip(lower=0.0)
            feat[f"tgt_jump_{horizon_min}m"] = ((total - bp).clip(lower=0.0)) / (total + EPS)

        origin = feat.iloc[::stride].copy()
        origin["meta_day"] = pd.Timestamp(day.date())
        rows.append(origin)

    if not rows:
        raise SystemExit("no cached days found; run `pull` first")
    panel = pd.concat(rows).sort_index()
    tgt_cols = [c for c in panel.columns if c.startswith("tgt_")]
    feat_cols = [c for c in panel.columns if c.startswith("feat_")]
    panel = panel.dropna(subset=tgt_cols + feat_cols)
    out = ROOT / "artifacts" / "hf_lob_es_panel.parquet"
    out.parent.mkdir(exist_ok=True)
    panel.to_parquet(out)
    print(f"panel: {panel.shape} -> {out}", flush=True)
    return panel


def day_folds(panel: pd.DataFrame, n_folds: int = 4, purge_days: int = 1):
    days = np.array(sorted(pd.to_datetime(panel["meta_day"]).unique()))
    dpos = {d: i for i, d in enumerate(days)}
    fold_of_day = np.minimum(np.arange(len(days)) * n_folds // len(days), n_folds - 1)
    fold = panel["meta_day"].map(lambda d: fold_of_day[dpos[np.datetime64(pd.Timestamp(d))]]).values
    pos = panel["meta_day"].map(lambda d: dpos[np.datetime64(pd.Timestamp(d))]).values

    def train_mask(k: int):
        kpos = np.where(fold_of_day == k)[0]
        lo, hi = kpos.min(), kpos.max()
        return (fold != k) & ((pos < lo - purge_days) | (pos > hi + purge_days))

    return fold, train_mask


def _robust_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    zclip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-fold robust scaling: median/IQR, clip, nan -> 0."""
    med = np.nanmedian(X_train, axis=0)
    q25 = np.nanpercentile(X_train, 25, axis=0)
    q75 = np.nanpercentile(X_train, 75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(X_train, axis=0)
    scale = np.where(scale > 1e-12, scale, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)

    def tx(X):
        Z = (X - med) / scale
        Z = np.clip(Z, -zclip, zclip)
        return np.nan_to_num(Z, nan=0.0, posinf=zclip, neginf=-zclip)

    return tx(X_train), tx(X_test)


def _standard_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-fold mean/std scaling, no clipping: appropriate for smooth HAR state."""
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)

    def tx(X):
        return np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    return tx(X_train), tx(X_test)


def fit_oos(
    X: np.ndarray,
    y: np.ndarray,
    fold,
    train_mask,
    model_fn,
    n_folds: int = 4,
    zclip: float = 8.0,
    scaler: str = "standard",
):
    p = np.full_like(y, np.nan, dtype=float)
    for k in range(n_folds):
        tr, te = train_mask(k), fold == k
        if scaler == "robust":
            Xtr, Xte = _robust_transform(X[tr], X[te], zclip=zclip)
        elif scaler == "standard":
            Xtr, Xte = _standard_transform(X[tr], X[te])
        else:
            raise ValueError(f"unknown scaler {scaler!r}")
        m = model_fn().fit(Xtr, y[tr])
        p[te] = m.predict(Xte)
    return p


def fit_oos_har_l1(
    X_har: np.ndarray,
    X_l1: np.ndarray,
    y: np.ndarray,
    fold,
    train_mask,
    model_fn,
    n_folds: int = 4,
    zclip: float = 8.0,
):
    """OOS fit with standard-scaled HAR state and robust-clipped L1 state."""
    p = np.full_like(y, np.nan, dtype=float)
    for k in range(n_folds):
        tr, te = train_mask(k), fold == k
        Htr, Hte = _standard_transform(X_har[tr], X_har[te])
        Ltr, Lte = _robust_transform(X_l1[tr], X_l1[te], zclip=zclip)
        Xtr = np.column_stack([Htr, Ltr])
        Xte = np.column_stack([Hte, Lte])
        m = model_fn().fit(Xtr, y[tr])
        p[te] = m.predict(Xte)
    return p


def evaluate(panel: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    feat_cols = [c for c in panel.columns if c.startswith("feat_")]
    tod_cols = ["feat_tod_sin", "feat_tod_cos", "feat_tod_frac"]
    har_cols = tod_cols + [
        "feat_log_rv_30s",
        "feat_log_rv_60s",
        "feat_log_rv_300s",
        "feat_log_rv_900s",
        "feat_log_vol_60s",
        "feat_log_vol_300s",
    ]
    l1_cols = [
        "feat_log_spread",
        "feat_rel_spread",
        "feat_imbalance",
        "feat_log_depth",
        "feat_micro_dev",
        "feat_ofi_1s",
        "feat_ofi_30s",
        "feat_ofi_60s",
        "feat_ofi_300s",
        "feat_abs_ofi_30s",
        "feat_abs_ofi_60s",
        "feat_abs_ofi_300s",
        "feat_imb_30s",
        "feat_imb_60s",
        "feat_imb_300s",
        "feat_hl_60s",
        "feat_hl_300s",
        "feat_ret_30s",
        "feat_ret_300s",
    ]
    har_cols = [c for c in har_cols if c in panel.columns]
    l1_cols = [c for c in l1_cols if c in panel.columns]
    Xs = {
        "tod": panel[tod_cols].values.astype(float),
        "har_flow": panel[har_cols].values.astype(float),
        "gbt_har": panel[har_cols].values.astype(float),
    }
    X_har = panel[har_cols].values.astype(float)
    X_l1 = panel[l1_cols].values.astype(float)
    fold, tm = day_folds(panel)
    rows = []
    for tgt in [c for c in panel.columns if c.startswith("tgt_")]:
        y = panel[tgt].values.astype(float)
        unc = np.full(len(y), np.nan)
        for k in range(4):
            unc[fold == k] = y[tm(k)].mean()
        base_mse = float(np.nanmean((y - unc) ** 2))
        pred = {"unconditional": unc}
        mse = {"unconditional": base_mse}
        for name, X in Xs.items():
            if name.startswith("gbt"):
                fn = lambda: HistGradientBoostingRegressor(
                    max_iter=600,
                    learning_rate=0.03,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=0.0,
                    random_state=7,
                )
            else:
                fn = lambda: Ridge(alpha=1.0)
            pred[name] = fit_oos(X, y, fold, tm, fn)
            mse[name] = float(np.nanmean((y - pred[name]) ** 2))
        pred["har_l1_ridge"] = fit_oos_har_l1(
            X_har, X_l1, y, fold, tm, lambda: Ridge(alpha=1.0)
        )
        mse["har_l1_ridge"] = float(np.nanmean((y - pred["har_l1_ridge"]) ** 2))
        pred["gbt_har_l1"] = fit_oos_har_l1(
            X_har,
            X_l1,
            y,
            fold,
            tm,
            lambda: HistGradientBoostingRegressor(
                max_iter=600,
                learning_rate=0.03,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.0,
                random_state=7,
            ),
        )
        mse["gbt_har_l1"] = float(np.nanmean((y - pred["gbt_har_l1"]) ** 2))

        # Incremental L1 linear gate: can signed/top-of-book features explain the
        # robust HAR/flow residual using train-fold scaling only?
        e = y - pred["har_flow"]
        ep_l1 = fit_oos(
            X_l1,
            e,
            fold,
            tm,
            lambda: Ridge(alpha=1.0),
            scaler="robust",
        )
        eres_base = np.full(len(e), np.nan)
        for k in range(4):
            eres_base[fold == k] = e[tm(k)].mean()
        l1_resid_r2 = 1.0 - np.nanmean((e - ep_l1) ** 2) / np.nanmean((e - eres_base) ** 2)

        best = min(mse, key=mse.get)
        har_r2 = 1.0 - mse["har_flow"] / base_mse
        har_l1_r2 = 1.0 - mse["har_l1_ridge"] / base_mse
        gbt_har_r2 = 1.0 - mse["gbt_har"] / base_mse
        gbt_l1_r2 = 1.0 - mse["gbt_har_l1"] / base_mse
        for name, val in mse.items():
            rows.append(
                {
                    "target": tgt,
                    "model": name,
                    "mse": val,
                    "r2_vs_unc": 1.0 - val / base_mse,
                    "best": name == best,
                    "delta_r2_vs_har": (
                        har_l1_r2 - har_r2 if name == "har_l1_ridge"
                        else gbt_l1_r2 - gbt_har_r2 if name == "gbt_har_l1"
                        else np.nan
                    ),
                    "l1_resid_gate_r2": l1_resid_r2 if name == "har_l1_ridge" else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    path = ROOT / "artifacts" / "hf_lob_oracle.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"report -> {path}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cost", "pull", "build", "eval", "all"])
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    days = business_days(args.start, args.days)
    costs = None
    cost_path = ROOT / "artifacts" / f"hf_lob_costs_{args.start.replace('-', '')}_{args.days}.csv"
    if args.cmd in ("cost", "pull", "all"):
        if args.cmd in ("pull", "all") and cost_path.exists() and not args.refresh:
            print(f"[cost-cache] using {cost_path}", flush=True)
            costs = pd.read_csv(cost_path)
        else:
            costs = cost_days(days, verbose=True)
            cost_path.parent.mkdir(exist_ok=True)
            costs.to_csv(cost_path, index=False)
            print(f"[cost-cache] wrote {cost_path}", flush=True)
        live = costs[costs["total"].fillna(0) > 0].head(args.days)
        print(live.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nestimated nonzero {len(live)} sessions: ${live['total'].sum():.4f}")
        if args.cmd == "cost":
            return
    assert costs is not None or args.cmd not in ("pull", "all")
    live = costs[costs["total"].fillna(0) > 0].head(args.days) if costs is not None else None
    live_days = pd.to_datetime(live["day"]).tolist() if live is not None else []
    if args.cmd in ("pull", "all"):
        pull_days(live_days, max_cost=args.max_cost, refresh=args.refresh, costs=live)
    panel = None
    if args.cmd in ("build", "all"):
        panel = build_panel(args.start, args.days)
    if args.cmd in ("eval", "all"):
        if panel is None:
            panel_path = ROOT / "artifacts" / "hf_lob_es_panel.parquet"
            panel = pd.read_parquet(panel_path)
        evaluate(panel)


if __name__ == "__main__":
    main()
