"""Targeted 0DTE pull: same-day-expiry ATM strip, final ~20 min, minute NBBO.

Uses cached definitions to find each session's 0DTE contracts near ATM (strikes within
a band of the ES level), then pulls cbbo-1m for JUST those raw symbols, 15:40-16:01 ET.
~30-60 contracts x ~20 rows/day => a few KB/session, a few dollars total.

Usage: python scripts/pull_0dte.py --config configs/spx.yaml --workers 6 [--band 0.015]
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.data.databento import DatabentoAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--band", type=float, default=0.015)
    args = ap.parse_args()

    cfg = get_config(args.config)
    adapter = DatabentoAdapter(cfg)
    bars = adapter.minute_bars(cfg.start, cfg.end)          # cache hit
    ref_px = bars["close"].groupby(bars.index.normalize()).last()

    days = [d for d in adapter.cal.trading_days(cfg.start, cfg.end)
            if os.path.exists(os.path.join(adapter.cache_dir,
                f"{adapter.root_tag}_defs_{d.strftime('%Y%m%d')}.parquet"))
            and not os.path.exists(os.path.join(adapter.cache_dir,
                f"{adapter.root_tag}0dte_{d.strftime('%Y%m%d')}.parquet"))]
    print(f"[0dte] {len(days)} sessions to pull", flush=True)

    def do_day(day):
        a = DatabentoAdapter(cfg)
        tag = day.strftime("%Y%m%d")
        out = os.path.join(a.cache_dir, f"{a.root_tag}0dte_{tag}.parquet")
        try:
            defs = pd.read_parquet(os.path.join(a.cache_dir, f"{a.root_tag}_defs_{tag}.parquet"))
            defs = defs.drop_duplicates(subset=["instrument_id"], keep="last")
            exp = pd.to_datetime(defs["expiration"]).dt.tz_localize(None).dt.normalize()
            px = ref_px.get(day.normalize(), np.nan)
            if not np.isfinite(px):
                return "no-ref"
            m = defs[(exp == day.normalize())
                     & (defs["strike_price"].astype(float) > px * (1 - args.band))
                     & (defs["strike_price"].astype(float) < px * (1 + args.band))]
            if m.empty:
                pd.DataFrame().to_parquet(out)   # cache the negative (pre-0DTE era days)
                return "no-0dte"
            close_utc = (a.cal.session_close(day).tz_localize("America/New_York")
                         .tz_convert("UTC"))
            data = a._client().timeseries.get_range(
                dataset="OPRA.PILLAR", schema="cbbo-1m",
                symbols=m["raw_symbol"].tolist(), stype_in="raw_symbol",
                start=(close_utc - pd.Timedelta(minutes=21)).isoformat(),
                end=(close_utc + pd.Timedelta(minutes=1)).isoformat(),
            )
            df = data.to_df().reset_index()
            keep = df[["ts_recv", "instrument_id", "bid_px_00", "ask_px_00"]].merge(
                m[["instrument_id", "strike_price", "instrument_class"]],
                on="instrument_id", how="left")
            keep.to_parquet(out)
            return "ok"
        except Exception as e:
            return f"err:{type(e).__name__}"

    counts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_day, d): d for d in days}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            counts[r] = counts.get(r, 0) + 1
            if r.startswith("err"):
                print(f"[{futs[f].date()}] {r}", flush=True)
            if i % 100 == 0 or i == len(days):
                print(f"[0dte] {i}/{len(days)} | {counts}", flush=True)
    print(f"DONE: {counts}", flush=True)


if __name__ == "__main__":
    main()
