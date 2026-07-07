"""Parallel historical pull: N concurrent day-workers, upfront cost estimate.

Vs the serial puller: no per-day cost quotes (an upfront ranged defs quote + one
sampled quote-window sets the estimate; the account's own spending limit is the hard
backstop — a 402 fails harmlessly and the run is resumable). Each worker uses its own
adapter/HTTP client; cache files are per-day so writes never collide.

Usage: python scripts/pull_parallel.py --config configs/nvda.yaml --workers 6
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.data.databento import DatabentoAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = get_config(args.config)
    adapter = DatabentoAdapter(cfg)
    meta = adapter._client().metadata
    s, e = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)

    # bars (cached after the earlier run) + upfront estimate
    bars = adapter.minute_bars(cfg.start, cfg.end)
    print(f"[bars] {len(bars)} RTH bars ready", flush=True)
    days = adapter.cal.trading_days(cfg.start, cfg.end)
    todo = [d for d in days if not (
        os.path.exists(os.path.join(adapter.cache_dir,
                                    f"{adapter.root_tag}_defs_{d.strftime('%Y%m%d')}.parquet"))
        and os.path.exists(os.path.join(adapter.cache_dir,
                                        f"{adapter.root_tag}_cbbo_{d.strftime('%Y%m%d')}.parquet"))
    )]
    try:
        defs_total = float(meta.get_cost(dataset="OPRA.PILLAR", schema="definition",
                                         symbols=adapter.option_roots, stype_in="parent",
                                         start=s.strftime("%Y-%m-%d"), end=e.strftime("%Y-%m-%d")))
        mid = todo[len(todo) // 2]
        close_utc = (adapter.cal.session_close(mid).tz_localize("America/New_York")
                     .tz_convert("UTC"))
        q1 = float(meta.get_cost(dataset="OPRA.PILLAR", schema="cbbo-1m",
                                 symbols=adapter.option_roots, stype_in="parent",
                                 start=(close_utc - pd.Timedelta(minutes=adapter.close_window_min)).isoformat(),
                                 end=(close_utc + pd.Timedelta(minutes=1)).isoformat()))
        print(f"[estimate] defs whole-range ${defs_total:.2f} + quotes ~${q1:.4f} x "
              f"{len(todo)} days ~= ${defs_total + q1 * len(todo):.2f} "
              f"(upper bound; account limit is the backstop)", flush=True)
    except Exception as ex:
        print(f"[estimate] unavailable ({ex}) — proceeding, account limit protects", flush=True)

    print(f"[chain] {len(todo)} sessions to pull with {args.workers} workers", flush=True)

    def do_day(day):
        a = DatabentoAdapter(cfg)          # own client per task
        try:
            chain = a._session_chain(day)  # caches defs + quotes
            return "ok" if chain is not None and not chain.empty else "empty"
        except Exception as ex:
            return f"err:{type(ex).__name__}"

    counts = {"ok": 0, "empty": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(do_day, d): d for d in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            counts["ok" if r == "ok" else ("empty" if r == "empty" else "err")] += 1
            if r.startswith("err"):
                print(f"[{futures[fut].date()}] {r}", flush=True)
            if i % 100 == 0 or i == len(todo):
                print(f"[chain] {i}/{len(todo)} done | ok {counts['ok']} "
                      f"empty {counts['empty']} err {counts['err']}", flush=True)

    print(f"\nDONE: {counts}", flush=True)


if __name__ == "__main__":
    main()
