"""Budget-metered historical pull: ES minute bars + per-day SPX chain (defs + close quotes).

Every day's cost is quoted with the FREE metadata.get_cost call before pulling, and the
run hard-stops at --budget. Already-cached days are skipped, so the script is resumable
and re-runs are free. Uses the adapter's own fetch/cache paths, so the panel build finds
everything afterward.

Usage:
    python scripts/pull_spx_history.py --config configs/spx.yaml --budget 80
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.data.databento import DatabentoAdapter

_SENTINEL = object()
_pool = ThreadPoolExecutor(max_workers=1)


def with_timeout(fn, *args, timeout: float = 180.0, retries: int = 1):
    """Run fn with a hard timeout; abandon hung workers (Databento streams can stall
    indefinitely with no client-side timeout — observed live, 2021-08-03). Returns
    _SENTINEL if every attempt timed out."""
    global _pool
    for _ in range(retries + 1):
        fut = _pool.submit(fn, *args)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            _pool = ThreadPoolExecutor(max_workers=1)  # leave the hung thread behind
        except Exception:
            raise
    return _SENTINEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    ap.add_argument("--budget", type=float, default=80.0, help="hard spend cap, USD")
    args = ap.parse_args()

    cfg = get_config(args.config)
    adapter = DatabentoAdapter(cfg)
    client = adapter._client()
    meta = client.metadata
    total = 0.0

    def quote_cost(**kw) -> float:
        try:
            return float(meta.get_cost(**kw))
        except Exception:
            return 0.0

    # --- ES minute bars (single ranged pull; free if already cached) -----------------
    s, e = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    es_cache = os.path.join(
        adapter.cache_dir,
        f"es_ohlcv1m_{adapter.es_symbol.replace('.', '-')}_{s.date()}_{e.date()}.parquet",
    )
    es_cached = os.path.exists(es_cache)
    es_cost = 0.0 if es_cached else quote_cost(
        dataset=adapter.u_dataset, schema="ohlcv-1m", symbols=[adapter.es_symbol],
        stype_in=adapter.u_stype, start=s.strftime("%Y-%m-%d"),
        end=(e + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    print(f"[es] {'cache hit, $0.00' if es_cached else f'quoted ${es_cost:.2f}'} "
          f"for {s.date()} -> {e.date()}", flush=True)
    bars = adapter.minute_bars(cfg.start, cfg.end)
    total += es_cost
    print(f"[es] {len(bars)} RTH bars | total ${total:.2f}", flush=True)

    # --- SPX chain, day by day -------------------------------------------------------
    days = adapter.cal.trading_days(cfg.start, cfg.end)
    print(f"[chain] {len(days)} candidate sessions", flush=True)
    pulled = skipped = empty = 0
    for i, day in enumerate(days):
        tag = day.strftime("%Y%m%d")
        defs_p = os.path.join(adapter.cache_dir, f"{adapter.root_tag}_defs_{tag}.parquet")
        cbbo_p = os.path.join(adapter.cache_dir, f"{adapter.root_tag}_cbbo_{tag}.parquet")
        if os.path.exists(defs_p) and os.path.exists(cbbo_p):
            skipped += 1
            continue

        close_utc = (adapter.cal.session_close(day)
                     .tz_localize("America/New_York").tz_convert("UTC"))

        def _day_cost():
            return quote_cost(
                dataset="OPRA.PILLAR", schema="definition", symbols=adapter.option_roots,
                stype_in="parent", start=day.strftime("%Y-%m-%d"),
                end=(day + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            ) + quote_cost(
                dataset="OPRA.PILLAR", schema="cbbo-1m", symbols=adapter.option_roots,
                stype_in="parent",
                start=(close_utc - pd.Timedelta(minutes=adapter.close_window_min)).isoformat(),
                end=(close_utc + pd.Timedelta(minutes=1)).isoformat(),
            )

        # get_cost hangs on poisoned days just like the data call — watchdog EVERYTHING
        day_cost = with_timeout(_day_cost, timeout=60.0)
        if day_cost is _SENTINEL:
            print(f"[TIMEOUT] {day.date()} cost-quote hung twice; skipping day", flush=True)
            empty += 1
            continue
        if total + day_cost > args.budget:
            print(f"[STOP] budget cap ${args.budget:.2f} would be exceeded at {day.date()} "
                  f"(spent ${total:.2f}, day ${day_cost:.2f})", flush=True)
            break

        chain = with_timeout(adapter._session_chain, day)  # pulls + caches defs and quotes
        if chain is _SENTINEL:
            print(f"[TIMEOUT] {day.date()} hung twice; skipping (re-run to retry)", flush=True)
            empty += 1
            continue
        total += day_cost
        if chain is None or chain.empty:
            empty += 1
        else:
            pulled += 1
        if (pulled + empty) % 25 == 0 or i == len(days) - 1:
            print(f"[chain] {day.date()} | pulled {pulled} empty {empty} skipped {skipped} "
                  f"| spent ${total:.2f}", flush=True)

    print(f"\nDONE: pulled {pulled}, empty/holiday {empty}, already-cached {skipped}, "
          f"total spent ~${total:.2f}", flush=True)


if __name__ == "__main__":
    main()
