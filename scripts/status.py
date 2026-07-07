"""One-command project status: data cache, artifacts, ledger. Run via `make status`."""
import os
import time

import pandas as pd

CACHE, ART = "data_cache", "artifacts"

print("=== data_cache ===")
files = os.listdir(CACHE) if os.path.isdir(CACHE) else []
for root in ("spx", "nvda"):
    defs = sum(f.startswith(f"{root}_defs") for f in files)
    cbbo = sum(f.startswith(f"{root}_cbbo") for f in files)
    dte = sum(f.startswith(f"{root}0dte") for f in files)
    print(f"  {root.upper():5s} defs {defs:5d} | close-window quotes {cbbo:5d} | 0dte strips {dte:5d}")
bars = [f for f in files if f.startswith("es_ohlcv")]
print(f"  underlying bar files: {len(bars)}")
if files:
    newest = max((os.path.join(CACHE, f) for f in files), key=os.path.getmtime)
    age = (time.time() - os.path.getmtime(newest)) / 60
    print(f"  newest file: {os.path.basename(newest)} ({age:.0f} min ago"
          f"{' — a pull is likely ACTIVE' if age < 5 else ''})")

print("\n=== artifacts ===")
if os.path.isdir(ART):
    for f in sorted(os.listdir(ART)):
        p = os.path.join(ART, f)
        mb = os.path.getsize(p) / 1e6
        print(f"  {f:35s} {mb:8.2f} MB  {time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}")

print("\n=== experiments ledger ===")
if os.path.exists("experiments/ledger.csv"):
    df = pd.read_csv("experiments/ledger.csv")
    print(f"  {len(df)} runs | experiments: {sorted(df.exp.unique().tolist())}")
    print(f"  latest: {df.iloc[-1].ts}  {df.iloc[-1].exp}/{df.iloc[-1].arm}  val_qlike={df.iloc[-1].val_qlike}")
