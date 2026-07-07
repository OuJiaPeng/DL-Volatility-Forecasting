"""Data-quality audit: cache coverage, quote sanity, and IV-solve cross-validation.

Read-only over data_cache/ (safe to run alongside an active pull). Solves a SAMPLE of
sessions through volsolve for the not-yet-panelized range; after the panel rebuild the
surface cache makes the full audit cheap. External cross-checks:
  * parity forward vs same-day ES close          (should track within ~2%)
  * implied discount rate vs the rate regime      (2021 ~0%, 2023-24 ~4-5%)
  * solved ATM(21d) vs Cboe VIX                   (VIX above ATM by the skew premium)

Usage:
    python scripts/audit_data.py --config configs/spx.yaml [--sample 10] [--out artifacts/audit_report.md]
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.calendar import Calendar
from volforecast.data import volsolve
from volforecast.data.databento import DatabentoAdapter, ANN

REPORT = []


def log(line=""):
    print(line, flush=True)
    REPORT.append(line)


def verdict(ok: bool, warn: bool, label: str, detail: str):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    log(f"[{tag}] {label}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    ap.add_argument("--sample", type=int, default=10, help="solve every Nth session for IV checks")
    ap.add_argument("--out", default="artifacts/audit_report.md")
    args = ap.parse_args()

    cfg = get_config(args.config)
    cal = Calendar()
    adapter = DatabentoAdapter(cfg)
    cache = adapter.cache_dir
    files = set(os.listdir(cache))

    days = cal.trading_days(cfg.start, cfg.end)
    log(f"# Data audit — {cfg.start} .. {cfg.end}  ({len(days)} candidate bdays)\n")

    # --- 1. session coverage -----------------------------------------------------------
    have_defs, have_cbbo, complete, missing = [], [], [], []
    for d in days:
        tag = d.strftime("%Y%m%d")
        fd, fq = f"spx_defs_{tag}.parquet" in files, f"spx_cbbo_{tag}.parquet" in files
        if fd:
            have_defs.append(d)
        if fq:
            have_cbbo.append(d)
        (complete if (fd and fq) else missing).append(d)
    log(f"## 1. Coverage: {len(complete)} complete sessions, {len(missing)} missing/incomplete")
    miss_by_year = pd.Series([m.year for m in missing]).value_counts().sort_index()
    log(f"missing by year: {miss_by_year.to_dict()}")
    # ~10 holidays + ~6 half-days per year are expected misses
    per_year_expected = 18
    bad_years = {y: c for y, c in miss_by_year.items()
                 if c > per_year_expected and y < pd.Timestamp(cfg.end).year}
    verdict(not bad_years, len(bad_years) <= 1, "coverage",
            f"unexplained-gap years: {bad_years or 'none'} (>{per_year_expected}/yr threshold; "
            f"holidays+half-days expected)")
    orphans = len(have_defs) + len(have_cbbo) - 2 * sum(
        1 for d in days if f"spx_defs_{d.strftime('%Y%m%d')}.parquet" in files
        and f"spx_cbbo_{d.strftime('%Y%m%d')}.parquet" in files)
    verdict(orphans < 20, orphans < 50, "file pairing", f"{orphans} orphan def/quote files")

    # --- 2. quotes sanity (sampled) ----------------------------------------------------
    sampled = complete[:: args.sample]
    rows, bid_pos, crossed, expiries = [], [], [], []
    fwd_dev, imp_rate, atm21 = [], [], []
    es = adapter.minute_bars(cfg.start, cfg.end)
    es_close = es["close"].groupby(es.index.normalize()).last()
    for d in sampled:
        chain = adapter._session_chain(d)
        if chain is None or chain.empty:
            continue
        rows.append(len(chain))
        bid_pos.append(float((chain["bid"] > 0).mean()))
        crossed.append(float((chain["ask"] < chain["bid"]).mean()))
        expiries.append(chain["expiry"].nunique())
        # IV solve cross-checks on the ~monthly expiry
        session = pd.Timestamp(d).normalize()
        best = None
        for expiry, grp in chain.groupby("expiry"):
            cd = (pd.Timestamp(expiry) - session).days
            if 20 <= cd <= 45:
                best = (cd, grp)
                break
        if best is None:
            continue
        cd, grp = best
        solved = volsolve.solve_expiry(grp, cd / 365.0)
        if solved is None:
            continue
        es_c = es_close.get(session, np.nan)
        if np.isfinite(es_c):
            fwd_dev.append(abs(solved["forward"] / es_c - 1.0))
        imp_rate.append((-np.log(solved["discount"]) / (cd / 365.0), session.year))
        summary = volsolve.surface_summary(chain, session, [21])
        if summary:
            atm21.append((session, summary["atm_iv_21"]))

    if rows:
        log(f"\n## 2. Chain sanity ({len(rows)} sampled sessions)")
        log(f"quote rows/session: median {int(np.median(rows))}, "
            f"min {min(rows)}, max {max(rows)}")
        verdict(min(rows) > 500, min(rows) > 200, "chain depth",
                f"thinnest sampled session has {min(rows)} quotes")
        verdict(np.mean(crossed) < 0.01, np.mean(crossed) < 0.05, "crossed quotes",
                f"{np.mean(crossed):.2%} mean crossed rate")
        log(f"expiries/session: median {int(np.median(expiries))} "
            f"(0DTE era should trend higher post-2022)")

    # --- 3. parity forward vs ES ------------------------------------------------------
    log("\n## 3. IV-solve cross-validation")
    if fwd_dev:
        med = float(np.median(fwd_dev))
        verdict(med < 0.02, med < 0.05, "parity forward vs ES close",
                f"median |F/ES-1| = {med:.2%} over {len(fwd_dev)} sessions "
                "(SPX-vs-ES basis + close-time mismatch expected, ~<2%)")
    if imp_rate:
        df_r = pd.DataFrame(imp_rate, columns=["r", "year"])
        by_year = df_r.groupby("year")["r"].median()
        log(f"implied discount rate by year (monthly expiry): "
            f"{ {y: f'{v:.1%}' for y, v in by_year.items()} }")
        ok = all((v < 0.02) if y <= 2021 else (0.0 < v < 0.08)
                 for y, v in by_year.items())
        verdict(ok, True, "rate regime",
                "2021 should be ~0-1%, 2023+ ~4-6% (parity regression sanity)")

    # --- 4. ATM vs VIX -----------------------------------------------------------------
    if atm21:
        # surface_summary called directly returns ANNUALIZED vol (the adapter converts
        # to daily only for the panel) — so scale by 100 only, NOT by sqrt(252)
        s_atm = pd.Series({d: v for d, v in atm21}) * 100
        vix = adapter._vix_series().reindex(s_atm.index)
        both = pd.concat([s_atm.rename("atm"), vix.rename("vix")], axis=1).dropna()
        if len(both) > 10:
            corr = both["atm"].corr(both["vix"])
            spread = (both["vix"] - both["atm"]).median()
            above = float((both["vix"] > both["atm"]).mean())
            verdict(corr > 0.9, corr > 0.75, "ATM vs VIX correlation",
                    f"corr={corr:.3f} over {len(both)} sessions")
            verdict(0.5 < spread < 6.0 and above > 0.8, above > 0.6, "skew premium",
                    f"median VIX-ATM = {spread:.2f} vol pts, VIX above ATM {above:.0%} of days")

    # --- 5. ES bars --------------------------------------------------------------------
    bars_per_day = es["close"].groupby(es.index.normalize()).count()
    full_days = float((bars_per_day >= 300).mean())
    log("\n## 4. ES minute bars")
    verdict(full_days > 0.95, full_days > 0.9, "ES session completeness",
            f"{full_days:.1%} of sessions have >=300 RTH bars "
            f"({len(bars_per_day)} sessions total)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
