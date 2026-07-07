"""Ladder experiment runner: walk-forward arms -> experiments/ledger.csv.

See experiments/README.md for the protocol. Test columns stay NaN unless --milestone.

Usage:
    python scripts/run_experiment.py --config configs/spx.yaml --exp E0 \
        --arms har_rv,hybrid,persistence,garch,naive_iv --note "re-baseline"
    python scripts/run_experiment.py --config configs/spx.yaml --exp E2 \
        --arms hybrid --trunk lstm --note "E2 lstm arm"
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.splits import make_walkforward
from volforecast.baselines import DEFAULT_BASELINES
from volforecast.eval.walkforward import run_walkforward

LEDGER = "experiments/ledger.csv"
COLS = ["ts", "exp", "arm", "trunk", "overrides", "folds", "val_qlike", "val_cov80",
        "val_dm_t", "val_dm_p", "test_qlike", "test_dm_t", "test_dm_p", "seeds", "git",
        "note", "decision"]


def migrate_ledger() -> None:
    """Insert newly-added columns into an existing ledger (pads old rows)."""
    if not os.path.exists(LEDGER):
        return
    df = pd.read_csv(LEDGER, dtype=str)
    if list(df.columns) == COLS:
        return
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df[COLS].to_csv(LEDGER, index=False)


def git_stamp() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arms", required=True, help="comma list: har_rv,hybrid,...")
    ap.add_argument("--trunk", default=None, help="hybrid trunk override (E2 axis)")
    ap.add_argument("--set", dest="overrides", default="", help="model overrides k=v,k=v")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--milestone", action="store_true", help="reveal TEST columns (budgeted)")
    ap.add_argument("--note", default="")
    ap.add_argument("--panel", default="artifacts/panel.parquet")
    args = ap.parse_args()

    cfg = get_config(args.config)
    panel = pd.read_parquet(args.panel)
    folds = make_walkforward(panel, cfg, n_folds=args.folds)

    # apply model-config overrides (the single experiment axis)
    mcfg = SimpleNamespace(**vars(cfg.model))
    if args.trunk:
        mcfg.trunk = args.trunk
    for kv in filter(None, args.overrides.split(",")):
        k, v = kv.split("=", 1)
        cur = getattr(mcfg, k, None)
        setattr(mcfg, k, type(cur)(v) if cur is not None and not isinstance(cur, list) else v)

    factories = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        if arm == "hybrid":
            cube = None
            trunk_name = args.trunk or getattr(mcfg, "trunk", "itransformer")
            if "intra" in trunk_name:
                import numpy as np
                cube_path = "artifacts/intraday_cube.npy"
                if not os.path.exists(cube_path):
                    raise SystemExit(f"trunk {trunk_name!r} needs {cube_path}; rebuild the panel")
                cube = np.load(cube_path)
                if len(cube) != len(panel):
                    raise SystemExit("intraday cube is stale (row count != panel); rebuild the panel")

            def make_hybrid(split, _m=mcfg, _cube=cube):
                from volforecast.models.hybrid import HybridResidualForecaster
                f = HybridResidualForecaster(cfg.horizons, _m).bind_split(split)
                return f.bind_intraday(_cube) if _cube is not None else f
            factories[arm] = make_hybrid
        elif arm in ("statehar", "gbt", "har_iv", "hariv_x", "har_iv_m", "wedge_gbt", "ewa", "tabpfn"):
            from volforecast.models import classical_arms as ca
            _cls = {"statehar": ca.StateDependentHAR, "gbt": ca.GradientBoostedTrees,
                    "har_iv": ca.HARIV, "hariv_x": ca.HARIVX, "har_iv_m": ca.HARIVM, "wedge_gbt": ca.WedgeGBT,
                    "ewa": ca.EWAAggregator, "tabpfn": ca.TabPFNArm}[arm]
            factories[arm] = lambda split, _c=_cls: _c(cfg.horizons)
        elif arm == "gate":
            from volforecast.models.gate import NeuralGate
            factories[arm] = lambda split: NeuralGate(cfg.horizons)
        elif arm in DEFAULT_BASELINES:
            factories[arm] = lambda split, _c=DEFAULT_BASELINES[arm]: _c(cfg.horizons)
        else:
            raise SystemExit(f"unknown arm {arm!r}")

    table = run_walkforward(panel, folds, factories, cfg.horizons, milestone=args.milestone)
    print(f"\nexp={args.exp} trunk={args.trunk or getattr(mcfg, 'trunk', 'itransformer')} "
          f"folds={args.folds} milestone={args.milestone}")
    print(table.to_string(float_format=lambda x: f"{x:.6f}"))

    os.makedirs("experiments", exist_ok=True)
    migrate_ledger()
    new_file = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(COLS)
        for arm, r in table.iterrows():
            w.writerow([
                dt.datetime.now().isoformat(timespec="seconds"), args.exp, arm,
                args.trunk or getattr(mcfg, "trunk", "itransformer") if arm == "hybrid" else "",
                args.overrides, args.folds,
                f"{r['VAL_QLIKE']:.6f}", f"{r['VAL_COV80']:.4f}",
                f"{r['VAL_DM_t']:.4f}", f"{r['VAL_DM_p']:.4f}",
                f"{r['TEST_QLIKE']:.6f}", f"{r['TEST_DM_t']:.4f}", f"{r['TEST_DM_p']:.4f}",
                getattr(mcfg, "n_runs", ""), git_stamp(), args.note, "",
            ])
    print(f"\nledger: appended {len(table)} row(s) -> {LEDGER}")


if __name__ == "__main__":
    main()
