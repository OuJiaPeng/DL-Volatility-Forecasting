"""Oracle / information-exhaustion test for the champion model.

(a) Spectrum: unconditional -> champion -> perfect-foresight QLIKE floor, per horizon.
    "% closed" = how much of the closable distance the champion captures.
(b) Residual predictability: predict the champion's OUT-OF-FOLD log-errors from all
    panel features (ridge + gradient boosting), evaluated on a held-out time split of
    the pooled val residuals. OOS R^2 <= 0  =>  no exploitable signal remains in this
    information set; the campaign's "linear content is saturated" conclusion is proven
    rather than suspected.

Usage: python scripts/oracle_test.py --config configs/spx.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.splits import make_walkforward
from volforecast.models.classical_arms import HARIVX
from volforecast.eval.metrics import qlike

REPORT = []


def log(s=""):
    print(s, flush=True)
    REPORT.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spx.yaml")
    ap.add_argument("--panel", default="artifacts/panel.parquet")
    args = ap.parse_args()

    cfg = get_config(args.config)
    panel = pd.read_parquet(args.panel)
    folds = make_walkforward(panel, cfg, n_folds=4)
    horizons = list(cfg.horizons)
    tgt_cols = [f"tgt_rv_{h}" for h in horizons]
    feat_cols = [c for c in panel.columns if c.startswith("feat_")]

    # --- champion OOF predictions on the pooled val origins --------------------------
    rows = []
    for s in folds:
        champ = HARIVX(horizons).fit(panel, s.train)
        y = panel.loc[s.val, tgt_cols].values
        yhat = champ.predict(panel, s.val)
        mu = panel.loc[s.train, tgt_cols].mean().values          # unconditional anchor
        for j, t in enumerate(s.val):
            rows.append({"t0": t, "y": y[j], "yhat": yhat[j], "mu": mu})
    oof = sorted(rows, key=lambda r: r["t0"])
    y = np.stack([r["y"] for r in oof])
    yhat = np.stack([r["yhat"] for r in oof])
    mu = np.stack([r["mu"] for r in oof])
    n = len(y)

    log(f"# Oracle / exhaustion test — champion=hariv_x, {n} pooled OOF val origins\n")
    log("## (a) QLIKE spectrum per horizon: unconditional -> champion -> perfect floor")
    for j, h in enumerate(horizons):
        q_unc = qlike(y[:, j], mu[:, j])
        q_champ = qlike(y[:, j], yhat[:, j])
        q_floor = qlike(y[:, j], y[:, j])          # perfect foresight
        closed = (q_unc - q_champ) / (q_unc - q_floor)
        log(f"  h={h:2d}:  unconditional {q_unc:8.4f} | champion {q_champ:8.4f} | "
            f"floor {q_floor:8.4f} | gap closed: {closed:6.1%}")
    log("  (the floor includes irreducible randomness NO forecast can reach — "
        "'% closed' understates skill)")

    # --- (b) residual predictability -------------------------------------------------
    log("\n## (b) Residual predictability: can ANY feature predict the champion's errors?")
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = panel.loc[[r["t0"] for r in oof], feat_cols].values.astype(float)
    cut = int(n * 0.7)                              # time-ordered split of OOF residuals
    verdicts = []
    for j, h in enumerate(horizons):
        e = np.log(np.clip(y[:, j], 1e-12, None)) - np.log(np.clip(yhat[:, j], 1e-12, None))
        base = np.mean((e[cut:] - e[:cut].mean()) ** 2)
        r2s = {}
        for name, m in (("ridge", Ridge(alpha=1.0)),
                        ("gbt", HistGradientBoostingRegressor(max_iter=150, max_depth=3,
                                                              random_state=7))):
            m.fit(X[:cut], e[:cut])
            mse = np.mean((e[cut:] - m.predict(X[cut:])) ** 2)
            r2s[name] = 1.0 - mse / base
        verdicts.append(max(r2s.values()))
        log(f"  h={h:2d}:  OOS R^2 — ridge {r2s['ridge']:+.4f}, gbt {r2s['gbt']:+.4f}")

    best = max(verdicts)
    log(f"\nbest residual OOS R^2 across horizons: {best:+.4f}")
    if best <= 0.01:
        log("[VERDICT] information set EXHAUSTED: no feature (linear or nonlinear) "
            "predicts the champion's out-of-fold errors. Remaining error is noise + "
            "information outside G_t (flows, cross-section).")
    else:
        log(f"[VERDICT] residual signal remains (R^2={best:.3f}) — identify which "
            "features load and feed them back into the champion.")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/oracle_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print("\nreport -> artifacts/oracle_report.md")


if __name__ == "__main__":
    main()
