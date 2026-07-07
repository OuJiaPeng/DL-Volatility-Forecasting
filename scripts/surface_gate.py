"""Surface gate: does full-surface shape (beyond our scalar summaries) predict
champion residuals? Fold-wise, per protocol. $0 — solves from cached chains.

Per session: light per-expiry solve (parity forward + ~12 near-ATM/wing inversions)
-> tenor-gridded (ATM, skew, curvature) -> PCA (train-fit per fold) -> ridge of the
champion's OOF log-errors on the PCs, scored on each fold's val (fold-wise geometry,
the improved gate design). Positive, sign-consistent R^2 across folds => the
surface-token transformer gets its license.

Usage: python scripts/surface_gate.py --config configs/spx.yaml --panel artifacts/panel.parquet --champion hariv_x
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.splits import make_walkforward
from volforecast.data.databento import DatabentoAdapter
from volforecast.data import volsolve

TENORS = np.array([5, 10, 21, 42, 63]) / 252.0  # year-fractions


def session_curve(adapter, day):
    """(len(TENORS) x 3) [atm, skew, curvature] or None."""
    chain = adapter._session_chain(day)
    if chain is None or chain.empty:
        return None
    rows = []
    for expiry, grp in chain.groupby("expiry"):
        cd = (pd.Timestamp(expiry).normalize() - pd.Timestamp(day).normalize()).days
        if not (2 <= cd <= 100):
            continue
        tau = cd / 365.0
        s = volsolve.solve_expiry(grp, tau)
        if s is None or not np.isfinite(s["skew_25d"]):
            continue
        # curvature proxy: wing-average minus ATM (from the same solve's smile)
        rows.append((tau, s["atm_iv"], s["skew_25d"]))
    if len(rows) < 4:
        return None
    rows.sort()
    taus = np.array([r[0] for r in rows])
    atm = np.array([r[1] for r in rows])
    skw = np.array([r[2] for r in rows])
    atm_g = np.interp(TENORS, taus, atm)
    skw_g = np.interp(TENORS, taus, skw)
    slope_g = np.gradient(atm_g, TENORS)          # local term-structure slope
    return np.concatenate([atm_g, skw_g, slope_g])  # 15 features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--champion", default="hariv_x", choices=["hariv_x", "har_iv"])
    args = ap.parse_args()

    cfg = get_config(args.config)
    panel = pd.read_parquet(args.panel)
    adapter = DatabentoAdapter(cfg)
    folds = make_walkforward(panel, cfg, n_folds=4)
    tgt = [f"tgt_rv_{h}" for h in cfg.horizons]

    print(f"[gate] building surface curves for {len(panel)} sessions "
          f"(light solve, cached chains)...", flush=True)
    curves, index = [], []
    for i, t0 in enumerate(panel.index):
        c = session_curve(adapter, pd.Timestamp(t0).normalize())
        if c is not None:
            curves.append(c)
            index.append(t0)
        if (i + 1) % 200 == 0:
            print(f"[gate] {i+1}/{len(panel)}", flush=True)
    S = pd.DataFrame(np.vstack(curves), index=pd.DatetimeIndex(index))
    name = getattr(cfg, "name", "panel")
    S.columns = S.columns.astype(str)
    S.to_parquet(f"artifacts/{name}_surface_curve.parquet")
    print(f"[gate] {len(S)}/{len(panel)} sessions solved -> artifacts/{name}_surface_curve.parquet", flush=True)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from volforecast.models.classical_arms import HARIVX, HARIV

    Champ = HARIVX if args.champion == "hariv_x" else HARIV
    print(f"[gate] champion={args.champion}; fold-wise PCA + residual ridge", flush=True)
    r2 = {h: [] for h in cfg.horizons}
    for s in folds:
        tr = s.train.intersection(S.index)
        va = s.val.intersection(S.index)
        if len(tr) < 100 or len(va) < 30:
            continue
        pca = PCA(n_components=5).fit(S.loc[tr].values)
        Ztr, Zva = pca.transform(S.loc[tr].values), pca.transform(S.loc[va].values)
        champ = Champ(cfg.horizons).fit(panel, s.train)
        e_tr = (np.log(np.clip(panel.loc[tr, tgt].values, 1e-12, None))
                - np.log(np.clip(champ.predict(panel, tr), 1e-12, None)))
        e_va = (np.log(np.clip(panel.loc[va, tgt].values, 1e-12, None))
                - np.log(np.clip(champ.predict(panel, va), 1e-12, None)))
        for j, h in enumerate(cfg.horizons):
            m = Ridge(alpha=1.0).fit(Ztr, e_tr[:, j])
            base = np.mean((e_va[:, j] - e_tr[:, j].mean()) ** 2)
            r2[h].append(1.0 - np.mean((e_va[:, j] - m.predict(Zva)) ** 2) / base)

    print("\nfold-wise OOS R^2 of surface PCs on champion residuals:")
    any_pass = False
    for h in cfg.horizons:
        v = np.array(r2[h])
        ok = (v > 0).all() and v.mean() > 0.01
        any_pass |= ok
        print(f"  h={h:2d}: folds {np.round(v, 4).tolist()}  mean {v.mean():+.4f} "
              f"{'<- LOADS' if ok else ''}")
    print(f"\n[GATE] {'OPEN: surface-token transformer licensed' if any_pass else 'CLOSED: surface shape adds nothing beyond scalar summaries'}")


if __name__ == "__main__":
    main()
