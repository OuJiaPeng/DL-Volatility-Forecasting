"""IV-filtration-only protocol: oracle ceiling -> non-strawman ladder -> exhaustion.

Restrict to G^IV (option-derived features + VIX; NO realized-vol history) and forecast
forward RV. The shortfall from IV is the variance risk premium, so every rung doubles
as a premium model. Sequence per the playbook:

  1. Spectrum: unconditional -> each rung -> perfect-foresight floor ("gap closed").
  2. Ladder (log-space, per-horizon, per-fold):
       naive_iv    IV as-is                       (the strawman, shown for scale)
       mz_iv       log y = a + b log IV_h         (Mincer-Zarnowitz bias fix)
       iv_scalars  ridge on tenor IVs+skew+slope+VIX
       iv_curve    ridge on surface-curve PCs (+scalars)
       gbt_iv      trees on all IV features       (nonlinearity probe)
  3. Fold-wise residual exhaustion of the best rung within G^IV.
  4. Reference row: the full-filtration champion (hariv_x) — what RV history adds.

Usage: python scripts/iv_only.py --config configs/spx.yaml --panel artifacts/panel.parquet
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.config import get_config
from volforecast.splits import make_walkforward
from volforecast.eval.metrics import qlike, qlike_per_origin, dm_test
from volforecast.models.classical_arms import HARIVX

IV_SCALARS = ["feat_atm_iv_1", "feat_atm_iv_5", "feat_atm_iv_10", "feat_atm_iv_21",
              "feat_skew", "feat_term_slope", "feat_vix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--panel", required=True)
    args = ap.parse_args()

    cfg = get_config(args.config)
    panel = pd.read_parquet(args.panel)
    name = getattr(cfg, "name", "panel")
    # surface-curve PCs are optional (only ever solved for SPX); without them the
    # iv_curve rung is skipped and trees/exhaustion run on scalars alone
    curve_path = f"artifacts/{name}_surface_curve.parquet"
    has_curve = os.path.exists(curve_path)
    S = (pd.read_parquet(curve_path) if has_curve
         else pd.DataFrame(index=panel.index))
    if not has_curve:
        print(f"[note] {curve_path} missing -> iv_curve rung skipped\n")
    folds = make_walkforward(panel, cfg, n_folds=4)
    H = list(cfg.horizons)
    tgt = [f"tgt_rv_{h}" for h in H]

    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor

    def logc(a):
        return np.log(np.clip(np.asarray(a, dtype=float), 1e-12, None))

    def iv_col(h):
        return f"feat_atm_iv_{h}" if f"feat_atm_iv_{h}" in panel.columns else "feat_atm_iv"

    preds = {k: [] for k in ["naive_iv", "mz_iv", "iv_scalars", "iv_curve", "gbt_iv",
                             "hariv_x"]}
    y_all, mu_all = [], []
    for s in folds:
        va = s.val.intersection(S.index)
        tr = s.train.intersection(S.index)
        y_all.append(panel.loc[va, tgt].values)
        mu_all.append(np.tile(panel.loc[s.train, tgt].mean().values, (len(va), 1)))

        # full-filtration reference
        preds["hariv_x"].append(HARIVX(H).fit(panel, s.train).predict(panel, va))

        scal_tr = panel.loc[tr, [c for c in IV_SCALARS if c in panel.columns]].values
        scal_va = panel.loc[va, [c for c in IV_SCALARS if c in panel.columns]].values
        Ztr = np.column_stack([logc(scal_tr[:, :4]), scal_tr[:, 4:]])
        Zva = np.column_stack([logc(scal_va[:, :4]), scal_va[:, 4:]])
        if has_curve:
            pca = PCA(n_components=5).fit(S.loc[tr].values)
            Ztr = np.column_stack([Ztr, pca.transform(S.loc[tr].values)])
            Zva = np.column_stack([Zva, pca.transform(S.loc[va].values)])

        cols = {k: [] for k in ["naive_iv", "mz_iv", "iv_scalars", "iv_curve", "gbt_iv"]
                if has_curve or k != "iv_curve"}
        for j, h in enumerate(H):
            iv_tr, iv_va = panel.loc[tr, iv_col(h)].values, panel.loc[va, iv_col(h)].values
            ylog_tr = logc(panel.loc[tr, f"tgt_rv_{h}"].values)

            cols["naive_iv"].append(iv_va)
            m = Ridge(alpha=1e-6).fit(logc(iv_tr)[:, None], ylog_tr)
            cols["mz_iv"].append(np.exp(m.predict(logc(iv_va)[:, None])))
            m = Ridge(alpha=1e-2).fit(np.column_stack([logc(scal_tr[:, :4]), scal_tr[:, 4:]]), ylog_tr)
            cols["iv_scalars"].append(np.exp(m.predict(np.column_stack([logc(scal_va[:, :4]), scal_va[:, 4:]]))))
            if has_curve:
                m = Ridge(alpha=1e-2).fit(Ztr, ylog_tr)
                cols["iv_curve"].append(np.exp(m.predict(Zva)))
            g = HistGradientBoostingRegressor(max_iter=200, max_depth=3, random_state=7)
            g.fit(Ztr, ylog_tr)
            cols["gbt_iv"].append(np.exp(g.predict(Zva)))
        for k, v in cols.items():
            preds[k].append(np.column_stack(v))

    y = np.vstack(y_all)
    mu = np.vstack(mu_all)
    print(f"# IV-only protocol — {name}, {len(y)} pooled OOF val origins\n")

    print("## 1+2. Spectrum & ladder (per-horizon QLIKE; 'closed' = share of unconditional->floor gap)")
    order = [k for k in ["naive_iv", "mz_iv", "iv_scalars", "iv_curve", "gbt_iv", "hariv_x"]
             if has_curve or k != "iv_curve"]
    for j, h in enumerate(H):
        q_unc, q_flr = qlike(y[:, j], mu[:, j]), qlike(y[:, j], y[:, j])
        line = f"  h={h:2d}: unc {q_unc:7.3f} | "
        for k in order:
            qk = qlike(y[:, j], np.vstack(preds[k])[:, j])
            line += f"{k} {qk:7.3f} ({(q_unc-qk)/(q_unc-q_flr):5.1%}) | "
        print(line + f"floor {q_flr:7.3f}")

    print("\n## DM vs best classical IV-only rung (pooled horizons)")
    losses = {k: qlike_per_origin(y, np.vstack(preds[k])) for k in order}
    best_lin = min([k for k in ("mz_iv", "iv_scalars", "iv_curve") if k in losses],
                   key=lambda k: losses[k].mean())
    print(f"best classical IV-only rung: {best_lin}")
    for k in order:
        if k == best_lin:
            continue
        t, p = dm_test(losses[k], losses[best_lin], hac_lag=max(H) - 1)
        print(f"  {k:10s} vs {best_lin}: QLIKE {losses[k].mean():.4f} vs "
              f"{losses[best_lin].mean():.4f}, DM t={t:+.2f} p={p:.3f}")

    print("\n## 3. Exhaustion within G^IV (leave-one-fold-out residual test, best rung)")
    from sklearn.linear_model import Ridge as R2
    e_all, X_all, fold_id = [], [], []
    for fi, s in enumerate(folds):
        tr, va = s.train.intersection(S.index), s.val.intersection(S.index)
        e_all.append(logc(panel.loc[va, tgt].values) - logc(np.vstack([preds[best_lin][fi]])))
        Xf = panel.loc[va, [c for c in IV_SCALARS if c in panel.columns]].values
        if has_curve:
            pca = PCA(n_components=5).fit(S.loc[tr].values)
            Xf = np.column_stack([Xf, pca.transform(S.loc[va].values)])
        X_all.append(Xf)
        fold_id.append(np.full(len(va), fi))
    E, X, F = np.vstack(e_all), np.vstack(X_all), np.concatenate(fold_id)
    for j, h in enumerate(H):
        vals = []
        for fi in range(len(folds)):
            m = R2(alpha=1.0).fit(X[F != fi], E[F != fi, j])
            base = np.mean((E[F == fi, j] - E[F != fi, j].mean()) ** 2)
            vals.append(1.0 - np.mean((E[F == fi, j] - m.predict(X[F == fi])) ** 2) / base)
        v = np.array(vals)
        print(f"  h={h:2d}: LOFO R^2 {np.round(v, 4).tolist()}  mean {v.mean():+.4f}"
              f"{'  <- signal remains' if (v > 0).all() and v.mean() > 0.01 else ''}")


if __name__ == "__main__":
    main()
