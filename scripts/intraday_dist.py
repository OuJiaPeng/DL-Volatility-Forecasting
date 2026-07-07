"""Distributional chapter, oracle-first: is the RESIDUAL SCALE predictable?

The point-forecast gate is closed (E[e|X] = 0 vs ridge, GBT, shape features).
This tests the next moment: license = can X predict log(e^2)? If yes, a
distributional model has a legitimate target the point champion can't hit.
Then the quantile ladder on pinball loss [0.1, 0.5, 0.9] + 80% coverage:
  gauss_const  mu = aug-ridge, sigma = train residual std (constant)
  gauss_het    sigma(X) from linear log(e^2) model  <- linear-once-named bar
  gbt_q        gradient-boosted quantile regression (capacity)
  nn_q         MLP quantile head, pinball-trained (capacity)

Usage: python scripts/intraday_dist.py [--panel artifacts/intraday_spx.parquet]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from volforecast.intraday import aug_features, day_folds, fit_oos

QS = [0.1, 0.5, 0.9]


def pinball(y, q_pred, q):
    d = y - q_pred
    return np.mean(np.maximum(q * d, (q - 1) * d))


def report(name, y, Q, fold):
    pb = {q: pinball(y, Q[:, j], q) for j, q in enumerate(QS)}
    cov = np.mean((y >= Q[:, 0]) & (y <= Q[:, 2]))
    covf = [np.mean((y[fold == k] >= Q[fold == k, 0]) & (y[fold == k] <= Q[fold == k, 2]))
            for k in range(4)]
    print(f"    {name:12s} pinball {sum(pb.values()):.4f} "
          f"(q10 {pb[0.1]:.4f} | q50 {pb[0.5]:.4f} | q90 {pb[0.9]:.4f})  "
          f"cov80 {cov:.3f} folds {np.round(covf, 3).tolist()}")
    return sum(pb.values())


def nn_quantiles(X, y, fold, train_mask, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    qs = torch.tensor(QS)
    out = np.full((len(y), 3), np.nan)
    for k in range(4):
        tr, te = train_mask(k), fold == k
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)[:, None]
        Xte = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(net[-1].weight)
        with torch.no_grad():
            net[-1].bias.copy_(torch.quantile(ytr, qs))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(300):
            opt.zero_grad()
            d = ytr - net(Xtr)
            loss = torch.mean(torch.maximum(qs * d, (qs - 1) * d))
            loss.backward()
            opt.step()
        with torch.no_grad():
            out[te] = np.sort(net(Xte).numpy(), axis=1)  # monotone quantiles
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="artifacts/intraday_spx.parquet")
    args = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor as GBT

    df = pd.read_parquet(args.panel)
    y = df["y"].values
    X = aug_features(df)
    fold, tm = day_folds(df)

    p = fit_oos(X, y, fold, tm, lambda: Ridge(alpha=1.0))
    e = y - p
    print(f"point champion (aug-ridge) OOS MSE {np.mean(e ** 2):.4f}")

    # ---- license: is log(e^2) predictable? ------------------------------------------
    z = np.log(e ** 2 + 1e-10)
    print("\nscale license: fold-wise OOS R^2 predicting log(e^2):")
    for name, fn in [("ridge", lambda: Ridge(alpha=1.0)),
                     ("gbt", lambda: GBT(random_state=0, max_iter=300, learning_rate=0.06))]:
        pz = fit_oos(X, z, fold, tm, fn)
        r2 = [1 - np.mean((z[fold == k] - pz[fold == k]) ** 2)
              / np.mean((z[fold == k] - z[tm(k)].mean()) ** 2) for k in range(4)]
        v = np.array(r2)
        print(f"    {name:6s} folds {np.round(v, 4).tolist()}  mean {v.mean():+.4f}"
              f"  -> {'LICENSED' if (v > 0).all() and v.mean() > 0.01 else 'no license'}")

    # ---- quantile ladder --------------------------------------------------------------
    print("\nquantile ladder (pinball on log RV, 80% interval coverage):")
    zq = norm.ppf(QS)

    Qc = np.full((len(y), 3), np.nan)
    for k in range(4):
        s = e[tm(k)].std()
        Qc[fold == k] = p[fold == k, None] + s * zq[None, :]
    report("gauss_const", y, Qc, fold)

    pz = fit_oos(X, z, fold, tm, lambda: Ridge(alpha=1.0))
    Qh = np.full((len(y), 3), np.nan)
    for k in range(4):
        te = fold == k
        # per-fold calibration: E[e^2] = exp(pz)*c, c fixes Jensen gap on train
        c = np.mean(e[tm(k)] ** 2) / np.mean(np.exp(pz[tm(k)])) if np.isfinite(
            pz[tm(k)]).all() else 1.0
        sig = np.sqrt(np.exp(pz[te]) * c)
        Qh[te] = p[te, None] + sig[:, None] * zq[None, :]
    report("gauss_het", y, Qh, fold)

    Qg = np.full((len(y), 3), np.nan)
    for j, q in enumerate(QS):
        Qg[:, j] = fit_oos(X, y, fold, tm,
                           lambda q=q: GBT(random_state=0, max_iter=300, learning_rate=0.06,
                                           loss="quantile", quantile=q))
    Qg = np.sort(Qg, axis=1)
    report("gbt_q", y, Qg, fold)

    Qn = nn_quantiles(X, y, fold, tm)
    report("nn_q", y, Qn, fold)

    # residual-scale drivers, for the writeup
    pz_gbt = fit_oos(X, z, fold, tm,
                     lambda: GBT(random_state=0, max_iter=300, learning_rate=0.06))
    hi = pd.Series(pz_gbt, index=df.index).groupby(df.block.values).mean()
    print("\npredicted log-scale by time-of-day block (higher = fatter residuals):")
    print("   ", {int(b): round(v, 2) for b, v in hi.items()})


if __name__ == "__main__":
    main()
