"""Non-torch E2 arms: state-dependent HAR and a gradient-boosted-trees control.

StateDependentHAR — the "HAR whose coefficients breathe with the regime" idea in its
linear form: beta(z) = beta0 + W z applied to the HAR features X is algebraically a
ridge regression on [X, z (x) X] (interactions). Tiny, interpretable, nests plain HAR
exactly at W = 0 (which ridge shrinkage favors — the same nesting philosophy as the
neural arms' zero-init).

GradientBoostedTrees — the tabular control. The small-n literature says trees beat
nets on problems this size embarrassingly often; if this wins E2, we want to be the
ones who found out. sklearn's HistGradientBoosting keeps it dependency-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.base import Forecaster

HAR_COLS = ["feat_rv_d", "feat_rv_w", "feat_rv_m"]
STATE_COLS = ["feat_atm_iv", "feat_skew", "feat_term_slope", "feat_vix",
              "feat_jump_share", "feat_semi_neg_share"]
VOL_FLOOR = 1e-6


class StateDependentHAR(Forecaster):
    name = "statehar"

    def __init__(self, horizons, alpha: float = 1.0):
        super().__init__(horizons)
        self.alpha = alpha

    def _design(self, panel: pd.DataFrame, idx) -> np.ndarray:
        X = panel.loc[idx, HAR_COLS].values.astype(float)                    # (n, 3)
        cols = [c for c in STATE_COLS if c in panel.columns]
        Z = panel.loc[idx, cols].values.astype(float)                        # (n, k)
        Z = (Z - self._z_mean) / self._z_std                                 # train-stat z-score
        inter = (Z[:, :, None] * X[:, None, :]).reshape(len(X), -1)          # (n, k*3)
        return np.column_stack([np.ones(len(X)), X, inter])

    def fit(self, panel, train_idx):
        from sklearn.linear_model import Ridge

        cols = [c for c in STATE_COLS if c in panel.columns]
        Ztr = panel.loc[train_idx, cols].values.astype(float)
        self._z_mean = Ztr.mean(axis=0)
        self._z_std = np.where(Ztr.std(axis=0) > 0, Ztr.std(axis=0), 1.0)
        D = self._design(panel, train_idx)
        self.models = {}
        for h in self.horizons:
            y = panel.loc[train_idx, f"tgt_rv_{h}"].values.astype(float)
            m = Ridge(alpha=self.alpha, fit_intercept=False)
            m.fit(D, y)
            self.models[h] = m
        return self

    def predict(self, panel, origins):
        D = self._design(panel, origins)
        cols = [np.clip(self.models[h].predict(D), VOL_FLOOR, None) for h in self.horizons]
        return np.column_stack(cols)


class HARIV(Forecaster):
    """HAR-X with implied vol: per-horizon ridge on [HAR features + tenor-matched IV].

    One of the few literature-backed upgrades over plain HAR (implied vol subsumes
    information daily RV history cannot see). Doubles as the 'har_iv' PRIOR for the
    hybrid chassis — raising the nesting floor lifts every neural arm above it.
    """
    name = "har_iv"

    def __init__(self, horizons, alpha: float = 1e-4):
        super().__init__(horizons)
        self.alpha = alpha

    def _iv_col(self, panel, h):
        c = f"feat_atm_iv_{h}"
        return c if c in panel.columns else "feat_atm_iv"

    def _X(self, panel, idx, h):
        return np.column_stack([
            np.ones(len(idx)),
            panel.loc[idx, HAR_COLS].values.astype(float),
            panel.loc[idx, self._iv_col(panel, h)].values.astype(float),
        ])

    def fit(self, panel, train_idx):
        from sklearn.linear_model import Ridge

        self.models = {}
        for h in self.horizons:
            y = panel.loc[train_idx, f"tgt_rv_{h}"].values.astype(float)
            m = Ridge(alpha=self.alpha, fit_intercept=False)
            m.fit(self._X(panel, train_idx, h), y)
            self.models[h] = m
        return self

    def predict(self, panel, origins):
        cols = [np.clip(self.models[h].predict(self._X(panel, origins, h)), VOL_FLOOR, None)
                for h in self.horizons]
        return np.column_stack(cols)


class HARIVX(HARIV):
    """HAR-IV with a state-dependent IV discount: adds an IV x z(VIX) interaction.

    The linear form of 'estimate when the risk premium is fat and discount harder' —
    the one structure the probes found beyond HAR-IV. Six coefficients per horizon.
    """
    name = "hariv_x"

    def fit(self, panel, train_idx):
        v = panel.loc[train_idx, "feat_vix"].astype(float)
        self._vmu = float(v.mean())
        self._vsd = float(v.std()) or 1.0
        return super().fit(panel, train_idx)

    def _X(self, panel, idx, h):
        X = super()._X(panel, idx, h)
        vz = ((panel.loc[idx, "feat_vix"].astype(float) - self._vmu) / self._vsd).values
        return np.column_stack([X, X[:, -1] * vz])   # IV x standardized-VIX


class HARIVM(HARIV):
    """HAR-IV + MARKET-state correction (additive z-VIX term).

    Licensed by the NVDA residual diagnostic (Jul 2026): champion errors load
    negatively on VIX, growing with horizon — single-name IV carries systemic-fear
    premium that doesn't realize in single-name vol, so the discount deepens with
    market state. The cross-asset channel in its minimal linear form.
    """
    name = "har_iv_m"

    def fit(self, panel, train_idx):
        v = panel.loc[train_idx, "feat_vix"].astype(float)
        self._vmu = float(v.mean())
        self._vsd = float(v.std()) or 1.0
        return super().fit(panel, train_idx)

    def _X(self, panel, idx, h):
        X = super()._X(panel, idx, h)
        vz = ((panel.loc[idx, "feat_vix"].astype(float) - self._vmu) / self._vsd).values
        return np.column_stack([X, vz])


class WedgeGBT(Forecaster):
    """Predict the premium wedge W = IV_h - y_h with trees; forecast y = IV - What.

    (The LINEAR wedge regression is algebraically identical to HARIV, so the wedge
    idea only exists in nonlinear form.) Anchoring on IV means the trees model only
    premium variation, not the vol level.
    """
    name = "wedge_gbt"

    def __init__(self, horizons, seed: int = 7):
        super().__init__(horizons)
        self.seed = seed

    def _iv(self, panel, idx, h):
        c = f"feat_atm_iv_{h}"
        return panel.loc[idx, c if c in panel.columns else "feat_atm_iv"].values.astype(float)

    def fit(self, panel, train_idx):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.feat_cols = [c for c in panel.columns if c.startswith("feat_")]
        X = panel.loc[train_idx, self.feat_cols].values.astype(float)
        self.models = {}
        for h in self.horizons:
            w = self._iv(panel, train_idx, h) - panel.loc[train_idx, f"tgt_rv_{h}"].values
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                              max_depth=3, random_state=self.seed)
            m.fit(X, w)
            self.models[h] = m
        return self

    def predict(self, panel, origins):
        X = panel.loc[origins, self.feat_cols].values.astype(float)
        cols = [np.clip(self._iv(panel, origins, h) - self.models[h].predict(X),
                        VOL_FLOOR, None) for h in self.horizons]
        return np.column_stack(cols)


class EWAAggregator(Forecaster):
    """Online exponentially-weighted aggregation over classical experts.

    Weights at origin t use only losses from origins whose target windows have fully
    CLOSED by t (delay = max horizon + 1) — the causal bookkeeping matters more than
    the eta. Eta fixed a priori (no tuning; regret bounds, not optimization).
    """
    name = "ewa"

    def __init__(self, horizons, eta: float = 2.0):
        super().__init__(horizons)
        self.eta = eta

    def fit(self, panel, train_idx):
        from ..baselines.har_rv import HARRV

        self.experts = {"har_rv": HARRV(self.horizons).fit(panel, train_idx),
                        "har_iv": HARIV(self.horizons).fit(panel, train_idx),
                        "hariv_x": HARIVX(self.horizons).fit(panel, train_idx)}
        return self

    def predict(self, panel, origins):
        from ..eval.metrics import qlike_per_origin

        delay = max(self.horizons) + 1
        pos = {t: i for i, t in enumerate(panel.index)}
        tgt_cols = [f"tgt_rv_{h}" for h in self.horizons]
        # expert forecasts for ALL panel rows once (cheap, linear models)
        all_idx = panel.index
        preds = {k: e.predict(panel, all_idx) for k, e in self.experts.items()}
        y_all = panel[tgt_cols].values
        losses = {k: qlike_per_origin(y_all, p) for k, p in preds.items()}

        out = []
        names = list(self.experts)
        for t in origins:
            p = pos[t]
            cum = np.array([np.sum(losses[k][: max(p - delay, 0)]) for k in names])
            w = np.exp(-self.eta * (cum - cum.min()))
            w = w / w.sum()
            out.append(sum(w[i] * preds[names[i]][p] for i in range(len(names))))
        return np.asarray(out)


class TabPFNArm(Forecaster):
    """TabPFN in-context tabular regression (no gradient steps on our data).

    Requires `pip install tabpfn`. Fits log-vol per horizon; the different failure
    mode (prior-fitted ICL vs SGD) is the point — immune to residual overfitting.
    """
    name = "tabpfn"

    def fit(self, panel, train_idx):
        self.feat_cols = [c for c in panel.columns if c.startswith("feat_")]
        self._Xtr = panel.loc[train_idx, self.feat_cols].values.astype(np.float32)
        self._ytr = {h: np.log(np.clip(panel.loc[train_idx, f"tgt_rv_{h}"].values,
                                       VOL_FLOOR, None)) for h in self.horizons}
        return self

    def predict(self, panel, origins):
        from tabpfn import TabPFNRegressor

        X = panel.loc[origins, self.feat_cols].values.astype(np.float32)
        cols = []
        for h in self.horizons:
            reg = TabPFNRegressor(device="cpu", random_state=7)
            reg.fit(self._Xtr, self._ytr[h])
            cols.append(np.exp(reg.predict(X)))
        return np.column_stack(cols)


class GradientBoostedTrees(Forecaster):
    name = "gbt"

    def __init__(self, horizons, max_iter: int = 300, learning_rate: float = 0.05,
                 max_depth: int = 3, seed: int = 7):
        super().__init__(horizons)
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate,
                           max_depth=max_depth, random_state=seed)

    def fit(self, panel, train_idx):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.feat_cols = [c for c in panel.columns if c.startswith("feat_")]
        X = panel.loc[train_idx, self.feat_cols].values.astype(float)
        self.models = {}
        for h in self.horizons:
            y = np.log(np.clip(panel.loc[train_idx, f"tgt_rv_{h}"].values.astype(float),
                               VOL_FLOOR, None))
            m = HistGradientBoostingRegressor(**self.params)
            m.fit(X, y)                      # log-vol space, matching the neural arms
            self.models[h] = m
        return self

    def predict(self, panel, origins):
        X = panel.loc[origins, self.feat_cols].values.astype(float)
        cols = [np.exp(self.models[h].predict(X)) for h in self.horizons]
        return np.column_stack(cols)
