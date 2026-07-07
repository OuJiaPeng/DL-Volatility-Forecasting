"""HybridResidualForecaster: HAR prior + residual transformer, behind the Forecaster ABC.

``y_hat(t0, h) = exp( prior_log(t0, h) + residual_quantile(t0, h, q) )``

Implements the same ``fit(panel, train_idx)`` / ``predict(panel, origins)`` contract as
the baselines, so it plugs into ``eval.compare`` on identical origins. Early stopping
uses an INNER validation slice carved (with a purge gap) from the tail of the provided
train origins — the outer test set is never touched during fitting. ``n_runs`` models
with distinct seeds are trained in-process and their quantile forecasts averaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..baselines.base import Forecaster
from ..datasets import PanelWindowDataset
from ..seed import set_seed
from ..train.engine import train_model
from .prior import HARPrior
from .trunks import build_trunk


class HybridResidualForecaster(Forecaster):
    name = "hybrid"

    def __init__(self, horizons, mcfg):
        super().__init__(horizons)
        self.cfg = mcfg
        self.quantiles = list(getattr(mcfg, "quantiles", [0.1, 0.5, 0.9]))
        diffs = [abs(q - 0.5) for q in self.quantiles]
        self.median_idx = int(np.argmin(diffs))
        self.lookback = int(getattr(mcfg, "lookback", 30))
        self.n_runs = int(getattr(mcfg, "n_runs", 1))
        self.seed = int(getattr(mcfg, "seed", 7))
        self.device = str(getattr(mcfg, "device", "cpu"))
        self._split = None  # set via bind_split (normalization stats live with the split)
        self._cube = None   # optional raw intraday cube (E3), set via bind_intraday

    def bind_split(self, split) -> "HybridResidualForecaster":
        self._split = split
        return self

    def bind_intraday(self, cube) -> "HybridResidualForecaster":
        self._cube = cube
        return self

    # -- Forecaster contract -------------------------------------------------------
    def fit(self, panel: pd.DataFrame, train_idx: pd.DatetimeIndex) -> "HybridResidualForecaster":
        if self._split is None:
            raise RuntimeError("call bind_split(split) before fit (normalization stats)")
        cfg = self.cfg
        purge = max(self.horizons) + 5
        cut = int(len(train_idx) * 0.85)
        inner_train = train_idx[: max(cut - purge, 1)]
        inner_val = train_idx[cut:]

        prior_kind = str(getattr(cfg, "prior", "har"))
        self.prior = HARPrior(self.horizons, kind=prior_kind).fit(panel, train_idx)
        train_ds = self._dataset(panel, inner_train)
        val_ds = self._dataset(panel, inner_val)

        self.models = []
        trunk_name = str(getattr(cfg, "trunk", "itransformer"))
        for k in range(self.n_runs):
            seed = self.seed + k
            set_seed(seed)
            model = build_trunk(
                trunk_name,
                n_vars=train_ds.n_vars,
                lookback=self.lookback,
                n_surface=train_ds.n_surface,
                n_horizons=len(self.horizons),
                n_quantiles=len(self.quantiles),
                emb_dim=int(getattr(cfg, "emb_dim", 32)),
                n_heads=int(getattr(cfg, "n_heads", 4)),
                n_layers=int(getattr(cfg, "n_layers", 2)),
                dropout=float(getattr(cfg, "dropout", 0.1)),
            )
            model, best_val = train_model(
                model,
                train_ds,
                val_ds,
                quantiles=self.quantiles,
                median_idx=self.median_idx,
                lr=float(getattr(cfg, "lr", 1e-3)),
                epochs=int(getattr(cfg, "epochs", 60)),
                patience=int(getattr(cfg, "patience", 10)),
                batch_size=int(getattr(cfg, "batch_size", 64)),
                lambda_qlike=float(getattr(cfg, "lambda_qlike", 0.1)),
                seed=seed,
                device=self.device,
                verbose=bool(getattr(cfg, "verbose", False)),
            )
            self.models.append(model)

        # optional conformal (CQR) width calibration on the inner val slice: fixes the
        # measured under-coverage of the raw quantile band. Margin in log-vol space;
        # E_i = max(q_lo - y, y - q_hi), margin = (1-alpha)-quantile of scores.
        self._conf_m = 0.0
        if str(getattr(cfg, "conformal", "0")).lower() in ("1", "true", "yes"):
            q_val = self.predict_quantiles(panel, inner_val)          # (n, H, Q) vol units
            ql = np.log(np.clip(q_val, 1e-12, None))
            tgt_cols = [f"tgt_rv_{h}" for h in self.horizons]
            y_log = np.log(np.clip(panel.loc[inner_val, tgt_cols].values, 1e-12, None))
            scores = np.maximum(ql[..., 0] - y_log, y_log - ql[..., -1]).ravel()
            self._conf_m = float(np.quantile(scores, 0.80))
        return self

    def predict(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        """Point forecast = ensemble-mean median quantile, vol units, (n, H)."""
        q = self.predict_quantiles(panel, origins)
        return q[..., self.median_idx]

    # -- distributional output -----------------------------------------------------
    def predict_quantiles(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
        """Vol-space quantile forecasts (n, H, Q), averaged over the run ensemble."""
        ds = self._dataset(panel, origins)
        if len(ds.skipped):
            raise ValueError(
                f"{len(ds.skipped)} origin(s) lack {self.lookback} rows of panel history "
                f"(first: {ds.skipped[0]})"
            )
        x = torch.stack([ds[i][0] for i in range(len(ds))]).to(self.device)
        s = torch.stack([ds[i][1] for i in range(len(ds))]).to(self.device)
        intra = torch.stack([ds[i][2] for i in range(len(ds))]).to(self.device)
        prior = torch.stack([ds[i][3] for i in range(len(ds))]).to(self.device)

        outs = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                q_log = prior.unsqueeze(-1) + model(x, s, x_intra=intra)
                outs.append(q_log.cpu().numpy())
        q_log_mean = np.mean(outs, axis=0)
        m = getattr(self, "_conf_m", 0.0)
        if m:  # conformal band adjustment (outer quantiles only; median untouched)
            q_log_mean[..., 0] -= m
            q_log_mean[..., -1] += m
        self.last_quantiles_ = np.exp(q_log_mean)  # (n, H, Q) in vol units
        return self.last_quantiles_

    # -- internals -----------------------------------------------------------------
    def _dataset(self, panel, origins) -> PanelWindowDataset:
        from ..datasets import SURFACE_COLS, TENOR_SURFACE_COLS

        prior_log = self.prior.prior_log(panel, origins)
        surf = (TENOR_SURFACE_COLS if str(getattr(self.cfg, "surface_tokens", "base")) == "tenor"
                else SURFACE_COLS)
        return PanelWindowDataset(
            panel, origins, self._split, self.lookback, self.horizons, prior_log,
            surface_cols=surf, cube=self._cube,
            intra_days=int(getattr(self.cfg, "intra_days", 5)),
        )
