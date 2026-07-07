"""Neural gate: y = HAR-part + g(state) * IV — the learned state-dependent discount.

Nests hariv_x's linear gate: the HAR betas and the gate's linear part warm-start from
the fitted HARIV/HARIVX solution; the MLP's output layer is zero-init, so training
begins EXACTLY at the linear champion and must earn any nonlinearity. Trained per
fold with early stopping on an inner purged slice, averaged over seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baselines.base import Forecaster
from .classical_arms import HAR_COLS, HARIV, VOL_FLOOR

GATE_STATE = ["feat_vix", "feat_term_slope", "feat_skew", "feat_jump_share",
              "feat_semi_neg_share", "feat_rv_d"]


class NeuralGate(Forecaster):
    name = "gate"

    def __init__(self, horizons, hidden: int = 16, epochs: int = 200, lr: float = 3e-3,
                 patience: int = 25, n_runs: int = 5, seed: int = 7):
        super().__init__(horizons)
        self.hidden, self.epochs, self.lr = hidden, epochs, lr
        self.patience, self.n_runs, self.seed = patience, n_runs, seed

    def _tensors(self, panel, idx):
        import torch

        cols = [c for c in GATE_STATE if c in panel.columns]
        Z = panel.loc[idx, cols].values.astype(np.float32)
        Z = (Z - self._z_mu) / self._z_sd
        X = panel.loc[idx, HAR_COLS].values.astype(np.float32)
        IV = np.column_stack([
            panel.loc[idx, f"feat_atm_iv_{h}" if f"feat_atm_iv_{h}" in panel.columns
                       else "feat_atm_iv"].values for h in self.horizons
        ]).astype(np.float32)
        return (torch.from_numpy(Z), torch.from_numpy(X), torch.from_numpy(IV))

    def fit(self, panel, train_idx):
        import torch
        import torch.nn as nn

        cols = [c for c in GATE_STATE if c in panel.columns]
        Ztr = panel.loc[train_idx, cols].values.astype(np.float32)
        self._z_mu, self._z_sd = Ztr.mean(0), np.where(Ztr.std(0) > 0, Ztr.std(0), 1.0)

        # warm start: the linear champion's coefficients
        lin = HARIV(self.horizons).fit(panel, train_idx)
        beta0 = np.stack([lin.models[h].coef_ for h in self.horizons])  # (H, 5)

        purge = max(self.horizons) + 5
        cut = int(len(train_idx) * 0.85)
        inner_tr, inner_va = train_idx[: max(cut - purge, 1)], train_idx[cut:]
        H = len(self.horizons)
        y_tr = torch.from_numpy(panel.loc[inner_tr, [f"tgt_rv_{h}" for h in self.horizons]]
                                .values.astype(np.float32))
        y_va = torch.from_numpy(panel.loc[inner_va, [f"tgt_rv_{h}" for h in self.horizons]]
                                .values.astype(np.float32))
        t_tr, t_va = self._tensors(panel, inner_tr), self._tensors(panel, inner_va)

        class Gate(nn.Module):
            def __init__(self, k, hidden, beta0):
                super().__init__()
                self.har = nn.Parameter(torch.tensor(beta0[:, :4], dtype=torch.float32))  # (H,4): int+3
                self.g0 = nn.Parameter(torch.tensor(beta0[:, 4], dtype=torch.float32))    # (H,)
                self.mlp = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, H))
                nn.init.zeros_(self.mlp[-1].weight)
                nn.init.zeros_(self.mlp[-1].bias)

            def forward(self, Z, X, IV):
                ones = torch.ones(len(X), 1)
                har = torch.cat([ones, X], 1) @ self.har.T          # (n, H)
                g = self.g0.unsqueeze(0) + self.mlp(Z)              # (n, H)
                return har + g * IV

        self.nets, self._val = [], []
        for k in range(self.n_runs):
            torch.manual_seed(self.seed + k)
            net = Gate(len(cols), self.hidden, beta0)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best, best_state, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = ((net(*t_tr) - y_tr) ** 2).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    v = float(((net(*t_va) - y_va) ** 2).mean())
                if v < best - 1e-12:
                    best, best_state, bad = v, {n: p.clone() for n, p in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            net.load_state_dict(best_state)
            self.nets.append(net)
        return self

    def predict(self, panel, origins):
        import torch

        t = self._tensors(panel, origins)
        with torch.no_grad():
            preds = np.mean([net(*t).numpy() for net in self.nets], axis=0)
        return np.clip(preds, VOL_FLOOR, None)
