"""Single seeded training loop with early stopping on validation pinball loss.

The in-process ensemble (n_runs models, distinct seeds) lives in ``models/hybrid.py`` —
deliberately NOT the legacy subprocess-and-regex-scrape pattern, which silently NaN-ed
every run when a print string drifted.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader

from ..models.losses import combined_loss, pinball_loss


def train_model(
    model,
    train_ds,
    val_ds,
    quantiles,
    median_idx: int,
    lr: float = 1e-3,
    epochs: int = 60,
    patience: int = 10,
    batch_size: int = 64,
    lambda_qlike: float = 0.1,
    weight_decay: float = 1e-4,
    seed: int = 7,
    device: str = "cpu",
    verbose: bool = False,
):
    device = torch.device(device)
    model = model.to(device)
    q = torch.tensor(quantiles, dtype=torch.float32, device=device)

    gen = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_dl = DataLoader(val_ds, batch_size=256)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        for x, s, intra, prior, y in train_dl:
            x, s, intra = x.to(device), s.to(device), intra.to(device)
            prior, y = prior.to(device), y.to(device)
            opt.zero_grad()
            q_log = prior.unsqueeze(-1) + model(x, s, x_intra=intra)
            loss = combined_loss(y, q_log, q, median_idx, lambda_qlike)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for x, s, intra, prior, y in val_dl:
                x, s, intra = x.to(device), s.to(device), intra.to(device)
                prior, y = prior.to(device), y.to(device)
                q_log = prior.unsqueeze(-1) + model(x, s, x_intra=intra)
                val_loss += pinball_loss(y, q_log, q).item() * len(y)
                n += len(y)
        val_loss /= max(n, 1)
        if verbose:
            print(f"  epoch {epoch + 1}: val pinball {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val
