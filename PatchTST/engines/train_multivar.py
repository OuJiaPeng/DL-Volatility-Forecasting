# Training script for multivariate PatchTST
# (Multiple assets/features)
# ...copy code from engines/train.py and update imports to use patchtst_multivar...
# TODO: Update data loading and model config for multivariate input

# PatchTST/train_multivar.py
import os
import sys
# add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from numpy.lib.stride_tricks import sliding_window_view

from utils.config import get_config
from models.patchtst_multivar import PatchTST
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Dataset for slicing multivariate time series into input/output pairs
class SliceDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, horizon: int):
        self.series = series.astype('float32')
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return max(0, len(self.series) - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len, :].T  # shape: (features, seq_len)
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.horizon, 0]  # target: first column
        return torch.from_numpy(x), torch.from_numpy(y)

# training function

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='PatchTST/default_multivar.yaml')
    parser.add_argument('--csv', type=str, default='compare/targets/with_all_targets.csv')
    parser.add_argument('--target_col', type=str, default='target_std')
    parser.add_argument('--feature_cols', nargs='+', default=None, help='List of feature columns to use (default: all except date)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_config(args.config)
    cfg.train.lr = float(cfg.train.lr)

    # load data
    df = pd.read_csv(args.csv, parse_dates=['date'], index_col='date').sort_index()
    if args.feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'date']
    else:
        feature_cols = args.feature_cols
    # Use all features for input, target_col for output
    X = df[feature_cols].values
    y = df[args.target_col].values
    # Normalize all features
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-8)
    y_mean, y_std = y.mean(), y.std()
    y = (y - y_mean) / (y_std + 1e-8)
    # Replace target column in X with normalized y (if needed)
    if args.target_col in feature_cols:
        X[:, feature_cols.index(args.target_col)] = y
    # split data into train/val/test
    seq_len, horizon = cfg.model.seq_len, cfg.model.out_horizon
    train_idx = (df.index >= '2014-01-01') & (df.index <= '2022-12-31')
    val_idx   = (df.index >= '2023-01-01') & (df.index <= '2023-12-31')
    test_idx  = (df.index >= '2024-01-01')
    train_ds = SliceDataset(X[train_idx], seq_len, horizon)
    val_ds   = SliceDataset(X[val_idx], seq_len, horizon)
    test_ds  = SliceDataset(X[test_idx], seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.train.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=cfg.train.batch_size)

    model = PatchTST(**cfg.model.__dict__).to(device)
    torch.nn.init.trunc_normal_(model.pos_embed, std=0.02)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=total_steps)

    writer = SummaryWriter(log_dir='checkpoints/logs/')
    os.makedirs('checkpoints/logs', exist_ok=True)

    best_val, patience = float('inf'), 0
    for epoch in range(cfg.train.epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = x.unsqueeze(0) if x.ndim == 2 else x  # ensure batch dimension
            pred = model(x).view(-1, horizon)
            loss = F.mse_loss(pred, y.view(-1, horizon))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * x.size(0)
        avg_tr = tr_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = x.unsqueeze(0) if x.ndim == 2 else x
                p = model(x).view(-1, horizon)
                val_loss += F.mse_loss(p, y.view(-1, horizon), reduction='sum').item()
        avg_val = val_loss / len(val_ds)

        writer.add_scalar('train/mse', avg_tr, epoch+1)
        writer.add_scalar('val/mse', avg_val, epoch+1)
        print(f"Epoch {epoch+1} — tr: {avg_tr:.5e}, val: {avg_val:.5e}, lr: {scheduler.get_last_lr()[0]:.2e}")

        torch.save(model.state_dict(), 'checkpoints/logs/patchtst.pt')
        if avg_val < best_val:
            best_val, patience = avg_val, 0
        else:
            patience += 1
            if patience >= 20:
                print("Early stopping")
                break

    # Test
    model.load_state_dict(torch.load('checkpoints/logs/patchtst.pt'))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x = x.unsqueeze(0) if x.ndim == 2 else x
            p = model(x).view(-1, horizon)
            all_preds.append(p.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    preds_unscaled = preds * y_std + y_mean

    # Save predictions
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/patch_preds_multivar.npy', preds_unscaled)

if __name__ == '__main__':
    train()
