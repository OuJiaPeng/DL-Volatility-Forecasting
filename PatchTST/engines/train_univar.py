# PatchTST/train_univar.py

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
from models.patchtst_univar import PatchTST
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# dataset for slicing time series into input / output pairs
class SliceDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, horizon: int):
        self.series = series.astype('float32')
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return max(0, len(self.series) - self.seq_len - self.horizon + 1)

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)

# training function
def train():
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='PatchTST/default_univar.yaml')
    parser.add_argument('--target_col', type=str, default='target_std')
    parser.add_argument('--csv', type=str, default='compare/targets/with_all_targets.csv')
    args = parser.parse_args()

    # sets device and loads config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_config(args.config)
    cfg.train.lr = float(cfg.train.lr)

    # load and normalize data
    df = pd.read_csv(args.csv, parse_dates=['date'], index_col='date').sort_index()
    raw_target = df[args.target_col].copy()
    mean, std = raw_target.mean(), raw_target.std()
    df[args.target_col] = (raw_target - mean) / std
    df = df.dropna(subset=[args.target_col])

    # split data into train/val/test
    seq_len, horizon = cfg.model.seq_len, cfg.model.out_horizon
    train_df = df['2014-01-01':'2022-12-31']
    val_df   = df['2023-01-01':'2023-12-31']
    today    = datetime.today().strftime('%Y-%m-%d')
    test_df  = df['2024-01-01': today]

    # create datasets
    train_ds = SliceDataset(train_df[args.target_col].values, seq_len, horizon)
    val_ds   = SliceDataset(val_df[args.target_col].values, seq_len, horizon)
    test_ds  = SliceDataset(test_df[args.target_col].values, seq_len, horizon)

    # create data loaders
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.train.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=cfg.train.batch_size)

    # initialize model and optimizer
    model = PatchTST(**cfg.model.__dict__).to(device)
    torch.nn.init.trunc_normal_(model.pos_embed, std=0.02)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-4)
    total_steps = cfg.train.epochs * len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.train.lr, total_steps=total_steps)

    # sets up TensorBoard writer
    log_dir = 'checkpoints/logs/'
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Ensure output directory exists for univar outputs
    output_dir = 'outputs/univar_outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Ensure checkpoint directory exists for univar checkpoints
    checkpoint_dir = 'checkpoints/univar'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # training loop with early stopping
    best_val, patience = float('inf'), 0
    for epoch in range(cfg.train.epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x).view(-1, horizon)
            loss = F.mse_loss(pred, y.view(-1, horizon))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item() * x.size(0)
        avg_tr = tr_loss / len(train_ds)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x).view(-1, horizon)
                val_loss += F.mse_loss(p, y.view(-1, horizon), reduction='sum').item()
        avg_val = val_loss / len(val_ds)

        # metrics
        writer.add_scalar('train/mse', avg_tr, epoch+1)
        writer.add_scalar('val/mse', avg_val, epoch+1)
        print(f"Epoch {epoch+1} — tr: {avg_tr:.5e}, val: {avg_val:.5e}, lr: {scheduler.get_last_lr()[0]:.2e}")

        # save last model checkpoint every epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'last_univar_patchtst.pt'))
        # save best model checkpoint
        if avg_val < best_val:
            best_val, patience = avg_val, 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'univar_patchtst.pt'))
        else:
            patience += 1
            if patience >= 20:
                print("Early stopping")
                break

    # test loop; generate predictions on test set
    # Load best model for test
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'univar_patchtst.pt')))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            p = model(x).view(-1, horizon)
            all_preds.append(p.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    preds_unscaled = preds * std + mean

    # calculate real-vol MSE (all steps in window)
    raw = raw_target.loc[test_df.index].values
    windows = sliding_window_view(raw, window_shape=horizon)[:len(preds)]
    real_mse = ((windows - preds_unscaled)**2).mean()
    print(f"Real-vol MSE (all steps): {real_mse:.5e}")

    # calculate last-step MSE (to match plotting script)
    # Align last-step predictions with actuals
    last_step_preds = preds_unscaled[:, -1]
    last_step_actuals = raw[-len(last_step_preds):]
    last_step_mse = ((last_step_actuals - last_step_preds) ** 2).mean()
    print(f"Last-step MSE: {last_step_mse:.5e}")

    # Save predictions (univariate only)
    np.save(os.path.join(output_dir, 'patch_preds_univar.npy'), preds_unscaled)

if __name__ == '__main__':
    train()
