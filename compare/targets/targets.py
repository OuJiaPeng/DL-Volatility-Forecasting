# compare/targets.py

import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm
import os
from pykalman import KalmanFilter

# sample variance and log transform
def compute_logvar_target(returns: np.ndarray, window: int) -> np.ndarray:
    return np.array([
        np.log(returns[i+1:i+1+window].var(ddof=1) + 1e-8)
        if i + 1 + window <= len(returns) else np.nan
        for i in range(len(returns))
    ])

# sample std dev
def compute_rolling_std(returns: np.ndarray, window: int) -> np.ndarray:
    return np.array([
        returns[i+1:i+1+window].std(ddof=1)
        if i + 1 + window <= len(returns) else np.nan
        for i in range(len(returns))
    ])

# EWMA w/ std  (ddof=0)
def compute_ewma_vol(df: pd.DataFrame, span: int) -> pd.Series:
    return df['return'].ewm(span=span, adjust=False).std()

# GARCH(1,1) forecast; one day ahead; forecast on raw returns; rescaling = False
def compute_garch_target(returns: np.ndarray, verbose: bool=False) -> pd.Series:
    garch_forecast = []
    for i in tqdm(range(len(returns)), disable=not verbose, desc="GARCH"):
        r = returns[:i]
        if len(r) < 100:
            garch_forecast.append(np.nan)
            continue

        model = arch_model(r, vol='Garch', p=1, q=1, rescale=False)
        try:
            res = model.fit(disp='off')
            var = res.forecast(horizon=1).variance.values[-1, 0]
            garch_forecast.append(np.sqrt(var))
        except:
            garch_forecast.append(np.nan)

    return pd.Series(garch_forecast)

# Kalman filter for volatility estimation (random walk model)
# # Model: observed returns ~ N(0, volatility), volatility follows random walk
def compute_kalman_vol(returns: np.ndarray) -> np.ndarray:

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=returns.var(),
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    # Use squared returns as proxy for variance
    state_means, _ = kf.filter(returns**2)
    return np.sqrt(np.maximum(state_means.flatten(), 0))

def generate_all_targets(df: pd.DataFrame, window: int, out_dir: str):
    df = df.copy()
    returns = df['return'].values
    os.makedirs(out_dir, exist_ok=True)

    df['target_logvar'] = compute_logvar_target(returns, window)
    df['target_std']    = compute_rolling_std(returns, window)
    df['target_ewma']   = compute_ewma_vol(df, span=window)
    garch_series = compute_garch_target(returns, verbose=True)
    garch_series.index = df.index
    df['target_garch']  = garch_series
    # Kalman filter volatility
    df['target_kalman'] = compute_kalman_vol(returns)

    # Save with all targets
    df.to_csv(os.path.join(out_dir, 'with_all_targets.csv'))
    print(f"Saved all targets to {os.path.join(out_dir, 'with_all_targets.csv')}")
    df[['target_logvar']].dropna().to_csv(f"{out_dir}/logvar.csv")
    df[['target_std']].dropna().to_csv(f"{out_dir}/rolling_std.csv")
    df[['target_ewma']].dropna().to_csv(f"{out_dir}/ewma.csv")
    df[['target_garch']].dropna().to_csv(f"{out_dir}/garch.csv")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/btc_2014_now.csv')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default='compare/targets')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['date'], index_col='date').sort_index()
    if 'return' not in df.columns:
        df['return'] = np.log(df['adjusted_close']).diff().fillna(0)

    generate_all_targets(df, args.window, args.out_dir)
