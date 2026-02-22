"""Benchmark target generation for volatility forecasting.

Computes realised-volatility targets (rolling std, log-var, EWMA)
and classical model forecasts (GARCH(1,1), Kalman filter) in both
full-window and rolling-window variants.

Usage:
    python compare/targets/targets.py [--csv PATH] [--window 5]
"""
import numpy as np
import pandas as pd
from arch import arch_model
from tqdm.auto import tqdm
import os
from pykalman import KalmanFilter

# Sample variance and log transform
def compute_logvar_target(returns: np.ndarray, window: int) -> np.ndarray:
    return np.array([
        np.log(returns[i+1:i+1+window].var(ddof=1) + 1e-8)
        if i + 1 + window <= len(returns) else np.nan
        for i in range(len(returns))
    ])

# Sample std dev
def compute_rolling_std(returns: np.ndarray, window: int) -> np.ndarray:
    return np.array([
        returns[i+1:i+1+window].std(ddof=1)
        if i + 1 + window <= len(returns) else np.nan
        for i in range(len(returns))
    ])

# EWMA w/ std (using default ddof=1 due to pandas version)
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
# Model: observed returns ~ N(0, volatility), volatility follows random walk
def compute_kalman_vol(returns: np.ndarray, verbose: bool = True) -> np.ndarray:

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=returns.var(),
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    
    # use squared returns as proxy for variance
    observations = returns**2
    n_timesteps = len(observations)
    state_means = np.zeros(n_timesteps)

    # iterate through observations to use tqdm
    filtered_state_mean = kf.initial_state_mean
    filtered_state_covariance = kf.initial_state_covariance

    for t in tqdm(range(n_timesteps), disable=not verbose, desc="Kalman"):
        filtered_state_mean, filtered_state_covariance = kf.filter_update(
            filtered_state_mean,
            filtered_state_covariance,
            observation=observations[t]
        )
        state_means[t] = filtered_state_mean

    return np.sqrt(np.maximum(state_means.flatten(), 0))

# Rolling forecast GARCH(1,1)
# fits a GARCH model on the preceding `rolling_window` of returns and forecasts `horizon` steps ahead
def compute_garch_rolling_forecast(
    returns: np.ndarray,
    horizon: int,
    rolling_window: int,
    test_indices: np.ndarray,
    verbose: bool = True
) -> np.ndarray:

    all_forecasts = []
    for i in tqdm(test_indices, disable=not verbose, desc="Rolling GARCH"):
        train_returns = returns[i - rolling_window : i]
        if len(train_returns) < rolling_window:
            all_forecasts.append(np.full(horizon, np.nan))
            continue

        try:
            model = arch_model(train_returns, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=horizon, reindex=False)
            vol_forecast = np.sqrt(forecast.variance.values.flatten())
            all_forecasts.append(vol_forecast)
        except Exception:
            all_forecasts.append(np.full(horizon, np.nan))

    return np.array(all_forecasts)

# Rolling forecast Kalman filter
# uses the last state as the forecast for all future steps
def compute_kalman_rolling_forecast(
    returns: np.ndarray,
    horizon: int,
    rolling_window: int,
    test_indices: np.ndarray,
    verbose: bool = True
) -> np.ndarray:

    all_forecasts = []
    for i in tqdm(test_indices, disable=not verbose, desc="Rolling Kalman"):
        window_returns = returns[i - rolling_window : i]
        if len(window_returns) < rolling_window:
            all_forecasts.append(np.full(horizon, np.nan))
            continue

        try:
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=window_returns.var(),
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=0.01
            )
            state_means, _ = kf.filter(window_returns**2)
            last_vol = np.sqrt(np.maximum(state_means[-1, 0], 0))
            all_forecasts.append(np.full(horizon, last_vol))
        except Exception:
            all_forecasts.append(np.full(horizon, np.nan))

    return np.array(all_forecasts)


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
    df['target_kalman'] = compute_kalman_vol(returns, verbose=True)

    # save with all targets
    df.to_csv(os.path.join(out_dir, 'with_all_targets.csv'))
    print(f"Saved all targets to {os.path.join(out_dir, 'with_all_targets.csv')}")
    df[['target_logvar']].dropna().to_csv(f"{out_dir}/logvar.csv")
    df[['target_std']].dropna().to_csv(f"{out_dir}/rolling_std.csv")
    df[['target_ewma']].dropna().to_csv(f"{out_dir}/ewma.csv")
    df[['target_garch']].dropna().to_csv(f"{out_dir}/garch.csv")
    df[['target_kalman']].dropna().to_csv(f"{out_dir}/kalman.csv")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/btc_2014_now.csv')
    parser.add_argument('--window', type=int, default=5, help="Window for realized vol targets.")
    parser.add_argument('--out_dir', type=str, default='compare/targets')

    # new arguments for rolling forecasts
    parser.add_argument('--skip_rolling', action='store_true', help="Skip rolling GARCH/Kalman forecasts.")
    parser.add_argument('--horizon', type=int, default=5, help="Forecast horizon for rolling models.")
    parser.add_argument('--rolling_window', type=int, default=100, help="Lookback window for rolling models.")
    parser.add_argument('--test_start', type=str, default='2024-01-01', help="Start date for test set.")

    args = parser.parse_args()

    df = pd.read_csv(args.csv, parse_dates=['date'], index_col='date').sort_index()
    if 'return' not in df.columns:
        df['return'] = np.log(df['adjusted_close']).diff().fillna(0)

    print("Generating all targets for the full series...")
    generate_all_targets(df, args.window, args.out_dir)

    if not args.skip_rolling:
        print("Computing rolling forecasts...")
        returns = df['return'].values
        try:
            test_start_idx = df.index.get_loc(args.test_start)
        except KeyError:
            raise KeyError(f"Test start date '{args.test_start}' not found in the data index.")
        test_indices = np.arange(test_start_idx, len(df))

        # compute and save rolling GARCH
        garch_rolling = compute_garch_rolling_forecast(
            returns, args.horizon, args.rolling_window, test_indices
        )
        garch_path = os.path.join(args.out_dir, 'garch_rolling.npy')
        np.save(garch_path, garch_rolling)
        print(f"Saved rolling GARCH forecasts to {garch_path}")

        # compute and save rolling Kalman
        kalman_rolling = compute_kalman_rolling_forecast(
            returns, args.horizon, args.rolling_window, test_indices
        )
        kalman_path = os.path.join(args.out_dir, 'kalman_rolling.npy')
        np.save(kalman_path, kalman_rolling)
        print(f"Saved rolling Kalman forecasts to {kalman_path}")
