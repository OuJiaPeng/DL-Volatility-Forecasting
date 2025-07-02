#outputs/univar_outputs/plot_preds_univar.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from PatchTST.utils.config import get_config

# Load config to get horizon and seq_len
cfg = get_config('./PatchTST/utils/default_univar.yaml')
CSV_PATH    = "./compare/targets/with_all_targets.csv"
TARGET_COL  = "target_std"
GARCH_FULL_COL = "target_garch"
KALMAN_FULL_COL = "target_kalman"
HORIZON     = cfg.model.out_horizon
SEQ_LEN     = cfg.model.seq_len
# Prediction paths
PATCH_PRED_PATH   = "./outputs/univar_outputs/patchtst_preds/patch_preds_univar.npy"
GARCH_ROLL_PATH = "./compare/targets/garch_rolling.npy"
KALMAN_ROLL_PATH= "./compare/targets/kalman_rolling.npy"
# Output paths
OUTPUT_PLOT = "./outputs/univar_outputs/realized_vol_prediction_plot.png"
PLOT_WINDOW = None  # Show full test window if None
METRICS_OUT_PATH = './outputs/univar_outputs/metrics_comparison.csv'
# Try to use mean prediction from run_n_times if available
MEAN_PRED_PATH = "./outputs/univar_outputs/patchtst_preds/patch_preds_univar_mean.npy"
MSE_OUT_PATH = "./outputs/univar_outputs/patchtst_preds/patchtst_mses.npy"

# Load data
df = pd.read_csv(CSV_PATH, parse_dates=['date'], index_col='date').sort_index()
if TARGET_COL not in df.columns:
    raise ValueError(f"Missing {TARGET_COL}")
if GARCH_FULL_COL not in df.columns or KALMAN_FULL_COL not in df.columns:
    raise ValueError(f"Missing full-window benchmark columns ('{GARCH_FULL_COL}', '{KALMAN_FULL_COL}') in {CSV_PATH}")


# Load predictions
try:
    # Prefer the mean prediction if it exists
    if os.path.exists(MEAN_PRED_PATH):
        print("Loading mean predictions from multiple runs.")
        patch_preds = np.load(MEAN_PRED_PATH)
    else:
        print("Loading single-run predictions.")
        patch_preds = np.load(PATCH_PRED_PATH)
    garch_preds_rolling = np.load(GARCH_ROLL_PATH)
    kalman_preds_rolling = np.load(KALMAN_ROLL_PATH)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing prediction file: {e}. Please run training and `compare/targets/targets.py`.")

# Ensure predictions are 2D
if patch_preds.ndim == 1: patch_preds = patch_preds.reshape(-1, HORIZON)
if garch_preds_rolling.ndim == 1: garch_preds_rolling = garch_preds_rolling.reshape(-1, HORIZON)
if kalman_preds_rolling.ndim == 1: kalman_preds_rolling = kalman_preds_rolling.reshape(-1, HORIZON)

# Align predictions and actuals
test_start = pd.Timestamp("2024-01-01")
test_df = df.loc[df.index >= test_start].copy() # Use .copy() to avoid SettingWithCopyWarning

# Create ground truth windows that align with model predictions
actual_windows = sliding_window_view(test_df[TARGET_COL], window_shape=HORIZON)

# Create windows for full-window benchmarks
garch_full_windows = sliding_window_view(test_df[GARCH_FULL_COL], window_shape=HORIZON)
kalman_full_windows = sliding_window_view(test_df[KALMAN_FULL_COL], window_shape=HORIZON)

# Truncate all predictions to the shortest length to ensure alignment
min_len = min(len(patch_preds), len(garch_preds_rolling), len(kalman_preds_rolling), len(actual_windows) - SEQ_LEN)
patch_preds = patch_preds[:min_len]
garch_preds_rolling = garch_preds_rolling[:min_len]
kalman_preds_rolling = kalman_preds_rolling[:min_len]

# Align actuals and full-window benchmarks with the predictions (offset by SEQ_LEN)
actual_aligned = actual_windows[SEQ_LEN : SEQ_LEN + min_len]
garch_full_aligned = garch_full_windows[SEQ_LEN : SEQ_LEN + min_len]
kalman_full_aligned = kalman_full_windows[SEQ_LEN : SEQ_LEN + min_len]

# Get dates for plotting (corresponds to the last day of the forecast)
pred_dates = test_df.index[SEQ_LEN + HORIZON - 1 : SEQ_LEN + HORIZON - 1 + min_len]

# Metric calculations
def qlike(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.log(y_pred**2 + eps) + (y_true**2) / (y_pred**2 + eps))

def diracc(y_true, y_pred):
    return np.mean(np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true)))

# Prepare all models' predictions for metrics
model_preds = {
    'PatchTST': patch_preds,
    'GARCH (Full)': garch_full_aligned,
    'GARCH (Rolling)': garch_preds_rolling,
    'Kalman (Full)': kalman_full_aligned,
    'Kalman (Rolling)': kalman_preds_rolling
}

# Full Horizon Metrics
full_metrics = {}
for model, preds in model_preds.items():
    full_metrics[model] = {
        'MSE': mean_squared_error(actual_aligned, preds),
        'MAE': mean_absolute_error(actual_aligned, preds),
        'QLIKE': qlike(actual_aligned, preds),
        'DIRACC': diracc(actual_aligned, preds)
    }

# Last-Step predictions for plotting
actual_last_step = actual_aligned[:, -1]
patch_last_step = patch_preds[:, -1]
garch_rolling_last_step = garch_preds_rolling[:, -1]
kalman_rolling_last_step = kalman_preds_rolling[:, -1]
garch_full_last_step = garch_full_aligned[:, -1]
kalman_full_last_step = kalman_full_aligned[:, -1]

# Last-Step Metrics
last_metrics = {}
for model, preds in model_preds.items():
    preds_last_step = preds[:, -1]
    last_metrics[model] = {
        'MSE': mean_squared_error(actual_last_step, preds_last_step),
        'MAE': mean_absolute_error(actual_last_step, preds_last_step),
        'QLIKE': qlike(actual_last_step, preds_last_step),
        'DIRACC': diracc(actual_last_step, preds_last_step)
    }

# Print and save metrics 
full_df = pd.DataFrame(full_metrics).T[['MSE', 'MAE', 'QLIKE', 'DIRACC']]
last_df = pd.DataFrame(last_metrics).T[['MSE', 'MAE', 'QLIKE', 'DIRACC']]

print("\n=== Model Comparison Metrics (Full Horizon, All Steps) ===")
print(full_df.to_string(float_format=lambda x: f"{x:.6f}"))

# Save both metrics to a single CSV with section headers
with open(METRICS_OUT_PATH, 'w') as f:
    f.write('=== Model Comparison Metrics (Full Horizon, All Steps) ===\n')
    full_df.to_csv(f, float_format="%.6f")
    f.write('\n=== Model Comparison Metrics (Last Step Only) ===\n')
    last_df.to_csv(f, float_format="%.6f")

# PatchTST MSE CI from run_n_times
if os.path.exists(MSE_OUT_PATH):
    patchtst_mses = np.load(MSE_OUT_PATH)
    patchtst_mses = patchtst_mses[~np.isnan(patchtst_mses)]
    if len(patchtst_mses) > 1:
        import scipy.stats as stats
        mean_mse = np.mean(patchtst_mses)
        std_mse = np.std(patchtst_mses, ddof=1)
        ci = stats.t.interval(0.95, len(patchtst_mses)-1, loc=mean_mse, scale=std_mse/np.sqrt(len(patchtst_mses)))
        print(f"\nPatchTST MSE (all steps, {len(patchtst_mses)} runs): {mean_mse:.6f} (95% CI: {ci[0]:.6f}, {ci[1]:.6f})")
    elif len(patchtst_mses) == 1:
        print(f"\nPatchTST MSE (all steps, 1 run): {patchtst_mses[0]:.6f}")
    else:
        print("\nNo valid PatchTST MSEs found in patchtst_mses.npy.")
else:
    print("\nNo patchtst_mses.npy found for PatchTST MSE CI.")


# Plot forecast
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(20, 10))

# Determine slice for plotting
if PLOT_WINDOW is None:
    plot_slice = slice(None)
else:
    plot_slice = slice(-PLOT_WINDOW, None)

# Plot Actuals
plt.plot(pred_dates[plot_slice], actual_last_step[plot_slice], label="Actual Realized Vol", color="#005d9b", linewidth=2, zorder=10)
plt.plot(pred_dates[plot_slice], patch_last_step[plot_slice], linestyle='-', label="PatchTST Forecast", color="#ffae00", linewidth=1.5)
plt.plot(pred_dates[plot_slice], garch_full_last_step[plot_slice], linestyle=':', label="GARCH (Full Window)", color="#229247")
plt.plot(pred_dates[plot_slice], garch_rolling_last_step[plot_slice], linestyle='--', label="GARCH (Rolling)", color="#229247")
plt.plot(pred_dates[plot_slice], kalman_full_last_step[plot_slice], linestyle=':', label="Kalman (Full Window)", color="#9125c4")
plt.plot(pred_dates[plot_slice], kalman_rolling_last_step[plot_slice], linestyle='--', label="Kalman (Rolling)", color="#9125c4")

plt.title(f"Full test ({HORIZON}-Day Ahead) Volatility Forecast Comparison", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT)
print(f"Plot saved to {OUTPUT_PLOT}")
plt.show()