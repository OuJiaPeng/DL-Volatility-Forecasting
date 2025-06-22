"""
Plot and evaluate univariate PatchTST predictions vs. GARCH and Kalman targets.
Saves metrics and plot to outputs/univar_outputs/.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ========== CONFIGURATION ==========
CSV_PATH    = "./compare/targets/with_all_targets.csv"
TARGET_COL  = "target_std"
HORIZON     = 5
SEQ_LEN     = 60
PRED_PATH   = "./outputs/univar_outputs/patch_preds_univar.npy"
OUTPUT_PLOT = "./outputs/univar_outputs/realized_vol_prediction_plot.png"
PLOT_WINDOW = 200
METRICS_OUT_PATH = './outputs/univar_outputs/metrics_patch_vs_garch_kalman.csv'

# ========== LOAD DATA ==========
df = pd.read_csv(CSV_PATH, parse_dates=['date'], index_col='date').sort_index()
if TARGET_COL not in df.columns:
    raise ValueError(f"Missing {TARGET_COL}")
if 'target_garch' not in df.columns or 'target_kalman' not in df.columns:
    raise ValueError("Missing GARCH or Kalman targets in CSV")

# ========== LOAD PREDICTIONS ==========
if not os.path.exists(PRED_PATH):
    raise FileNotFoundError(f"Prediction file not found: {PRED_PATH}")
preds = np.load(PRED_PATH)   # already in raw-vol units
if preds.ndim == 1:
    preds = preds.reshape(-1, HORIZON)
preds_flat = preds[:, -1]    # last-step forecast

# === DEFINE TEST PERIOD ===
test_start = pd.Timestamp("2024-01-01")
test_dates = df.index[df.index >= test_start]
# Each prediction window starts at one point in test_dates and predicts horizon ahead:
pred_dates = test_dates[SEQ_LEN : SEQ_LEN + len(preds_flat)]

# === ALIGN SERIES ===
actual = df.loc[pred_dates, TARGET_COL].values
garch  = df.loc[pred_dates, 'target_garch'].values
kalman = df.loc[pred_dates, 'target_kalman'].values if 'target_kalman' in df.columns else None

# === PRINT SAMPLE VALUES ===
print("Date       | Actual   | Predicted | GARCH   | Kalman")
for i, (d, a, p, g) in enumerate(zip(pred_dates[-10:], actual[-10:], preds_flat[-10:], garch[-10:])):
    k = kalman[-10:][i] if kalman is not None else np.nan
    print(f"{d.date()} | {a:.6f} | {p:.6f}  | {g:.6f} | {k:.6f}")

# === METRICS ===
def qlike(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.log(y_pred**2 + eps) + (y_true**2) / (y_pred**2 + eps))

mse_patch = mean_squared_error(actual, preds_flat)
mse_garch = mean_squared_error(actual, garch)
mse_kalman = mean_squared_error(actual, kalman) if kalman is not None else np.nan
mae_patch = mean_absolute_error(actual, preds_flat)
mae_garch = mean_absolute_error(actual, garch)
mae_kalman = mean_absolute_error(actual, kalman) if kalman is not None else np.nan
qlike_patch = qlike(actual, preds_flat)
qlike_garch = qlike(actual, garch)
qlike_kalman = qlike(actual, kalman) if kalman is not None else np.nan
dir_patch = np.mean(
    np.sign(preds_flat[1:] - preds_flat[:-1]) == np.sign(actual[1:] - actual[:-1])
)
dir_garch = np.mean(
    np.sign(garch[1:] - garch[:-1]) == np.sign(actual[1:] - actual[:-1])
)
dir_kalman = np.mean(
    np.sign(kalman[1:] - kalman[:-1]) == np.sign(actual[1:] - actual[:-1])
) if kalman is not None else np.nan

metrics = pd.DataFrame({
    'PatchTST': [mse_patch, mae_patch, qlike_patch, dir_patch],
    'GARCH':    [mse_garch, mae_garch, qlike_garch, dir_garch],
    'Kalman':   [mse_kalman, mae_kalman, qlike_kalman, dir_kalman]
}, index=['MSE', 'MAE', 'QLIKE', 'DirAcc'])

print("\n=== Full‐Test Metrics ===")
print(metrics.to_string(float_format=lambda x: f"{x:.6f}"))

# Save metrics to CSV for easy reading
os.makedirs(os.path.dirname(METRICS_OUT_PATH), exist_ok=True)
metrics.to_csv(METRICS_OUT_PATH)
print(f"\nMetrics saved to {METRICS_OUT_PATH}\n")

# === PLOT ===
plt.figure(figsize=(14,5))
plt.plot(pred_dates[-PLOT_WINDOW:], actual[-PLOT_WINDOW:], label="Actual Realized Vol")
plt.plot(pred_dates[-PLOT_WINDOW:], preds_flat[-PLOT_WINDOW:], '--', label="PatchTST Prediction")
plt.plot(pred_dates[-PLOT_WINDOW:], garch[-PLOT_WINDOW:], ':', label="GARCH(1,1) Vol")
if kalman is not None:
    plt.plot(pred_dates[-PLOT_WINDOW:], kalman[-PLOT_WINDOW:], '-.', label="Kalman Filter Vol")
plt.title("Realized Volatility: Actual vs. PatchTST vs. GARCH(1,1) vs. Kalman")
plt.xlabel("Date")
plt.ylabel("Volatility (decimal)")
plt.legend()
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
plt.savefig(OUTPUT_PLOT)
plt.show()