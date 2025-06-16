import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# ========== CONFIGURATION ==========
CSV_PATH    = "./compare/targets/with_all_targets.csv"
TARGET_COL  = "target_std"
HORIZON     = 10
SEQ_LEN     = 60
PRED_PATH   = "./outputs/patch_preds.npy"
OUTPUT_PLOT = "./outputs/realized_vol_prediction_plot.png"
PLOT_WINDOW = 200

# ========== LOAD DATA ==========
df = pd.read_csv(CSV_PATH, parse_dates=['date'], index_col='date').sort_index()
if TARGET_COL not in df.columns or 'target_garch' not in df.columns:
    raise ValueError(f"Required columns not found in CSV: {list(df.columns)}")

# ========== LOAD PREDICTIONS ==========
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

# === PRINT SAMPLE VALUES ===
print("Date       | Actual   | Predicted | GARCH")
for d, a, p, g in zip(pred_dates[-10:], actual[-10:], preds_flat[-10:], garch[-10:]):
    print(f"{d.date()} | {a:.6f} | {p:.6f}  | {g:.6f}")

# === METRICS ===
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse_patch = mean_squared_error(actual, preds_flat)
mse_garch = mean_squared_error(actual, garch)

mae_patch = mean_absolute_error(actual, preds_flat)
mae_garch = mean_absolute_error(actual, garch)

def qlike(y_true, y_pred):
    eps = 1e-8
    return np.mean(np.log(y_pred**2 + eps) + (y_true**2) / (y_pred**2 + eps))

qlike_patch = qlike(actual, preds_flat)
qlike_garch = qlike(actual, garch)

dir_patch = np.mean(
    np.sign(preds_flat[1:] - preds_flat[:-1]) == np.sign(actual[1:] - actual[:-1])
)
dir_garch = np.mean(
    np.sign(garch[1:] - garch[:-1]) == np.sign(actual[1:] - actual[:-1])
)

metrics = pd.DataFrame({
    'PatchTST': [mse_patch, mae_patch, qlike_patch, dir_patch],
    'GARCH':    [mse_garch, mae_garch, qlike_garch, dir_garch]
}, index=['MSE', 'MAE', 'QLIKE', 'DirAcc'])

print("\n=== Full‐Test Metrics ===")
print(metrics.to_string(float_format=lambda x: f"{x:.6f}"))

# Save metrics to CSV for easy reading
metrics_out_path = './outputs/metrics_patch_vs_garch.csv'
metrics.to_csv(metrics_out_path)
print(f"\nMetrics saved to {metrics_out_path}\n")

# === PLOT ===
plt.figure(figsize=(14,5))
plt.plot(pred_dates[-PLOT_WINDOW:], actual[-PLOT_WINDOW:], label="Actual Realized Vol")
plt.plot(pred_dates[-PLOT_WINDOW:], preds_flat[-PLOT_WINDOW:], '--', label="PatchTST Prediction")
plt.plot(pred_dates[-PLOT_WINDOW:], garch[-PLOT_WINDOW:], ':', label="GARCH(1,1) Vol")
plt.title("Realized Volatility: Actual vs. PatchTST vs. GARCH(1,1)")
plt.xlabel("Date")
plt.ylabel("Volatility (decimal)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.show()