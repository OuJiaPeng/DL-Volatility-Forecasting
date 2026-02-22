"""30-run PatchTST ensemble driver.

Runs train_univar.py N_RUNS times, collects per-run MSEs,
computes mean prediction and 95 % confidence interval, and
saves ensemble outputs to outputs/univar_outputs/patchtst_preds/.

Usage:
    python PatchTST/engines/run_patchtst_n_times.py
"""
import numpy as np
from scipy import stats
import subprocess
import sys
import re
import os

N_RUNS = 30
MSE_OUT_PATH = "./outputs/univar_outputs/patchtst_preds/patchtst_mses.npy"
ALL_PRED_PATH = "./outputs/univar_outputs/patchtst_preds/patch_preds_univar_all.npy"
MEAN_PRED_PATH = "./outputs/univar_outputs/patchtst_preds/patch_preds_univar_mean.npy"
PRED_PATH = "./outputs/univar_outputs/patchtst_preds/patch_preds_univar.npy"

def run_experiment():
    # Run train_univar.py as a subprocess and capture its output
    result = subprocess.run(
        [sys.executable, 'PatchTST/engines/train_univar.py'],
        capture_output=True, text=True
    )
    # Look for the 'Real-vol MSE (all steps):' line in stdout
    match = re.search(r'Real-vol MSE \(all steps\): ([0-9.eE+-]+)', result.stdout)
    if match:
        mse = float(match.group(1))
        return mse
    else:
        print("Output from failed run:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError('Could not parse MSE from train_univar.py output')

mses = []
all_preds = []
for i in range(N_RUNS):
    try:
        mse = run_experiment()
        mses.append(mse)
        print(f"Run {i+1}: MSE = {mse:.8f}")
        # After each run, load predictions and append
        preds = np.load(PRED_PATH)
        all_preds.append(preds)
    except Exception as e:
        print(f"Run {i+1} failed: {e}")
        mses.append(np.nan)
        all_preds.append(np.full_like(np.load(PRED_PATH), np.nan))  # fill with nans if failed

mses = np.array(mses)
all_preds = np.stack(all_preds)
mean_preds = np.nanmean(all_preds, axis=0)

mean_mse = np.nanmean(mses)
std_mse = np.nanstd(mses, ddof=1)
valid_runs = np.sum(~np.isnan(mses))
if valid_runs > 1:
    conf_int = stats.t.interval(0.95, valid_runs-1, loc=mean_mse, scale=std_mse/np.sqrt(valid_runs))
else:
    conf_int = (np.nan, np.nan)

print(f"\nMean MSE: {mean_mse:.8f}")
print(f"95% CI: ({conf_int[0]:.8f}, {conf_int[1]:.8f})")
print(f"Valid runs: {valid_runs}/{N_RUNS}")

# Save all MSEs for later analysis
os.makedirs(os.path.dirname(MSE_OUT_PATH), exist_ok=True)
np.save(MSE_OUT_PATH, mses)
print(f"All MSEs saved to {MSE_OUT_PATH}")

# Save all predictions and mean prediction for CI plotting
np.save(ALL_PRED_PATH, all_preds)
np.save(MEAN_PRED_PATH, mean_preds)
print(f"All predictions saved to {ALL_PRED_PATH}")
print(f"Mean prediction saved to {MEAN_PRED_PATH}")
