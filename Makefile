.PHONY: train ensemble targets plot clean

# Train a single PatchTST model
train:
	python PatchTST/engines/train_univar.py

# Run 30-model ensemble (averages predictions across runs)
ensemble:
	python PatchTST/engines/run_patchtst_n_times.py

# Generate targets and baselines (GARCH, Kalman, EWMA, rolling-std)
targets:
	python compare/targets/targets.py

# Generate comparison plot and metrics table
plot:
	python outputs/univar_outputs/plot_preds_univar.py

# Remove caches and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
