# Volatility Forecasting with Deep Learning

This project focuses on deep learning for stoch process modeling and vol forecating. My main goal is just exploring and implementing transformer models to do such tasks.

Currently, a PatchTST model is being used to perform vol forecasting for BTC.

---

## Features
- **Transformer-based Volatility Forecasting:** PatchTST for time series
- **Custom Deep Learning Algorithms:** Ongoing work towards building models from scratch
- **Comparitive Models:** GARCH(1,1), Kalman Filter
- **Comprehensive Evaluation:** MSE, MAE, QLIKE, Directional Accuracy 
- **Visualizations:** Prediction plots and metrics for model comparison

---

## Project Structure

    Vol-Forecasting-with-DL/
    ├── PatchTST/                      # Model code, configs, and training scripts
    │   ├── models/                    # PatchTST model definition
    │   ├── engines/                   # Training logic
    │   └── utils/                     # Config and helpers
    ├── compare/                       # Target generation and comparison
    ├── data/                          # Data loading and preprocessing
    ├── outputs/                       # Predictions, plots, and metrics
    │   ├── plot_preds.py              # Plotting and evaluation script
    │   ├── realized_vol_prediction_plot.png
    │   ├── metrics_patch_vs_garch.csv
    │   └── patch_preds.npy            # (ignored by git)
    ├── README.md                      # Project overview
    └── .gitignore                     # Git ignore file
    
---

## Results

- Comparative analysis of PatchTST, GARCH, and Kalman Filter
- Metrics: MSE, MAE, QLIKE, Directional Accuracy
- Visualizations: Realized volatility vs. predictions

---

## Future Work

- Develop custom deep learning models 
- Explore other transformer architectures
- Expand to more financial time series datasets

---
