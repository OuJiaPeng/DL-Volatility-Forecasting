# Volatility Forecasting with Deep Learning

This project focuses on deep learning for stoch process modeling and vol forecating. My main goal is just exploring and implementing transformer models to do such tasks.

Currently, a PatchTST model is being used to perform vol forecasting for BTC.

Working on the multivariate model; The univariate model already performs well.

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
    ├── PatchTST/                             # PatchTST model, engine, and config
    │   ├── models/                           # Model architecture
    │   ├── engines/                          # Training and evaluation logic
    │   └── utils/                            # Config files for multivar/univar
    │       ├── __init__.py
    │       ├── default_multivar.yaml
    │       └── default_univar.yaml
    │
    ├── compare/targets/                      # Target generation scripts and data
    │   ├── ewma.csv
    │   ├── garch.csv
    │   ├── logvar.csv
    │   ├── rolling_std.csv
    │   ├── std.csv
    │   ├── targets.py
    │   └── with_all_targets.csv
    │
    ├── data/                                 # Input dataset and loader
    │   ├── btc_2014_now.csv
    │   └── data.py
    │
    ├── outputs/                              # Output predictions and plots
    │   ├── metrics_patch_vs_garch.csv
    │   ├── metrics_patch_vs_garch_kalman.csv
    │   ├── patch_preds_univar.npy
    │   ├── plot_preds.py
    │   ├── plot_preds_multivar.py
    │   ├── plot_preds_univar.py
    │   └── realized_vol_prediction_plot.png
    │
    ├── README.md                             # Project overview and documentation
    ├── requirements.txt                      # Dependency list
    └── .gitignore                            # Files/directories to ignore in Git

---

## Results

- Comparative analysis of PatchTST, GARCH, and Kalman Filter
- Visualizations: Realized volatility vs. predictions

| Metric    | PatchTST   | GARCH      | Kalman     |
|-----------|------------|------------|------------|
| MSE       | 0.000139   | 0.000191   | 0.000173   |
| MAE       | 0.009365   | 0.011153   | 0.010192   |
| QLIKE     | -6.27192   | -6.16218   | -6.14855   |
| DirAcc    | 0.464363   | 0.466523   | 0.339093   |

---

## Future Work

- Develop custom deep learning models 
- Explore other transformer architectures
- Expand to more financial time series datasets

---
