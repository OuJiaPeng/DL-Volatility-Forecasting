# Financial Volatility Forecasting with Deep Learning

This project explores deep learning for volatility forecasting. The main goal is to implement and evaluate transformer-based models for this task.

Currently, a PatchTST model is being used to perform volatility forecasting for BTC-USD daily prices. The project includes a full pipeline for data processing, training, evaluation, and comparison against classical benchmarks like GARCH and Kalman filters.

Working on the multivariate model; The univariate model already performs well.

---

## Results

The univariate PatchTST model, using an ensemble of 30 runs, demonstrates superior performance against both full-window and rolling GARCH and Kalman filter benchmarks. The metrics below are for the full test horizon.

For reference, the average realized volatility over the test set was 0.0243

| Metric    | PatchTST (Ensemble) | GARCH (Full) | GARCH (Rolling) | Kalman (Full) | Kalman (Rolling) |
|-----------|---------------------|--------------|-----------------|---------------|------------------|
| MSE       | 0.000106            | 0.000175     | 0.000196        | 0.000163      | 0.000232         |
| MAE       | 0.007567            | 0.010574     | 0.010852        | 0.009942      | 0.011947         |
| QLIKE     | -6.326726           | -6.219139    | -6.160537       | -6.194408     | -6.004205        |
| DirAcc    | 0.594681            | 0.495213     | 0.492021        | 0.335638      | 0.000000         |

**Note on Kalman (Rolling) DirAcc:** The 0% directional accuracy for the rolling Kalman filter is due to its simplistic forecast, which predicts a constant volatility over the entire horizon, thus never matching the direction of change.

**Note on Ensembling:** Due to the stochastic nature of DL training, we ensure the results are statistically signicant and robust, thus the PatchTST metrics are calculated from the mean prediction of 30 independent model runs. 

- PatchTST MSE (all steps, 30 runs): 0.000109 (95% CI: 0.000108, 0.000110)
![realized_vol_prediction_plot](https://github.com/user-attachments/assets/ece9eba2-ff37-48c6-be8f-6da406c79d03)

---

## Features
- **Transformer-based Volatility Forecasting:** An ensemble of PatchTST models for time series forecasting.
- **Classical Benchmark Models:** GARCH(1,1) and Kalman Filter, implemented with both full-window and rolling-window forecasting.
- **Comprehensive Evaluation:** MSE, MAE, QLIKE, and Directional Accuracy metrics.
- **Automated Pipeline:** Scripts for running multiple experiments, generating targets, and plotting results.
- **Visualizations:** Plots comparing model predictions against realized volatility.

---

## Project Structure

```
Volatility Forecasting with Deep Learning/
├── checkpoints/
│   ├── logs/
│   └── univar/
│
├── compare/
│   └── targets/                          # target generation scripts and targets
│       └── targets.py
│
├── data/                                 # data loader and data
│   └── data.py
│
├── outputs/                              # model predictions, metrics, and plots
│   └── univar_outputs/
│       ├── metrics_patch_vs_garch_kalman.csv
│       └── realized_vol_prediction_plot.png
│
├── PatchTST/                             # PatchTST model, engine, and config
│   ├── engines/
│   ├── models/
│   └── utils/
│       ├── config.py
│       └── default_univar.yaml
│
├── .gitignore
├── README.md
└── requirements.txt
```
---

## Data

- **Source:** Daily BTC-USD price data (2014–present), from EODHD API.
- **Preprocessing:**
  - Daily log returns are computed from adjusted close prices.
  - The target variable is the 5-day future rolling standard deviation of log returns (realized volatility).
  - Benchmarks (GARCH, Kalman, EWMA) are also generated and stored.
- **Partitioning:**
  - **Train:** 2014-01-01 to 2022-12-31
  - **Validation:** 2023-01-01 to 2023-12-31
  - **Test:** 2024-01-01 to present
- **Windowing:**
  - **Input window (seq_len):** 100 days
  - **Forecast horizon:** 5 days
- **Files:**
  - Raw data: `data/btc_2014_now.csv`
  - Target generation: `compare/targets/targets.py`
  - Processed targets & benchmarks: `compare/targets/with_all_targets.csv`

---

## Future Work

- Develop custom deep learning models from scratch.
- Explore other transformer architectures (e.g., TFT, Informer, Autoformer).
- Expand to multivariate forecasting, incorporating additional features.
- Test on more financial time series datasets.

---
