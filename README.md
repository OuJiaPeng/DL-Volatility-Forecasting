# Volatility Forecasting with PatchTST

> 30-model PatchTST ensemble achieving **35 % MSE reduction** over GARCH / Kalman baselines on BTC 5-day forward realised volatility (directional accuracy 0.60 vs 0.49).

---

## Results

Average realised volatility over the test set: **0.0243**.

| Model | MSE | MAE | QLIKE | Dir. Accuracy |
|-------|----:|----:|------:|--------------:|
| **PatchTST (30-run ensemble)** | **0.000106** | **0.007567** | **−6.327** | **0.595** |
| GARCH (Full) | 0.000175 | 0.010574 | −6.219 | 0.495 |
| GARCH (Rolling) | 0.000196 | 0.010852 | −6.161 | 0.492 |
| Kalman (Full) | 0.000163 | 0.009942 | −6.194 | 0.336 |
| Kalman (Rolling) | 0.000232 | 0.011947 | −6.004 | 0.000 |

- PatchTST MSE (all steps, 30 runs): 0.000109 — 95 % CI: (0.000108, 0.000110)
- Kalman (Rolling) directional accuracy is 0 % because it predicts a constant volatility over the entire horizon.

![Realised volatility prediction plot](https://github.com/user-attachments/assets/ece9eba2-ff37-48c6-be8f-6da406c79d03)

---

## Methodology

### Model

| Component | Detail |
|-----------|--------|
| Architecture | PatchTST — channel-independent transformer with patched input embedding |
| Input | 60-day lookback of 5-day rolling realised volatility (std of log returns) |
| Output | 5-day ahead realised volatility forecast |
| Patch size | 12 (5 patches per sequence) |
| Encoder | 3-layer transformer, 4 heads, 64-dim embeddings, GELU activation |
| Training | AdamW + OneCycleLR, MSE loss, early stopping (patience 20) |
| Ensembling | 30 independent runs; predictions averaged to reduce variance |

### Baselines

| Baseline | Description |
|----------|-------------|
| **GARCH(1,1) — Full** | Fitted on all data up to each prediction point |
| **GARCH(1,1) — Rolling** | 100-day rolling window, h-step forecast |
| **Kalman Filter — Full** | Random-walk state-space model on squared returns |
| **Kalman Filter — Rolling** | 100-day rolling window Kalman |

---

## Data

- **Asset**: BTC-USD daily (2014 – present), from EODHD API
- **Target**: 5-day forward rolling standard deviation of log returns (realised volatility)
- **Train**: 2014-01-01 → 2022-12-31
- **Validation**: 2023-01-01 → 2023-12-31
- **Test (OOS)**: 2024-01-01 → 2025 H1

---

## Repository Structure

```
├── data/                       # Raw data & fetcher
│   ├── data.py                 #   EODHD API download script
│   └── btc_2014_now.csv        #   Daily BTC prices (gitignored)
├── compare/
│   └── targets/                # Benchmark target generation
│       └── targets.py          #   GARCH, Kalman, EWMA, rolling-std targets
├── PatchTST/                   # Model implementation
│   ├── models/
│   │   └── patchtst_univar.py  #   Univariate PatchTST architecture
│   ├── engines/
│   │   ├── train_univar.py     #   Single-run training & evaluation
│   │   └── run_patchtst_n_times.py  # 30-run ensemble driver
│   └── utils/
│       ├── config.py           #   YAML config loader
│       └── default_univar.yaml #   Hyperparameters
├── outputs/
│   └── univar_outputs/         # Metrics CSV, prediction plot
│       └── plot_preds_univar.py  # Plotting & metrics script
├── Makefile
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/OuJiaPeng/Vol-Forecasting-with-DL.git
cd Vol-Forecasting-with-DL
pip install -r requirements.txt

# 2. Generate targets & baselines (requires data/btc_2014_now.csv)
make targets

# 3. Train a single PatchTST model
make train

# 4. Run 30-model ensemble (takes ~30 min on GPU)
make ensemble

# 5. Generate comparison plot & metrics
make plot
```

### Configuration

All model hyperparameters are in [`PatchTST/utils/default_univar.yaml`](PatchTST/utils/default_univar.yaml).

---

## License

MIT
- Expand to multivariate forecasting, incorporating additional features.
- Test on more financial time series datasets.

---
