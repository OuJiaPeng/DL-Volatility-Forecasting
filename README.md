# 📈 Volatility Forecasting with Deep Learning

This project continues my research at HKUST, focusing on deep learning for stochastic process modeling and volatility forecasting. The main goal is to explore and implement advanced transformer architectures—such as Temporal Fusion Transformer (TFT) and PatchTST—for multivariate time series analysis.

---

## ✨ Features
- **Transformer-based Volatility Forecasting:** PatchTST and TFT models for time series
- **Custom Deep Learning Algorithms:** Ongoing work towards building models from scratch
- **Comprehensive Evaluation:** MSE, MAE, QLIKE, Directional Accuracy, and more
- **Visualizations:** Prediction plots and metrics for model comparison

---

## 🗂 Project Structure

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

## 🛠️ Setup

Clone this repository:
```bash
git clone <your-repo-url>
cd Vol-Forecasting-with-DL
```

Install the required packages:
```bash
pip install -r requirements.txt 
```

---

## 🚀 How to use

- Prepare your data in `data/` (see `data.py` for details)
- Train the model:
  ```bash
  python PatchTST/engines/train.py --config PatchTST/default.yaml
  ```
- Generate and plot predictions:
  ```bash
  python outputs/plot_preds.py
  ```

---

## 📊 Results

- Comparative analysis of PatchTST, GARCH, and other baselines
- Metrics: MSE, MAE, QLIKE, Directional Accuracy
- Visualizations: Realized volatility vs. predictions

---

## 📝 Future Work

- Develop custom deep learning models for stochastic processes
- Explore additional transformer architectures
- Expand to more financial time series datasets

---

## 🔗 Reference

- Related work: https://github.com/OuJiaPeng/Vol-Forecasting-with-DL

**Skills & Tools:** PyTorch · Deep Learning · Machine Learning · Mathematics · Python