# volforecast

Realized-volatility forecasting for SPX and NVDA options.

The question is direct: can a neural model beat strong classical volatility forecasts? On the daily horizons
tested here, the answer was no. HAR-style models, implied-vol features, and a few linear premium terms explain
most of what the data can support.

That negative result is useful. It says the bottleneck is information, not model size.

## The Problem

At a decision time `t0`, use only information known by then to forecast future realized volatility:

- next 1 / 5 / 10 / 21 trading sessions,
- plus intraday variants such as next-30-minute RV.

The main score is QLIKE, which behaves better than squared error when the target is a noisy volatility proxy.

## Data

- **Universe:** SPX options, NVDA options, and the underlying/minute bars.
- **Vendor:** Databento raw quotes and bars.
- **Coverage:** roughly 1,400 SPX sessions and 1,150 NVDA sessions from 2021-2026.
- **IV construction:** implied vol is solved from raw quotes using parity-implied forwards and Black-76
  inversion, then summarized into ATM IV, skew, and term-structure features.

The package is built around one point-in-time panel: features are known at `t0`, targets start after `t0`, and
walk-forward folds use purge/embargo rules.

## What We Found

- Volatility is persistent. A weak baseline makes ML look better than it is.
- Implied volatility is already a forecast. Once IV is in the feature set, the remaining bias is small and
  close to linear.
- On daily horizons, six-ish ridge coefficients often beat larger neural models.
- Error-oracle checks usually came back with out-of-sample `R^2 <= 0`, meaning the remaining error did not
  have clean predictable structure.

## Baselines

The main ladder:

- **Persistence / EWMA**
- **HAR-RV**
- **HAR-IV / HAR-X**
- **hariv_x:** HAR-IV plus a state-dependent premium term
- **GARCH(1,1)**
- **Naive IV:** use the market's implied vol as the forecast

The important baseline is naive IV. If a model cannot beat the market's own volatility forecast after proper
alignment, it is not adding much.

## Model Attempts

The repo includes neural trunks and hybrid models:

- residual transformer,
- LSTM/TCN-style trunks,
- HAR-prior hybrids,
- surface/gating experiments.

On the daily SPX/NVDA panels, these did not beat the simple HAR/IV ridge-style baselines. The data set is not
large in effective sample size, and the structure that remains after IV is included is simple.

## Two Useful Results

### Intraday Event Clock

The next-30-minute RV problem had real structure. It concentrated around specific clock blocks: the open, the
close, and event windows such as 2pm FOMC. Writing those as explicit clock-by-premium interactions beat the
tree oracle and improved on diurnal HAR+IV by about 7%.

That is still a classical-model win, but it came from the same protocol: check the oracle, localize the signal,
write the feature directly.

### 0DTE Premium Compression

The very-short-dated 0DTE straddle premium that appeared in 2022-2024 compressed toward zero in 2025-2026.
That matches the idea that systematic 0DTE selling became crowded.

## Repo Map

```text
volforecast/
  timeutil.py      point-in-time guards
  panel.py         feature/target alignment
  splits.py        walk-forward folds with purge/embargo
  data/            Databento adapter and IV solver
  features/        HAR, RV, IV surface, intraday features
  models/          neural trunks and HAR-prior hybrids
  eval/            QLIKE, pinball, DM tests, walk-forward runner
scripts/           experiment entrypoints
experiments/       run ledger and notes
tests/             no-lookahead and solver tests
configs/           synthetic, SPX, NVDA configs
```

## Reproduce

```bash
pip install -e ".[dev]"
make test        # runs on synthetic data, no API key needed
make all         # synthetic panel -> baselines -> comparison table
```

Real data needs:

```bash
pip install -e ".[databento]"
```

and `DATABENTO_API_KEY`. Pulls are cached. Price any pull first with `metadata.get_cost`.
`python scripts/status.py` shows cache and ledger state.

## Next

The remaining interesting signal is microstructure. Total realized volatility is mostly saturated by
persistence and IV, but the jump/liquidity component may still contain weak order-flow information. That
belongs in a separate market-impact or market-making project rather than another daily-RV model.

MIT.
