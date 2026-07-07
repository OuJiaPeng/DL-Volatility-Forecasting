# volforecast

Realized-volatility forecasting for SPX and NVDA options, set up to test one question: can a
neural net beat classical HAR-type models at forecasting RV? Across a range of data filtrations
and horizons the answer came out no, at least not with data you can buy off the shelf. Two useful
results came out of it anyway; they're near the bottom.

It's a rebuild of an older project that had look-ahead bugs (P.S. below). The rebuild separates
features and targets by a single decision timestamp, uses one alignment path for the whole
dataset, and measures the forecasting ceiling before any model is fit, so those bugs can't recur.

## The problem

Predict future realized volatility (RV) from what's known today: given everything up to the close,
forecast RV over the next 1 / 5 / 10 / 21 sessions (and later the next 5 / 15 / 30 minutes). Scoring
is QLIKE, which stays well-behaved when the target is a noisy vol proxy. The point was to see
whether a transformer or other neural net could beat HAR and its variants.

## The data

- **Universe:** SPX and NVDA options plus the underlying. Index first, then a single volatile name
  to check whether "crowded" markets were the reason things kept failing. They weren't.
- **Vendor:** Databento — raw options quotes (`SPX.OPT`/`SPXW.OPT`, `NVDA.OPT`) plus ES/NVDA minute
  bars. No vendor ships trustworthy historical implied vol, so we solve it from raw quotes:
  parity-implied forward, Black-76 inversion, then tenor-matched ATM IV, skew, and term slope.
- **Coverage:** ~1,400 SPX sessions and ~1,150 NVDA sessions, 2021–2026. Every pull is cached to
  parquet, so the API spend was one-time and small (it ran on a free key).

## The short version of what we found

- Volatility is very persistent. Today's RV is most of tomorrow's, so a baseline that ignores this
  is too weak to be a fair comparison, and models that look good often just rediscover it.
- Implied vol is already a forecast — the market's. Once IV is in the feature set, the only thing
  left to predict is its bias (the variance risk premium), and that bias is close to linear in logs.
  About six ridge coefficients capture it.
- So the ceiling is low. We checked directly by regressing the best model's out-of-sample errors on
  every available feature, with both linear models and trees. Out-of-sample R² came back ≤ 0 almost
  everywhere, which says the information is used up rather than the model being too small.

## Classical baselines

The bar was set with real baselines, not toy ones:

- **HAR-RV** (Corsi): daily / weekly / monthly trailing RV.
- **HAR-IV / HAR-X:** HAR plus the implied-vol term structure.
- **hariv_x:** HAR-IV plus a state-dependent premium term (IV × VIX). This was the best model on
  SPX, five or six coefficients, and nothing beat it.
- Also run: persistence/EWMA, GARCH(1,1), and naive-IV (use the market's IV as the forecast).
  naive-IV is the check that matters most: a model that can't beat it isn't adding anything.

## We beat it to death across filtrations and horizons

| What we tried | Result |
|---|---|
| Daily RV from RV history | HAR wins |
| Daily RV from RV + IV | HAR-IV / hariv_x wins; the IV bias is linear |
| Daily RV from IV only (the premium problem) | bias-corrected IV is the best rung; curves and trees hurt |
| Forecasting IV itself | long tenors ≈ martingale (persistence is the ceiling); short tenor has a small, linear event-cycle wrinkle |
| 0DTE terminal 15 min (1,133 sessions) | the closing straddle is efficient; gate closed |
| Intraday next-30-min RV (16k origins) | see below |
| NVDA, every one of the above | same conclusions, just noisier |

Each one ran the same protocol: measure the ceiling with the oracle, fit a strong baseline, add
only the features the diagnostics justify, re-check the oracle, stop when nothing's left.

## Why ML didn't help

Seven neural architectures on the full SPX panel (iTransformer, LSTM, TCN, a surface-attention
model, linear controls, and others) all lost to the six-coefficient ridge, and not by a small
margin. The reason was the same each time. Forecast error splits into approximation, estimation,
information gap, and irreducible noise. A neural net only helps with the first, and that part was
already small here because the structure is simple once you know what it is. What actually bound us
was the information gap (the market had already priced it) and estimation error (~1,400 days, heavily
autocorrelated, is only a few hundred effective samples — too few to justify a big model). No
architecture changes either of those.

The pattern held across the whole project: wherever there was predictable structure — persistence,
the risk premium, state dependence, the intraday event clock — a handful of linear coefficients
captured it, and a larger model approximating the same thing did worse.

## Two things that did work

1. **An intraday event-clock model.** Forecasting next-30-min RV, the oracle gate opened for once:
   trees found structure the linear model had missed. It localized to specific times — the 2pm FOMC
   block runs about 3× normal vol, plus the open and the close. Writing those as explicit interaction
   terms (clock × IV-premium), about 50 of them, beat the trees in every fold, roughly 7% better than
   diurnal HAR+IV, and closed the gate again. It's a better classical model rather than a neural win,
   but it came straight out of following the protocol.
2. **A dated empirical result.** The very-short-dated 0DTE straddle premium that was there in
   2022–2024 has compressed to about zero (slightly negative) in 2025–2026, around the time
   systematic 0DTE selling got crowded. I haven't seen this period documented elsewhere.

## What's in here

```
volforecast/     the leakage-safe package
  timeutil.py      point-in-time guards (features <= t0 < targets)
  panel.py         the single alignment path (feat_* | tgt_* | meta_*, indexed by t0)
  splits.py        walk-forward folds with purge/embargo
  data/            Databento adapter, volsolve.py (IV from raw quotes), synthetic vendor
  features/        HAR, realized RV, IV surface, intraday structure, forward-only targets
  models/          HAR-prior + residual-trunk hybrid, and the trunks that didn't beat it
  eval/            QLIKE, pinball, HAC Diebold–Mariano, walk-forward runner
scripts/         core chapters (oracle, iv_only, study_0dte, intraday_oracle, data pulls)
experiments/     protocol notes + ledger.csv (every run)
tests/           no-lookahead suite and solver round-trips
configs/         default.yaml (synthetic) | spx.yaml | nvda.yaml
archive/codas/   side-studies (distributional, deep-hedging, HF pilot)
archive/legacy/  the original project (see the P.S.)
```

## Reproduce

```bash
pip install -e ".[dev]"
make test        # full suite incl. the no-lookahead invariants; runs on synthetic data, no API key
make all         # build panel -> baselines -> comparison table (synthetic)
```

Real data needs `pip install -e ".[databento]"` and `DATABENTO_API_KEY` (a `.env` works). Pulls are
cached; price any pull first with `metadata.get_cost`. `python scripts/status.py` shows cache and
ledger state.

## Next

The remaining thread is microstructure. Total RV is saturated by persistence, but there's a weak,
unstable signal in the jump component from order flow. The plan is to forecast a market-impact or
liquidity parameter (Kyle's λ, or a GLFT market-making parameter) instead of volatility, since
there's no quoted "impact surface" the way there is an IV surface. That's a separate project in its
own repo.

## P.S.

This began as a BTC realized-vol forecaster (PatchTST plus a set of baselines). An audit found two
look-ahead bugs: a forward-looking target that was also fed in as an input, and two alignment code
paths that had drifted 60 rows apart. Both make results look better than they are. Rather than patch
them, the project was rebuilt so that kind of bug can't recur. The old code is in
[`archive/legacy/`](archive/legacy/).

---

MIT.
