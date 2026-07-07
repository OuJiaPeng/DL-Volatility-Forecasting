# Side studies (not the core result)

These branch off the main question but aren't part of the headline vol-forecasting story
(that's in the [top-level README](../../README.md)). Kept for the record, moved out of the
active `scripts/` to keep the core surface clean. They still run from the repo root
(`python archive/codas/<name>.py ...`); paths were adjusted for this deeper location.

- **`intraday_dist.py`** — distributional follow-up to the intraday chapter: the champion's
  residual scale is also unpredictable (ridge & GBT ≈ 0), and a constant-σ Gaussian beats GBT-
  and NN-quantile heads on pinball and coverage, so there's nothing to model in the second
  moment either.
- **`build_hedge_episodes.py`, `hedge_study.py`** — deep-hedging coda: 917 real 1-day SPXW
  straddle episodes hedged with ES. A neural policy that nests the Whalley–Wilmott no-trade
  band matches it and no more (paired t < 1.2 at every cost level); it only kept up once it
  was parametrized to start from WW.
- **`hf_lob_pilot.py`** — HF microstructure pilot (ES `bbo-1s`+`ohlcv-1s`, 24 sessions):
  total RV is persistence-saturated (R² 0.92–0.96), L1 top-of-book adds ~nothing; a faint,
  fold-unstable jump-share signal is the only thread. This is the seed of the next project
  (impact-parameter forecasting), which moves to its own repo.
