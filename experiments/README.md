# Lab notebook

How every result in this project was tested, and the results themselves in one place.
The main story is in the top-level [README](../README.md); this is the technical record.

## The protocol (same for every arena)

1. **Ceiling first.** Before any model, run the oracle (`scripts/oracle_test.py` and the
   per-arena equivalents): regress the current champion's out-of-fold errors on *all*
   features, ridge **and** gradient-boosted trees, out-of-sample. If best R² ≤ 0, the
   information set is exhausted — stop adding model capacity; the only licensed moves are
   new information or pooling.
2. **Strong baseline, never a strawman.** HAR / HAR-IV / hariv_x and naive-IV are the bar.
   A model that can't beat naive-IV out-of-sample is reported as adding nothing.
3. **Add only what the diagnostics license, then re-check the oracle.** Baseline → oracle →
   add-what-loads → re-oracle, until dry.
4. **Honest evaluation.** Walk-forward folds with purge/embargo; standardize on train only;
   selection on validation; test read once, at the end. QLIKE primary; Diebold–Mariano with
   HAC lag for significance. Every run appended to `ledger.csv`.

Amendment learned the hard way: run the oracle as a **gate between rungs, not once at the
end** — the daily SPX campaign burned ~10 full-panel neural arms a post-hariv_x oracle would
have predicted dead.

## Results by arena

| Arena | n | Champion | ML verdict | Oracle gate |
|---|---|---|---|---|
| SPX daily RV (h=1/5/10/21) | 1,412 | hariv_x (HAR-IV + IV×VIX, 6 coef) | 7 nets all lose | closed (resid R² ≤ 0) |
| NVDA daily RV | 1,152 | HAR-IV | lose | closed |
| IV-only filtration → RV | " | MZ-bias-corrected IV | curves/trees hurt | closed |
| IV as the target | " | persistence (long tenor); linear IV_5→IV_1 wrinkle (short) | trees worse | — |
| 0DTE terminal 15 min | 1,133 | bias-corrected straddle | — | closed |
| Intraday next-30-min RV (SPX) | 16,008 | **event-clock HAR** (see below) | GBT opened gate, then linear re-closed it | opened → named → closed |
| Intraday next-30-min RV (NVDA) | 13,056 | diurnal-HAR+IV | lose | closed |

**Intraday, the one gate that opened.** Ladder (log-MSE): unconditional 0.337 → time-of-day
0.318 → ES-lite 0.073 → diurnal-HAR+IV 0.0687. GBT then found residual signal in all folds
(+0.012, +0.027 at 8 folds) that ridge missed. Localized to 14:00 FOMC blocks (~2.9× normal
vol), the open, and the close; named as clock × IV-premium / FOMC × clock interactions and
encoded as ~50 explicit linear terms → MSE **0.0640**, beating GBT in every fold, gate
re-closed against ridge, GBT, and within-window shape features. A better *classical* model,
found by the discipline. (Caveat for any headline: interaction terms were designed while
seeing all folds; confirm on a fresh fold before publishing the exact +7%.)

**Thesis:** every predictable piece of vol has a name — persistence, premium,
state-dependence, event clock — and once named, a few linear coefficients beat any capacity
model approximating it.

## Codas (not central — moved to `archive/codas/`)

Kept for the record, out of the active surface. See [`archive/codas/`](../archive/codas/).

- **Distributional**: the champion's residual *scale* is also unpredictable (ridge & GBT ≈ 0);
  a constant-σ Gaussian beats GBT- and NN-quantile heads on pinball and coverage. Nothing left
  in mean or scale.
- **Deep hedging**: 917 real 1-day SPXW straddle episodes, hedged with ES. A neural policy that
  nests the Whalley–Wilmott no-trade band matches it and no more (paired t < 1.2 at every cost);
  the landscape near the classical optimum is flat. Lesson: parametrization beats capacity.
- **HF microstructure pilot**: 24 ES sessions, `bbo-1s`+`ohlcv-1s`. Total RV is
  persistence-saturated (R² 0.92–0.96), L1 book adds ~nothing; a faint, unstable jump-share
  signal is the only thread. Seed of the next (impact-forecasting) project, in its own repo.

## Ledger

`ledger.csv` — one row per (experiment, arm): timestamp, exp id, arm, trunk, overrides,
folds, val QLIKE, test QLIKE (blank unless milestone), DM stat/p, seeds, git SHA, note,
decision. Appended by `scripts/run_experiment.py`. A row without a decision is an open
question.

```bash
python scripts/run_experiment.py --config configs/spx.yaml --exp E0 \
    --arms har_rv,hariv_x,persistence,garch,naive_iv --note "re-baseline"
```
