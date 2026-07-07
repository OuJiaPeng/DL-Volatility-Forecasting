# volforecast — can ML beat classical models at forecasting volatility?

Short answer, after beating on it for a while: **no, not on data you can buy off the shelf.**
The long answer is the interesting part, and it's honest — every negative result here is a
real test, not a strawman. Two genuinely new things did fall out along the way (below).

This is a leakage-safe rebuild of an older project. The rules were simple: build the data so
look-ahead bugs are *impossible*, always establish the forecasting ceiling *before* touching a
model, and never let a fancy model win by beating a weak baseline.

## The problem

Forecast future realized volatility (RV) of an asset from what you know today. Concretely: given
everything up to the close, predict RV over the next 1 / 5 / 10 / 21 sessions (and later, the next
5 / 15 / 30 minutes). Score with **QLIKE**, the metric that's actually robust when your target is a
noisy vol proxy. The whole point was to see whether a transformer / neural net could beat the
classical econometric models (HAR and friends) that have owned this problem for 20 years.

## The data

- **Universe:** SPX and NVDA options + underlying. (Picked index first, then a single volatile
  name to check whether "crowded" markets were the reason things kept failing. They weren't.)
- **Vendor:** Databento — raw options quotes (`SPX.OPT`/`SPXW.OPT`, `NVDA.OPT`) + ES/NVDA minute
  bars. No vendor ships trustworthy *historical* implied vol, so we solve it ourselves from raw
  quotes: parity-implied forward → Black-76 inversion → tenor-matched ATM IV, skew, term slope.
- **Coverage:** ~1,400 SPX sessions and ~1,150 NVDA sessions, 2021–2026. Every remote pull is
  cached to parquet, so the API spend was one-time (and small — the whole thing ran on a free key).

## Data structure & the baby-analysis version of what we found

Strip away the machinery and here's the picture:

- **Volatility is extremely persistent.** Today's RV is most of tomorrow's RV. Any baseline that
  ignores this is a strawman; any model that "wins" mostly just rediscovers it.
- **Implied vol is a forecast — the market's forecast.** The option surface is the capital-weighted
  view of every professional vol desk, using everything public plus their own flow. The moment you
  put IV in your feature set, your only remaining edge is correcting its *bias* — the variance risk
  premium — and that bias turned out to be **linear in logs**. About six ridge coefficients capture
  it. There's nothing left for a big model to learn.
- **So the ceiling is low and the market sets it.** We measured this directly with an "oracle": regress
  the champion's out-of-sample errors on *every* feature we have, linear and trees. Out-of-sample R²
  came back **≤ 0** almost everywhere. That's the tell that the information is exhausted — not that the
  model is too small.

## Classical baselines (the bar to clear)

Not toy baselines — the real ones:

- **HAR-RV** (Corsi): daily/weekly/monthly trailing RV. The 20-year champion.
- **HAR-IV / HAR-X:** HAR plus the implied-vol term structure. This is the honest bar.
- **hariv_x:** HAR-IV plus a state-dependent premium (IV × VIX interaction). **This was the SPX
  champion** — 5–6 coefficients, and nothing beat it.
- Also ran: persistence/EWMA, GARCH(1,1), and **naive-IV** ("just quote the market's number"),
  which is the single most important sanity check — if your model can't beat naive-IV, it's adding
  nothing.

## We beat it to death — every filtration × horizon we could think of

| What we tried | Result |
|---|---|
| Daily RV from RV history | HAR wins |
| Daily RV from RV + IV | HAR-IV / hariv_x wins; the IV bias is linear |
| Daily RV from **IV only** (the premium problem) | bias-corrected IV is the best rung; curves & trees *hurt* |
| Forecasting **IV itself** | long tenors ≈ martingale (persistence is the ceiling); short tenor has a small, *linear* event-cycle wrinkle |
| **0DTE** terminal 15 min (1,133 sessions) | the closing straddle is efficient; gate closed |
| **Intraday** next-30-min RV (16k origins) | see discoveries below |
| NVDA, every one of the above | same conclusions; just noisier |

Every arena ran the same protocol: oracle/ceiling first → strong non-strawman baseline → add only
what the diagnostics license → re-check the oracle → stop when it's dry.

## ML didn't work (and *why*, which is the useful part)

Seven neural architectures on the full SPX panel (iTransformer, LSTM, TCN, linear controls, a
surface-attention model, and more) — **all lost to a 6-coefficient ridge.** Not by noise; by the
kind of margin that says "wrong tool." The reason is the same everywhere:

> Forecast error = *approximation* + *estimation* + *information gap* + *irreducible noise*.
> A neural net only fixes the first term. Ours was already ~zero (the structure is simple once
> named). The binding constraints were the **information gap** (the market already priced it) and
> **estimation** (≈1,400 autocorrelated days is a few hundred effective samples — too few to pay
> for a big model's variance). No architecture touches either.

**The one-liner the whole project converged to:** *every predictable piece of volatility has a name*
— persistence, risk premium, state-dependence, the event clock — *and once you name it, a handful
of linear coefficients beats any capacity model trying to approximate it.* The market is the model.

## The two things that *did* work

Because it wasn't all negative:

1. **An intraday "event-clock" model.** On next-30-min RV, an oracle gate actually *opened* — trees
   found structure a linear model missed. We localized it (FOMC 2pm blocks run ~3× normal vol; also
   the open and the close), **named it** as clock × IV-premium interactions, and encoded it as ~50
   explicit linear terms. That beat the trees in every fold (+7% over diurnal-HAR+IV) and re-closed
   the gate. A better *classical* model, discovered by the discipline — not a neural win, but real.
2. **A dated empirical finding:** the ultra-short 0DTE straddle premium that existed in 2022–2024
   has **compressed to zero / slightly inverted in 2025–2026**, exactly as systematic 0DTE selling
   got crowded. Nobody in the literature has this era yet.

## What's in here

```
volforecast/     the leakage-safe package (this is the reusable part)
  timeutil.py      point-in-time guards — look-ahead is impossible by construction
  panel.py         THE single alignment path (feat_* | tgt_* | meta_*, indexed by t0)
  splits.py        walk-forward folds with purge/embargo at every boundary
  data/            Databento adapter + volsolve.py (IV from raw quotes) + a synthetic vendor
  features/        HAR, realized RV, IV surface, intraday structure, forward-only targets
  models/          the HAR-prior + residual-trunk hybrid and the trunks that lost, honestly
  eval/            QLIKE, pinball, HAC Diebold–Mariano, walk-forward runner
scripts/         the core chapters as runnable scripts (oracle, iv_only, study_0dte, intraday_oracle, pulls)
experiments/     lab notebook: README = protocol + results, ledger.csv = every run
tests/           the no-lookahead suite + solver round-trips — the guarantees, automated
configs/         default.yaml (synthetic, no data needed) | spx.yaml | nvda.yaml
archive/codas/   side-studies, not the core result (distributional, deep-hedging, HF pilot)
archive/legacy/  the original project (see the P.S.)
```

## Reproduce

```bash
pip install -e ".[dev]"
make test        # full suite incl. the no-lookahead invariants — runs on synthetic data, no API key
make all         # build panel -> baselines -> comparison table (synthetic)
```

Real data needs `pip install -e ".[databento]"` and `DATABENTO_API_KEY` (a `.env` works). Pulls are
cached; price any pull first with `metadata.get_cost`. `python scripts/status.py` shows cache/ledger
state at a glance.

## Where it goes next

The one live thread is **microstructure**: total RV is saturated by persistence, but there's a faint,
unstable signal in *jumps* from order flow, and the natural pivot is forecasting a **market-impact /
liquidity parameter** (Kyle's λ, or a GLFT market-making parameter) instead of vol — because unlike
vol, the market publishes no "impact surface" for you to lose to. That's a new project, in its own repo.

## P.S.

This started life as a BTC realized-vol forecaster (PatchTST + a pile of baselines). An audit found
two structural look-ahead bugs — a forward-looking target reused as a model input, and two alignment
paths that had silently drifted 60 rows apart. Both are the kind of leak that quietly makes results
look great and mean nothing. Rather than patch them, the project was rebuilt from scratch so that
class of bug is impossible by construction. The old code is frozen in [`archive/legacy/`](archive/legacy/)
as a reminder of why the point-in-time discipline exists.

---

MIT.
