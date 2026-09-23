# Headroom analysis — where could more accuracy possibly come from? (2026-09-23)

Hindsight diagnostic, not a candidate. Every "oracle" below uses outcomes it could not have
known at the time. The oracles bound what any improvement could deliver, at production's own
per-origin call count and within its own universe. Reproduce with
`python research/oracle_headroom/oracle_headroom.py`. The workbook is read with
`maximum_position=267`; the locked evaluation (268–315) is untouched.

## The decomposition

A monthly directional hit rate comes from two sources:

1. **Which indicators** are chosen: indicators that drift upward rise in most months.
2. **When** the whole market falls: in a broad-down month, every Up call on a drifting
   indicator is still likely to miss.

## Result (accuracy at production's call count)

| | Tuning | Validation | Confirmation | **All 147 months** |
|---|---:|---:|---:|---:|
| **Production** | 64.05% | 58.52% | 63.58% | **62.44%** |
| Long-run drift oracle — knows each indicator's true Up-rate over all non-locked history | 64.33% | 59.11% | 63.58% | 62.71% |
| Window drift oracle — also knows how drift shifted inside each window | 64.80% | 62.67% | 65.21% | 64.36% |
| Breadth timing oracle — production's picks, all called Down in broad-down months | 76.94% | 70.07% | 83.35% | **77.13%** |
| Long-run drift oracle + breadth timing | 78.52% | 70.07% | 79.72% | 76.66% |
| Perfect Up selection — only indicators that rose, up to the cap | 85.34% | 88.89% | 87.73% | 87.03% |

Broad-down months (under half of the universe rose): 21 of 60 in Tuning, 16 of 40 in
Validation, 17 of 47 in Confirmation — **54 of 147 overall**.

## Reading

1. **Choosing indicators is solved.** Production is within **0.27 points** of an oracle that knows
   every indicator's true long-run Up-rate in hindsight, and it matches that oracle exactly on
   Confirmation. No better estimate of drift can add more than about 0.3 points. This is why
   every prior-estimator variant in the registry (windows, shrinkage, ensembles, seasonality)
   lands within noise of production.
2. **Knowing drift shifts in advance would be worth about 1.9 points over production** (window
   oracle, 64.36%), and 4.2 points on Validation. No causal estimator captures it: shorter
   trailing windows, which adapt faster, score lower (`research/prior_only_selector/`).
3. **Almost all the headroom is market timing.** Calling Down in exactly the broad-down months
   would lift production's picks by **14.7 points**. This is the only lever that could justify
   Down calls, which the project requires.
4. **Market timing shows no detectable signal.** Production's own walk-forward breadth forecast
   correlates with realized breadth at **0.035**. Its AUC for identifying broad-down months is
   **0.52** pooled (0.47 / 0.62 / 0.51 by window). Last month's breadth, 3-month breadth, and
   regime stress score AUC 0.48, 0.48, and 0.50. Realized breadth has autocorrelation −0.04 at
   lag 1 and −0.01 at lag 2: whether the market rises next month is unrelated to this month.
5. **A Down-timing signal must be right 61% of the time it fires just to break even.**
   Flipping production's picks to Down gains 6.89 hits on average in a broad-down month, but
   loses 10.75 in a broad-up month, because up months are more one-sided. Broad-down months are
   only 36.7% of the sample. A signal would therefore need about 1.66× lift over the base rate
   at its operating point, far beyond anything with AUC near 0.52.
6. **The coverage contract caps an Up-only selector at 87%** even with perfect foresight,
   because broad-down months do not contain 15–20 rising indicators.

## Conclusion

On this information set (anonymous price histories only), about 62–64% is the ceiling. The
remaining variation is market direction, which is unpredictable month to month. Production's
Down path makes 11 calls in 147 months, 7 of them correct. That conservatism is correct:
without breadth skill, frequent Down calls lose, as the Down V2/V3 and mixed-direction trials in
`docs/EXPERIMENT_REGISTRY.md` found. Any future claim of a large accuracy gain must first show
breadth-forecasting skill above AUC 0.5. Otherwise the gain is noise or leakage.

## Files

`metrics/headroom.csv`, `monthly_oracles.csv`, `summary.json`, `breadth_forecast_skill.json`,
`breadth_forecast_skill_monthly.csv`.
