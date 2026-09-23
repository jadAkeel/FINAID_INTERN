# Signal Ceiling Audit — materialized 2026-09-22

Non-promoting diagnostic. Reads only `research/regime_adaptive_selector/artifacts/predictions.parquet`
at origins `<= 266`; the locked evaluation (268–315) is never touched. Reproduce with
`python -m forecast_select.signal_ceiling_audit`.

## What it measures

The empirical accuracy of the active Regime Adaptive selector's 2,545 selected calls over
147 non-locked months, and how wide the plausible range of that accuracy is once monthly
dependence is respected (six-month block bootstrap, 5,000 replicates, seed `20260807`).

## Result

| Statistic | Value |
|---|---:|
| Selected calls / months | 2,545 / 147 |
| Pooled accuracy | **62.44%** |
| Tuning 120–179 | 686 / 1,071 = 64.05% |
| Validation 180–219 | 395 / 675 = 58.52% |
| Confirmation 220–266 | 508 / 799 = 63.58% |
| Block-bootstrap median | 62.60% |
| Block-bootstrap p10 – p90 | **59.96% – 65.11%** |
| Block-bootstrap p95 | 65.81% |
| P(accuracy ≥ 62%) | 61.6% |
| P(accuracy ≥ 65%) | **11.3%** |

Classification: **inconclusive for any stable > 65% claim.**

## Where the accuracy comes from

Accuracy tracks the realized Up-prevalence of the selected set almost exactly, year by year.
Years are the calendar year of the **target** month (the month the predicted change completes):

| Target year | Accuracy | Up-prevalence of selected |
|---|---:|---:|
| 2013 | 69.2% | 69.2% |
| 2015 | 46.7% | 46.7% |
| 2017 | 74.8% | 74.8% |
| 2018 | 52.0% | 50.0% |
| 2019 | 76.2% | 76.7% |
| 2021 | 67.3% | 66.8% |
| 2022 (Jan–Apr) | 30.2% | 30.2% |

Because 2,534 of 2,545 calls are Up, accuracy in any period is simply the fraction of
selected indicators that rose. The year-to-year swing (rolling-12 range 42.5% – 76.2%)
is market breadth, not model skill. This is consistent with the raw data: median lag-1
direction autocorrelation across indicators is 0.01 and no indicator exceeds 0.2.

## Rank profile

Ranks 1–2 hit 66–69%; ranks 3–15 hit 54–65%; ranks 16–19 hit 59–62%. Pooled, expansion
calls (ranks 16–20, 340 calls) run at 61.8% against 62.5% for the top-15 core — but the
gap is not stable. On the delivered active artifact, expansion calls are worse on Tuning
(102/171 = 59.7%), equal on Validation (44/75 = 58.7%), and *better* on Confirmation
(64/94 = 68.1%). A fixed 15-call policy on the same rows would have scored 1,379 / 2,205 =
62.54% versus the delivered 62.44%: dropping expansion buys nothing reliable.

## Files

`metrics/summary.json`, `block_bootstrap.csv`, `rank_and_coverage.csv`, `temporal_drift.csv`,
`window_baselines.csv` (copied from the regime-adaptive experiment ledger). Learning curves,
null-signal tests, error overlap, and oracle bounds are recorded as **unavailable** with the
reason: the artifacts they require were not pre-registered, and inventing them post hoc would
not be evidence.

## What this does not say

It does not say the model is useless. Paired at production's own call count and universe,
ranking by production's prior alone loses 13 hits on Tuning and 13 on Confirmation, with
p10–p90 block-bootstrap intervals entirely below zero, and gains 1 on Validation
(`research/seasonal_prior/`, control arm `prior_only`). The model stack is worth roughly
1.2–1.6 points in two of three windows. What the audit does say is that month-level breadth
moves accuracy by tens of points, so unpaired comparisons across windows or rules are
dominated by breadth. Every bounded challenger in the ledger moved Validation by −5 to +2
hits out of 675.
