# Calendar-seasonal prior — pre-registration (2026-09-23)

This file was committed **before** the experiment was run. The Results section is appended
afterwards; everything above it is the frozen protocol.

## Why this was never tested

Every model in this repository builds its lagged features from the latest *known* change.
`features.py` shifts levels by the one-month availability lag, so at origin `t` the
feature `changes` is the move `t-2 → t-1`, and `direction_lag_12` is the move `t-14 → t-13`.

The target is the move `t → t+1`. Its same-calendar-month counterpart one year earlier is
`t-12 → t-11`. The existing annual lags therefore sit **two months off season**, and no
experiment in `docs/EXPERIMENT_REGISTRY.md` conditions on calendar month. These indicators
are month-end financial series (commodities, currencies, fixed income, equity indices,
sectors). For such assets there is published evidence that relative performance in a given
calendar month tends to recur across years.

## Hypothesis

An indicator's Up-frequency in the target's calendar month carries directional information
beyond its trailing 48-label Up-rate.

## Causal contract

- At origin `t` the direction label `d_s = 1[X_{s+1} > X_s]` is usable only for `s ≤ t-2`.
- The scored target is `d_t`. The script asserts that its targets equal the production
  artifact's `y_true` on every universe row.
- No level position above 267 is read. 267 is the target level of origin 266. The locked
  evaluation (origins 268–315) is untouched.

## Existence tests (Tuning-era labels only, `s ≤ 177`)

Indicators with at least 60 Tuning-era labels are included. The null permutes calendar-month
labels **jointly across indicators** (2,000 permutations, seed `20260923`), which keeps
same-month cross-sectional dependence.

1. **Pooled heterogeneity** — `Σ n_im (r_im − r_i)² / (r_i (1 − r_i))` over indicator × month
   cells with at least 3 labels. Seasonality is declared detected when `p < 0.05`.
2. **Split-half reliability** — the correlation between month deviations estimated on
   even calendar years and on odd calendar years.
3. **Breadth seasonality** — month-of-year variation in cross-sectional Up breadth.
   Descriptive only; relevant to the expansion and Down decisions.

## Seasonal score (no tuned parameters)

At each origin, using labels `≤ t-2`:

- `dev_i` is indicator `i`'s all-history Up-rate in the target calendar month minus its
  all-history Up-rate.
- `τ²` is the method-of-moments seasonal variance, pooled over every indicator × month cell:
  mean of `dev² − sampling variance`, floored at 0.
- `δ_i = w_i · dev_i`, with `w_i = τ² / (τ² + sampling_i)`. Cells with fewer than 3 labels
  get `δ = 0`.

## Candidates

All candidates are compared at **production's own per-origin call count** (15–20) and within
production's selection universe (`level_c_ready` and not adaptively excluded). A candidate
calls Up, except that a row production itself called Down keeps its Down call. This matches
the Dr's contract, which requires Down and expansion.

| Name | Score | Role |
|---|---|---|
| **`H1_overlay`** | production `selection_score + 0.5·δ` | **Primary.** 0.5 is production's own `indicator_prior_weight`, so this is the first-order effect of replacing the prior with `prior + δ` in the existing blend. |
| `H1_raw_overlay` | `selection_score + 0.5·dev` (unshrunk) | Diagnostic: shows whether shrinkage matters |
| `H1_prior` | `indicator_prior + δ` | Science: the seasonal prior alone |
| `prior_only` | `indicator_prior` | Control for `H1_prior` |
| `eb_base` | trailing-48 Up-rate shrunk toward the cross-section | Secondary: shrinkage without seasonality |
| `window_ensemble` | mean percentile rank of trailing Up-rates over 36/48/60/96/all | Secondary: robustness to window choice |

`H1_prior` and `prior_only` are also reported at a fixed 15 calls, all Up, to answer the
science question free of production's coverage schedule.

## Pre-registered gate (primary candidate only)

Paired monthly hit deltas against production, with a six-month moving-block bootstrap
(5,000 replicates, seed `20260923`):

1. Tuning total delta ≥ 0 (sanity check: nothing is fitted on Tuning, but a loss there is disqualifying)
2. Validation total delta > 0
3. Validation bootstrap p10 ≥ 0
4. Confirmation total delta ≥ 0
5. No locked reads

The secondary candidates are reported whatever the outcome and **cannot** be promoted from
this run. Picking the best of six after seeing the results would be exactly the window-picking
that `research/prior_only_selector/` warns about.

If the primary passes, it becomes the single pre-registered candidate for the one-time locked
evaluation. That read requires confirmation first, because it cannot be undone.

## Reproduce

```bash
python research/seasonal_prior/seasonal_prior.py
```

Outputs go to `metrics/`: `existence_tests.json`, `window_results.csv`,
`monthly_records.csv`, `seasonal_strength.csv`, and `summary.json`.
