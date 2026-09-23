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

---

## Run log

1. First run: aborted by the date guard before any computation. The guard assumed 27–32 days
   between month-ends, but the dates are last business days (gaps of 28–33 days). The guard now
   checks consecutive calendar months.
2. Second run: aborted by the target-alignment assertion. The script had indexed rows from 0,
   but the project numbers positions from 1 (`io.py`). The script now reads through
   `forecast_select.io.load_workbook(maximum_position=267)`, and every one of the 5,921 universe
   targets equals the production artifact's `y_true`. The same error had affected an earlier
   diagnostic; see the erratum in `research/prior_only_selector/README.md`.
3. Third run: completed. **The protocol above was not changed.** Both fixes corrected data
   handling, and neither run produced a result.

## Results

### Existence tests (Tuning-era labels, 37 indicators, 2,000 joint permutations)

| Test | Observed | Null median | Null p95 | p-value |
|---|---:|---:|---:|---:|
| Pooled heterogeneity | 429.5 | 404.2 | 500.6 | **0.314** |
| Split-half reliability (even vs odd years) | **r = 0.006** | −0.003 | 0.150 | 0.467 |
| Breadth seasonality | 5.86 | 10.45 | 19.17 | 0.887 |

**No calendar seasonality is detectable.** The split-half result is the clearest: an indicator's
deviation in a given calendar month during even years has essentially zero correlation with the
same deviation during odd years.

### Candidates at production's call count (hits / calls; Δ = paired hits vs production)

| Candidate | Tuning | Validation | Confirmation |
|---|---:|---:|---:|
| Production | 686/1071 64.05% | 395/675 58.52% | 508/799 63.58% |
| **`H1_overlay` (primary)** | 677 **Δ −9** [p10 −12, p90 −2] | 391 **Δ −4** [−6, −1] | 506 **Δ −2** [−6, +2] |
| `H1_raw_overlay` (unshrunk) | 659 Δ −27 | 396 Δ +1 | 493 Δ −15 |
| `H1_prior` | 668 Δ −18 | 396 Δ +1 | 500 Δ −8 |
| `prior_only` (control) | 673 **Δ −13** [−19, −4] | 396 Δ +1 [−5, +7] | 495 **Δ −13** [−20, −6] |
| `eb_base` | 671 Δ −15 | 395 Δ 0 | 496 Δ −12 |
| `window_ensemble` | 666 Δ −20 | 406 Δ +11 [+5, +17] | 502 Δ −6 |

Fixed 15 calls, all Up: `H1_prior` vs `prior_only` is Δ −2 on Tuning, −5 on Validation, and 0 on
Confirmation.

### Gate (primary)

| Criterion | Result |
|---|---|
| Tuning Δ ≥ 0 | **fail** (−9) |
| Validation Δ > 0 | **fail** (−4) |
| Validation bootstrap p10 ≥ 0 | **fail** (−6) |
| Confirmation Δ ≥ 0 | **fail** (−2) |
| No locked reads | pass |

**Rejected.** Production is unchanged.

## What was learned

1. **Seasonality is absent, not merely weak.** The estimated seasonal SD (method of moments) was
   3–5 points of Up-rate across origins. Split-half reliability shows that variation does not
   recur. It comes from common monthly shocks: one bad October moves every indicator's October
   cell at once, and binomial sampling variance does not account for that. Shrinkage limited
   the damage (−9 on Tuning, against −27 unshrunk) but could not create signal.
2. **Production's model stack is real.** The control arm ranks production's universe by
   production's own prior (equal to a trailing-48 Up-rate, verified to 0.0 difference). It
   loses 13 hits on Tuning and 13 on Confirmation, and both p10–p90 intervals lie below zero.
   The logistic, graph, and group overlay add roughly 1.2–1.6 points in two of three windows.
   Validation is flat.
3. **`window_ensemble` is the window-picking trap made visible.** It gains 11 on Validation with
   p10 = +5, the strongest single-window result in this run, but loses 20 on Tuning and 6 on
   Confirmation. The pre-registration forbids promoting a secondary arm, and the other two
   windows show why.
4. **The two-month misalignment of `direction_lag_12` does not matter in practice.** With no
   seasonality to capture, aligning it correctly would not help. Recorded so no one re-runs this.

Outputs: `metrics/existence_tests.json`, `window_results.csv`, `monthly_records.csv`,
`seasonal_strength.csv`, `summary.json`.
