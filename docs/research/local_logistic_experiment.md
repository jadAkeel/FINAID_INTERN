# Local / Indicator-Specific Logistic Regression for the Up Selector

Experiment id: `local_logistic_selector`
Status: `rejected` (research evidence only; no production change)
Branch: `research/local-logistic`
Date: 2026-09-22

**Locked origins 268–315 were not read or used during this experiment.**

---

## 1. Objective

The production Up Selector is a single pooled ("global") logistic regression
over 44 causal features plus a one-hot `indicator_id` block. The one-hot block
lets the model learn *"this row belongs to X1"* — an indicator-specific
**intercept** — but every slope is shared. It cannot express *"momentum has a
strong positive effect for X1 and almost none for X12."*

This experiment asks two questions.

- **Main:** does allowing indicator-specific coefficients improve
  out-of-sample performance over the shared global logistic regression?
- **Secondary:** can indicator-specific behaviour be captured without the
  variance of fitting 50 fully independent models?

Four candidates were built and compared against the baseline on identical
rows, identical origins and the identical downstream selector.

| Candidate | Description |
| --- | --- |
| A | Global Logistic (production baseline, unchanged) |
| B | Pure Local Logistic — one model per indicator |
| C | Global + Local fixed shrinkage |
| D | Global + Local sample-size-aware shrinkage |
| E | Global Logistic + indicator-specific interaction terms |

---

## 2. Baseline reproduced from source

The baseline was regenerated with the same code path used for every
candidate, not copied from the README.

| Setting | Value | Source |
| --- | --- | --- |
| Estimator | `LogisticRegression` | `src/forecast_select/uptrend_model.py:80` |
| Penalty / solver | `l2` / `lbfgs` | same |
| `C` | `0.25` | `configs/uptrend_model.yaml` |
| `max_iter` | `120` | same |
| `class_weight` | **not set** (unweighted) | `uptrend_model.py:80` |
| `random_state` | `20260727` | `configs/config.yaml` |
| Numeric features | **44**, `FEATURE_COLUMNS` | `uptrend_model.py:17` |
| Preprocessing | median `SimpleImputer(add_indicator=True)` → `StandardScaler` | `uptrend_model.py:61` |
| `indicator_id` | `OneHotEncoder(handle_unknown="ignore")` | `uptrend_model.py:70` |
| Total fitted coefficients | **114** at origin 120 → **125** at origin 266 (44 numeric + 33 missingness indicators + 37→48 one-hot; the one-hot block grows as more indicators become eligible) | measured |
| Availability lag | 1 month — features use observations `≤ t-1` | `configs/uptrend_model.yaml` |
| Training labels | `≤ t-2` (`causal_training_rows`) | `src/forecast_select/validation.py:52` |
| Graph | signed correlation on first-differences, estimated through origin 119, `alpha = 0.35` | `uptrend_pipeline.py:182` |
| Prior blend | `0.5 × trailing-48-month Up rate + 0.5 × p_up`, requires ≥ 24 indicator months | `indicator_selection.py:112` |
| Selection | top 15 unique indicators per origin by `selection_score` | `indicator_selection.py:131` |

Reproduction check: the registered development result is 926/1500
(61.7333%) over origins 120–219. This experiment produced **579/900 on
tuning (120–179) plus 347/600 on validation (180–219) = 926/1500**, an exact
match. `tests/leakage/test_local_logistic_causality.py` additionally asserts
that the research selector stage reproduces `build_uptrend_predictions`
bit-for-bit on `p_up_raw`, `p_up`, `selection_score`, `indicator_prior`,
`accepted` and `predicted_direction`.

---

## 3. Temporal protocol

Identical to production, and unchanged for every candidate.

| Window | Origins | Use |
| --- | --- | --- |
| Tuning | 120–179 (60 months) | **all** hyperparameter selection |
| Validation | 180–219 (40 months) | generalization check, frozen parameters |
| Confirmation | 220–266 (47 months) | stability check, frozen parameters |
| Locked | 268–315 | **never read** |

- No random splitting anywhere. Every fit is walk-forward at a single origin.
- At origin `t`: features use observations `≤ t-1`; training labels stop at
  `t-2`.
- Imputers, scalers, one-hot encoders, local models and the interaction model
  are all re-fitted at every origin on that causal slice only.
- **Locked-set guard:** `configs/local_logistic_experiment.yaml` pins
  `maximum_readable_position: 267`, and the workbook is loaded with `nrows`
  capped there. Rows 268–316 are never read from disk. Every origin list also
  passes `assert_origins_unlocked`, which raises on any origin `≥ 268`.

---

## 4. Local Logistic design (Experiment A)

At each origin, one logistic regression is fitted per indicator on
**same-indicator history only**. A local model sees roughly one row per
month, so the 44 global predictors are not usable; three compact candidate
sets, all strict subsets of the production panel, were put on the tuning
grid.

| Set | Size | Features |
| --- | --- | --- |
| `lean5` | 5 | `direction_lag_1`, `momentum_3`, `momentum_6`, `distance_mean_12`, `cross_section_rank` |
| `core8` | 8 | `direction_lag_1`, `change_lag_1`, `momentum_3`, `momentum_6`, `rolling_std_12`, `robust_z_12`, `distance_mean_12`, `cross_section_rank` |
| `wide12` | 12 | `core8` + `direction_lag_2`, `direction_lag_3`, `momentum_12`, `cross_section_breadth` |

**Selected on tuning: `core8` (8 predictors).**

Fixed (not tuned), inherited from the existing `directional_downside`
experiment which used the same per-indicator construction:
`solver=liblinear`, `max_iter=500`, `minimum_local_class_rows=8`,
median imputation → standardization, `penalty=l2`.

### Regularization and minimum history

Searched on tuning only: `C ∈ {0.01, 0.03, 0.10, 0.25}` and
`minimum_local_rows ∈ {36, 48, 60}`.

| Candidate | Selected `C` | Selected minimum local rows |
| --- | --- | --- |
| B (pure local) | **0.01** | **60** |
| C / D (shrinkage) | **0.25** | **60** |

An indicator with fewer than the minimum labelled rows, or fewer than 8 rows
in its minority class, gets **no local model and falls back to the global
probability**. Fallback frequency over origins 120–266:

| Window | Fallback rate |
| --- | --- |
| Tuning (120–179) | 20.54% |
| Validation (180–219) | 11.20% |
| Confirmation (220–266) | 8.71% |
| Overall | 13.86% |

Local training history ranged from 0 to 240 rows (mean 131.1). Even at the
last permitted origin a local model never had more than 240 observations for
8 predictors.

---

## 5. Shrinkage / partial pooling

**Fixed shrinkage (C):** `p_hybrid = w · p_local + (1 − w) · p_global`, with
`w` searched on tuning over `{0.10, 0.20, 0.30, 0.40, 0.50}`.
**Selected `w = 0.40`.** Fallback rows keep `p_local = p_global`, so the
blend is a no-op there.

**Sample-aware shrinkage (D):** a fixed, predefined ramp,

```
w(n) = w_max · clip((n − n_min) / (n_ref − n_min), 0, 1),   w = 0 on fallback rows
```

with `w_max` from the same weight grid and `n_ref ∈ {120, 180, 240}`, reusing
C's frozen feature set, `C` and minimum-history rule.
**Selected `w_max = 0.10`, `n_ref = 180`** (mean effective weight 0.0569).

---

## 6. Indicator-specific interactions (Experiment B)

One pooled logistic regression, identical base design to the baseline, plus
`indicator_id × feature` products for a small predefined feature list:

`momentum_3`, `momentum_6`, `direction_lag_1`, `rolling_std_12`,
`distance_mean_12`, `cross_section_rank`.

Interactions are formed on the **scaled** features after the same causal
imputer/scaler. Only indicators present in the training slice are encoded, so
the block grows from 37 × 6 = 222 columns at origin 120 to 48 × 6 = 288 at
origin 266.

| Measure | Baseline (A) | Interaction (E) |
| --- | --- | --- |
| Raw features | 44 | 44 |
| Base encoded block (numeric + missingness + one-hot) | 114 → 125 | 114 → 125 |
| Interaction terms | 0 | 222 → 288 |
| Total trainable coefficients | **114 → 125** | **336 → 413** |

(Values shown at origin 120 → origin 266.)

Regularization is strong L2; `C` was searched on tuning over
`{0.01, 0.05, 0.10, 0.25}` and **0.01 was selected** — the most heavily
penalized value on the grid, which is itself a signal. `max_iter=500`, fixed.
No Elastic Net or other model family was introduced.

---

## 7. Tuning results — every candidate lost on its own tuning window

220 configurations were scored on tuning origins 120–179 against a baseline
of **579/900 (64.33%)**.

**Zero of 220 configurations beat the baseline.** The best was −1 hit.

| Family | Configurations | Best vs baseline | Worst vs baseline |
| --- | --- | --- | --- |
| Pure local | 36 | −2 hits | −25 hits |
| Fixed shrinkage | 180 | −1 hit | −15 hits |
| Sample-aware shrinkage | 15 | −5 hits | −13 hits |
| Interaction | 4 | −9 hits | −20 hits |

This matters for interpretation: the usual failure mode for local models is
*better on tuning, worse later*. That is **not** what happened. The local and
interaction candidates were already no better on the window whose data
selected their own hyperparameters.

---

## 8. Frozen results across all three periods

Parameters were frozen after §7 and never re-tuned.

### Top-15 selector accuracy

| Candidate | Tuning (120–179) | Validation (180–219) | Confirmation (220–266) |
| --- | --- | --- | --- |
| A. Global Logistic (baseline) | 579/900 (64.33%) | 347/600 (57.83%) | 436/705 (61.84%) |
| B. Pure Local Logistic | 577/900 (64.11%) | 354/600 (59.00%) | 436/705 (61.84%) |
| C. Global + Local fixed shrinkage | 578/900 (64.22%) | 355/600 (59.17%) | 435/705 (61.70%) |
| D. Global + Local sample-aware shrinkage | 574/900 (63.78%) | 350/600 (58.33%) | 435/705 (61.70%) |
| E. Global + indicator interactions | 570/900 (63.33%) | 351/600 (58.50%) | 438/705 (62.13%) |

### Hit delta versus baseline

| Candidate | Tuning | Validation | Confirmation |
| --- | --- | --- | --- |
| B. Pure Local | −2 (−0.22 pp) | **+7 (+1.17 pp)** | 0 (+0.00 pp) |
| C. Fixed shrinkage | −1 (−0.11 pp) | **+8 (+1.33 pp)** | −1 (−0.14 pp) |
| D. Sample-aware shrinkage | −5 (−0.56 pp) | +3 (+0.50 pp) | −1 (−0.14 pp) |
| E. Interactions | −9 (−1.00 pp) | +4 (+0.67 pp) | +2 (+0.28 pp) |

Every candidate's sign **flips across windows**: negative on tuning, positive
on validation, ~zero on confirmation. No candidate is positive in two
consecutive windows.

### Raw `p_up` quality, before the graph and prior blending

| Candidate | Period | Accuracy | AUC | Brier | Log loss |
| --- | --- | --- | --- | --- | --- |
| A. Global | tuning | 57.87% | 0.5538 | 0.2557 | 0.7140 |
| A. Global | validation | 51.03% | **0.4963** | 0.2591 | 0.7134 |
| A. Global | confirmation | 55.17% | 0.5198 | 0.2496 | 0.6933 |
| B. Pure Local | tuning | 57.20% | 0.5264 | 0.2488 | 0.6942 |
| B. Pure Local | validation | 53.71% | **0.5292** | 0.2492 | 0.6918 |
| B. Pure Local | confirmation | 54.76% | 0.5308 | 0.2470 | 0.6870 |
| C. Fixed shrinkage | tuning | 57.83% | 0.5531 | 0.2503 | 0.6986 |
| C. Fixed shrinkage | validation | 51.94% | 0.5117 | 0.2552 | 0.7044 |
| C. Fixed shrinkage | confirmation | 55.17% | 0.5161 | 0.2499 | 0.6934 |
| D. Sample-aware | tuning | 57.91% | 0.5529 | 0.2551 | 0.7117 |
| D. Sample-aware | validation | 51.20% | 0.4998 | 0.2582 | 0.7110 |
| D. Sample-aware | confirmation | 55.26% | 0.5199 | 0.2493 | 0.6925 |
| E. Interactions | tuning | 57.29% | 0.5150 | 0.2559 | 0.7088 |
| E. Interactions | validation | 51.83% | 0.4891 | 0.2540 | 0.7025 |
| E. Interactions | confirmation | 55.99% | 0.5153 | 0.2473 | 0.6882 |

Two things stand out.

- **Local models are consistently better calibrated.** Pure local wins Brier
  and log loss in all three windows. The global model is badly overconfident
  on Up: mean `p_up` 0.671 against a 57.0% Up base rate on tuning. On
  validation the global model's calibration curve is non-monotone (bottom
  probability quintile observed rate 52.9%, top quintile 54.3%), while the
  local model's is monotone (49.1% → 51.7% → 55.4% → 57.7%).
- **Better calibration did not become better ranking inside the selector.**
  Pure local's AUC is more stable across windows (0.526 / 0.529 / 0.531 vs
  the global 0.554 / 0.496 / 0.520) but is never high, and the top-15 stage
  cares only about ordering.

### Paired monthly comparison (circular block bootstrap, 6-month blocks, 500 replicates)

Monthly accuracy differences, not row-level — 50 indicators in one month
share a regime and are not independent.

| Candidate | Period | Mean monthly delta | Better/worse/equal months | 90% interval | Excludes zero |
| --- | --- | --- | --- | --- | --- |
| B. Pure Local | tuning | −0.22 pp | 18/23/19 | [−2.22, +1.56] pp | no |
| B. Pure Local | validation | +1.17 pp | 9/4/27 | [+0.00, +2.33] pp | yes |
| B. Pure Local | confirmation | +0.00 pp | 11/7/29 | [−1.42, +1.42] pp | no |
| C. Fixed shrinkage | tuning | −0.11 pp | 11/15/34 | [−1.00, +0.89] pp | no |
| C. Fixed shrinkage | validation | +1.33 pp | 10/3/27 | [+0.50, +2.17] pp | yes |
| C. Fixed shrinkage | confirmation | −0.14 pp | 7/7/33 | [−1.28, +0.85] pp | no |
| D. Sample-aware | tuning | −0.56 pp | 0/5/55 | [−0.78, −0.22] pp | yes (negative) |
| D. Sample-aware | validation | +0.50 pp | 3/0/37 | [+0.00, +1.00] pp | no |
| D. Sample-aware | confirmation | −0.14 pp | 1/2/44 | [−0.57, +0.28] pp | no |
| E. Interactions | tuning | −1.00 pp | 8/17/35 | [−1.89, −0.22] pp | yes (negative) |
| E. Interactions | validation | +0.67 pp | 7/4/29 | [−0.33, +1.67] pp | no |
| E. Interactions | confirmation | +0.28 pp | 7/4/36 | [−0.28, +0.85] pp | no |

Only two intervals exclude zero in a favourable direction, both on
validation, and both are contradicted by a negative tuning interval (E,
sample-aware D) or a null confirmation interval (B, C). With three windows
and four candidates, one or two nominally significant validation intervals
is what noise produces.

---

## 9. Selection changes (top-15 overlap)

| Candidate | Period | Mean overlap | Added calls correct | Removed calls correct | Net hits |
| --- | --- | --- | --- | --- | --- |
| B. Pure Local | tuning | 11.95/15 (79.7%) | 105/183 (57.4%) | 107/183 (58.5%) | −2 |
| B. Pure Local | validation | 13.68/15 (91.2%) | 31/53 (58.5%) | 24/53 (45.3%) | +7 |
| B. Pure Local | confirmation | 12.68/15 (84.5%) | 68/109 (62.4%) | 68/109 (62.4%) | 0 |
| C. Fixed shrinkage | tuning | 13.75/15 (91.7%) | 43/75 (57.3%) | 44/75 (58.7%) | −1 |
| C. Fixed shrinkage | validation | 14.25/15 (95.0%) | 20/30 (66.7%) | 12/30 (40.0%) | +8 |
| C. Fixed shrinkage | confirmation | 13.77/15 (91.8%) | 33/58 (56.9%) | 34/58 (58.6%) | −1 |
| D. Sample-aware | tuning | 14.83/15 (98.9%) | 4/10 (40.0%) | 9/10 (90.0%) | −5 |
| D. Sample-aware | validation | 14.82/15 (98.8%) | 6/7 (85.7%) | 3/7 (42.9%) | +3 |
| D. Sample-aware | confirmation | 14.72/15 (98.2%) | 8/13 (61.5%) | 9/13 (69.2%) | −1 |
| E. Interactions | tuning | 13.75/15 (91.7%) | 39/75 (52.0%) | 48/75 (64.0%) | −9 |
| E. Interactions | validation | 13.88/15 (92.5%) | 28/45 (62.2%) | 24/45 (53.3%) | +4 |
| E. Interactions | confirmation | 13.36/15 (89.1%) | 46/77 (59.7%) | 44/77 (57.1%) | +2 |

The candidates barely change the selected set (80–99% overlap), and where
they do, added and removed calls land at almost the same accuracy. The
apparent validation gain for B and C comes from *removed* calls being
unusually bad (45.3% and 40.0%) rather than from added calls being unusually
good — a pattern that does not repeat on tuning or confirmation.

---

## 10. Per-indicator results

Full table: `research/local_logistic/metrics/per_indicator.csv`.

Out-of-sample (validation + confirmation), raw accuracy, five worst and five
best for local modelling:

| Indicator | Mean local history | Global | Pure local | Local − Global | Times selected (global) |
| --- | --- | --- | --- | --- | --- |
| X33 | 195 | 56.41% | 44.65% | −11.76 pp | 32 |
| X30 | 195 | 56.04% | 48.59% | −7.45 pp | 0 |
| X44 | 134 | 55.35% | 49.47% | −5.88 pp | 0 |
| X31 | 195 | 54.41% | 49.79% | −4.63 pp | 26 |
| X39 | 195 | 66.36% | 61.91% | −4.44 pp | 87 |
| X46 | 142 | 47.34% | 55.03% | +7.69 pp | 0 |
| X17 | 142 | 49.47% | 57.47% | +8.01 pp | 0 |
| X47 | 148 | 46.41% | 55.85% | +9.44 pp | 0 |
| X36 | 195 | 49.10% | 61.04% | +11.94 pp | 19 |
| X48 | 142 | 40.59% | 58.16% | +17.58 pp | 0 |

**The heterogeneity is real, but it is aimed at the wrong indicators.**

A permutation null was built by keeping each indicator's row count and the
observed 26.05% global/local disagreement rate, then reassigning which model
was right on each disagreeing row by a fair coin
(`research/local_logistic/metrics/per_indicator_spread_null.json`):

- Observed SD of per-indicator (local − global) accuracy: **5.37 pp**
- Null median SD: 3.75 pp; null 95th percentile: 4.44 pp
- Empirical p-value: **0.000** — the observed spread genuinely exceeds noise.

But splitting by whether the selector ever picks the indicator
(`metrics/selection_reach.csv`):

| Group | Indicators | Mean (local − global) accuracy |
| --- | --- | --- |
| Ever selected by the baseline | 28 | **−0.51 pp** |
| Never selected by the baseline | 20 | **+2.87 pp** |
| Selection-weighted over all top-15 calls | — | **−0.68 pp** |

Four of the five largest local wins (X46, X17, X47, X48) are indicators the
Up Selector **never selects**. Local modelling improves prediction where it
does not matter and is slightly negative where it does.

---

## 11. Coefficient analysis

### Do indicators respond differently? Yes.

`core8`, `C = 0.25`, fitted at origin 266 — `momentum_3` slope by indicator:

| Indicator | Local rows | `momentum_3` | `direction_lag_1` | `distance_mean_12` |
| --- | --- | --- | --- | --- |
| X1 | 90 | −0.1571 | +0.1883 | +0.1051 |
| X2 | 183 | −0.0451 | +0.1643 | −0.2153 |
| X3 | 188 | **+0.2822** | +0.3366 | −0.2668 |
| X14 | 43 | −0.3059 | +0.3937 | +0.0310 |
| X16 | 179 | **+0.6161** | −0.0591 | −0.2693 |
| X19 | — | **−0.4534** | — | — |

Across 46 fitted indicators the `momentum_3` slope spans −0.4534 to +0.6161
(SD 0.2113) around a mean of +0.0396. The cross-indicator spread dwarfs the
mean for every feature — exactly the heterogeneity the global model cannot
represent.

### Are those differences stable? Yes, more than expected.

`metrics/coefficient_stability.csv` reports across-origin dispersion, but
expanding windows overlap almost completely between consecutive origins, so
that measure is optimistic by construction. A harder test compares slopes
fitted at origin 179 against origin 266 — training slices differing by 87
extra months (`metrics/coefficient_reproducibility.csv`):

| Feature | Correlation (179 vs 266) | Sign agreement |
| --- | --- | --- |
| `direction_lag_1` | 0.863 | 87.2% |
| `momentum_3` | 0.866 | 69.2% |
| `momentum_6` | 0.806 | 71.8% |
| `rolling_std_12` | 0.799 | 66.7% |
| `robust_z_12` | 0.788 | 76.9% |
| `change_lag_1` | 0.751 | 66.7% |
| `cross_section_rank` | 0.748 | 71.8% |
| `distance_mean_12` | 0.631 | 76.9% |

Correlations of 0.63–0.87 across 87 additional months of data mean the
per-indicator slopes are **not** noise that reshuffles every refit. This is
an important negative-result caveat: the failure is not instability.

### Interaction coefficients

At origin 266, the 288 interaction terms carry mean `|coefficient|` 0.0202
(max 0.1209) against 0.0401 (max 0.2278) for the 125 base terms — the
indicator-specific block is roughly half the magnitude of the pooled block,
under the heaviest penalty on the grid (`C = 0.01`). The largest terms are
`indicator[X24]×cross_section_rank` (−0.1209),
`indicator[X2]×direction_lag_1` (+0.1161) and
`indicator[X39]×cross_section_rank` (+0.1065).

---

## 12. Overfitting analysis

The expected symptom — better tuning, worse validation/confirmation — **did
not occur**.

| Candidate | Tuning | Validation | Confirmation |
| --- | --- | --- | --- |
| B | −2 | +7 | 0 |
| C | −1 | +8 | −1 |
| D | −5 | +3 | −1 |
| E | −9 | +4 | +2 |

All four candidates are *worse* on tuning and better only on validation. So
the evidence is not "local models overfit the tuning window". It is "local
models carry roughly zero net selector signal, and the window-to-window
differences are noise". Supporting facts:

- 0 of 220 tuning configurations beat the baseline.
- No candidate has the same sign in two consecutive windows.
- The tuning-best interaction `C` is the smallest on the grid (0.01), i.e.
  the search preferred to shrink the indicator-specific block toward zero.
- Sample-aware shrinkage (D), which by design gives local models the least
  influence (mean effective weight 0.057), is not systematically better or
  worse than fixed shrinkage at `w = 0.40` — the selector is insensitive to
  how much local signal is mixed in.

---

## 13. Why the selector is insensitive — a structural finding

Because the candidates barely moved the selector, we measured how much the
top-15 result depends on the logistic signal at all
(`metrics/signal_contribution.csv`). The graph, prior and top-15 stages are
held fixed; only `p_up` is replaced.

| `p_up` variant | Tuning | Validation | Confirmation |
| --- | --- | --- | --- |
| Global Logistic (production) | 579/900 (64.33%) | 347/600 (57.83%) | 436/705 (61.84%) |
| **Constant 0.5 (prior only)** | 570/900 (63.33%) | **352/600 (58.67%)** | **438/705 (62.13%)** |
| Global probabilities shuffled within each origin | 555/900 (61.67%) | 343/600 (57.17%) | **442/705 (62.70%)** |

Selecting purely on the trailing 48-month Up prior — discarding the logistic
model entirely — **matches the production selector on tuning (−9 hits) and
beats it on both validation (+5) and confirmation (+2)**. Even a randomly
shuffled signal is within noise out of sample.

`selection_score = 0.5 × indicator_prior + 0.5 × p_up_after_graph`, and the
graph stage further compresses `p_up` toward the cross-sectional consensus.
With out-of-sample raw AUC near 0.50, the logistic half contributes little
ordering information, and the prior half dominates.

This explains the whole experiment: **no change to the underlying logistic
signal — local, hybrid, or interaction — can move a selector that is mostly
not listening to it.** It also means the raw-model comparison in §8 is the
more informative view of the modelling question, and the selector comparison
mostly measures the prior.

This is a diagnostic observation from within this experiment's scope. It is
**not** a recommendation to change the production selector; it is flagged for
separate work.

---

## 14. Decision

> **GLOBAL LOGISTIC REMAINS PREFERRED.**
> The local, hybrid and interaction candidates are retained as research
> evidence only (`LOCAL LOGISTIC RESEARCH ONLY`). No production change is
> proposed or made.

Rationale:

1. Zero of 220 tuning configurations beat the baseline on the window that
   selected their own hyperparameters.
2. No candidate is positive in two consecutive evaluation windows; all sign
   flips are within block-bootstrap noise.
3. The single largest frozen effect (C, +8 validation hits) is contradicted
   by −1 on tuning and −1 on confirmation.
4. Candidates add complexity (50 extra models, or 336–413 coefficients versus
   114) for no demonstrated out-of-sample gain.

### Did partial pooling help?

Marginally, relative to pure local — fixed shrinkage at `w = 0.40` was the
best-scoring family on tuning (−1 hit vs pure local's −2) and the best on
validation (+8 vs +7). But "best of several candidates that all fail" is not
a promotion case. The partial-pooling *hypothesis* is not refuted; it simply
produced no measurable selector gain here.

### Did indicator interactions help?

No. Candidate E was the **worst** family on tuning (−9 to −20 hits), the
tuning search chose the most heavily penalized `C` on the grid, and its
validation AUC (0.4891) is below chance.

---

## 15. What was learned (positive findings inside a negative result)

1. **Indicator-specific structure is real.** Per-indicator accuracy spread
   exceeds a matched coin-flip null at p ≈ 0.000, and per-indicator slopes
   reproduce at r = 0.63–0.87 across 87 extra months.
2. **Local models are better calibrated** than the global model in all three
   windows (Brier and log loss), and fix the global model's Up
   overconfidence and non-monotone validation calibration curve.
3. **The heterogeneity is aimed at indicators the selector never picks**
   (+2.87 pp on never-selected indicators, −0.51 pp on selected ones).
4. **The top-15 selector is dominated by the trailing prior, not by the
   logistic signal** (§13).

---

## 16. Future extensions (not implemented)

- A **hierarchical / Bayesian logistic** with indicator slopes drawn from a
  shared prior is the principled version of §5. Given that simple partial
  pooling showed no selector gain and that §13 indicates the selector barely
  uses `p_up`, the payoff would likely be small; it should only be attempted
  after the prior-dominance question is settled. No probabilistic framework
  was introduced here.
- **Selection-aware local modelling**: fit local models only for indicators
  that reach the top 15, or weight the local objective by selection
  probability. §10 is direct evidence that untargeted local modelling wastes
  its gains.
- **Recalibration of the global model.** The global model is measurably
  overconfident; the local models' superior Brier/log loss suggests a
  causal recalibration layer is worth a separate experiment.

---

## 17. Reproduction

```bash
python -m forecast_select.local_logistic_runner tune       # tuning search, origins 120-179
python -m forecast_select.local_logistic_runner evaluate   # frozen scoring, origins 120-266
python -m forecast_select.local_logistic_report            # render metrics/tables.md
python research/local_logistic/diagnostics.py              # stability, null, prior ablation
python -m pytest tests/unit/test_local_logistic.py tests/leakage/test_local_logistic_causality.py
```

Artifacts:

| Path | Contents |
| --- | --- |
| `research/local_logistic/artifacts/predictions.parquet` (and `.csv`) | per-origin, per-indicator audit rows: `p_global`, `p_local`, `p_hybrid`, `p_hybrid_sample_aware`, `p_interaction`, ranks, selections, correctness, `local_training_sample_count`, `local_fallback_used` |
| `research/local_logistic/artifacts/full_signals.parquet` | raw walk-forward signals, origins 120–266 |
| `research/local_logistic/artifacts/tuning_signals.parquet` | all 12 local specs and 4 interaction `C` values, origins 120–179 |
| `research/local_logistic/artifacts/local_coefficients.parquet` | per-origin, per-indicator local slopes |
| `research/local_logistic/artifacts/interaction_coefficients.parquet` | per-origin interaction coefficients |
| `research/local_logistic/metrics/` | `summary.json`, `tuning_search.csv`, `per_indicator.csv`, `overlap_*.csv`, `calibration.csv`, `coefficient_stability.csv`, `coefficient_reproducibility.csv`, `signal_contribution.csv`, `tables.md` |

---

## 18. Locked set status

**Locked origins 268–315 were not read or used during this experiment.**

- The workbook is loaded with `nrows` capped at position 267
  (`maximum_readable_position`), so rows 268–316 never enter memory.
- `assert_origins_unlocked` raises on any origin `≥ 268` and is called on
  every origin list before any fitting.
- `summary.json` records `maximum_origin_evaluated: 266`,
  `maximum_readable_position: 267`, `locked_origins_read: false`.
- `tests/leakage/test_local_logistic_causality.py` asserts the panel, targets
  and prediction artifact all stop below origin 268, and that a locked origin
  is rejected.
- No locked labels, accuracies, correctness values or feature statistics were
  computed, inspected, printed or written at any point.
