# Selection Group, Selection Score, and Accuracy: Negative Results Registry

Last reviewed: 2026-08-29

This registry prevents failed experiments from being repeated under new names. A listed approach should only be revisited if the scientific hypothesis, information source, and temporal evaluation design materially change.

## Current decision

- No challenger passed the promotion gates.
- The active model remains the Regime Adaptive Bidirectional Selector without a recent-miss or group-stability overlay.
- The reference release is `forward_breadth_dynamic_cap_v3`, selecting 15–20 indicators per month.
- March–May 2026 has already been inspected and is no longer a blind acceptance holdout.

## Reference baseline

| Period | Directional AUC | Selection AUC | Accuracy | Correct/Total | Down calls |
|---|---:|---:|---:|---:|---:|
| Tuning 120–179 | 0.5795 | 0.5413 | 64.05% | 686/1071 | 4 |
| Validation 180–219 | 0.5094 | 0.4015 | 58.52% | 395/675 | 0 |
| Development 120–219 | 0.5533 | 0.5092 | 61.91% | 1081/1746 | 4 |
| Confirmation 220–266 | 0.5380 | 0.4411 | 63.58% | 508/799 | 7 |

The main weaknesses are unstable out-of-sample ranking, an extreme Up bias, poor reversal timing, and lower quality among additional ranks 16–20.

## Approaches that should not be repeated as-is

| Experiment | Best out-of-tuning result | Why it was rejected |
|---|---|---|
| Fixed group overlay, 12-month window, weight 0.25 | Validation group lift was only +0.0013; Confirmation lift was -0.0014 | No stable independent signal |
| Hierarchical empirical-Bayes group prior | Validation Directional AUC improved by about 0.0033 | Accuracy did not improve; Development fell by 0.11 percentage points |
| Cross-sectional logistic ranker | Tuning AUC reached 0.606 | Validation AUC fell to 0.488 and accuracy to about 48.4% |
| Selection-correctness calibration | Validation correctness AUC reached 0.4392 | It stayed below 0.50; the final corrected score fell to 0.3947 |
| Reliability-gated group weight | Validation Selection AUC reached 0.4026 | Accuracy fell by 0.89 percentage points and six hits |
| Group-residual shrinkage and reliability | Small gains appeared in individual windows | The gains did not persist and accuracy did not improve |
| Regime disagreement or reversal penalties | Some Selection AUC changes | Accuracy and Confirmation performance declined |
| Pairwise/pointwise rankers and lead/disagreement grids | Some grids approached or crossed Selection AUC 0.50 | Directional AUC or accuracy declined; extensive tuning increased overfitting risk |
| `p_down`-aware reranking and forced Down calls | More Down calls | Down precision was too low and net hits declined |
| Expanding from 15 toward 20 using breadth alone | More coverage | Marginal added ranks were close to random and reduced quality |

## Family F: recent-miss plus group stability

### Hypothesis

Family F attempted to improve the existing `p_up_selection_score` by:

1. Penalizing indicators with a high recent six-month miss rate.
2. Relaxing the miss threshold when more causal history was available.
3. Rewarding asset groups with a high and stable recent Up rate.

The intended adjustment was:

```text
adjusted_score = sigmoid(
    baseline_logit
    - 0.40 * recent_miss_penalty
    + 0.30 * group_stability_value
)
```

The monthly cap, Down logic, correlation graph, forward-breadth model, and regime-stress calculation were intended to remain unchanged.

### Initial implementation problem

The first implementation used this alignment:

```text
shift(1).rolling(...)
fit_through_origin = origin_position - 2
merge current origin == fit_through_origin
```

That merge was reversed. A score for forecast origin `t` could change when a label at `t+1` changed. A direct mutation test demonstrated the leak: changing a future label at origin 21 changed the recent-miss statistic used for origin 20 from `1.0` to `0.8333`.

Therefore, the initially reported Selection AUC values of approximately `0.5305` or `0.5894` were invalid and must not be used as evidence.

### Corrected causal rerun

The alignment was temporarily corrected so that each forecast origin `t` used labels through `t-2` only. The entire Regime Adaptive pipeline was rebuilt and compared with the reference baseline.

| Period | Directional AUC change | Selection AUC change | Accuracy change | Hit change |
|---|---:|---:|---:|---:|
| Tuning | -0.0138 | -0.0079 | -0.56 pp | -6 |
| Validation | +0.0051 | +0.0206, reaching 0.4222 | +0.30 pp | +2 |
| Development | -0.0048 | -0.0026 | -0.23 pp | -4 |
| Confirmation | -0.0084 | +0.0086, reaching 0.4497 | -0.50 pp | -4 |

### Decision

Family F was rejected and removed from production. Its two-hit Validation gain did not generalize, Selection AUC remained below 0.50, and accuracy declined in Tuning, Development, and Confirmation.

No part of Family F is active in the model configuration, pipeline, active artifact, or June–August forecast.

## Consumed March–May evaluation

The frozen baseline achieved 29/51 = 56.86%, with every call predicted Up:

| Month | Correct/Selected | Accuracy |
|---|---:|---:|
| March 2026 | 1/17 | 5.88% |
| April 2026 | 15/17 | 88.24% |
| May 2026 | 13/17 | 76.47% |

This result demonstrates reversal sensitivity but cannot be reused for threshold or feature selection.

## Flexible joint directional selection trial (2026-09-21)

A research-only selector ranked each indicator's strongest Tuning-calibrated
Up or Down call, allowed at most five Down calls, and made positions 16–20
optional. Its Tuning-selected policy took 15 calls in every Validation month.
It scored 357/600 versus 351/600 for the active model at matched coverage, but
445/709 versus 447/709 on Confirmation. At the active model's original monthly
caps, it scored 394/675 versus 395/675 on Validation and 505/799 versus
508/799 on Confirmation. It did not pass the non-locked screen and was not
promoted. See `research/flexible_directional_selection/README.md` and
`docs/EXPERIMENT_REGISTRY.md` for its reproducible record.

A follow-up trained directly on Up and Down call correctness while preserving
the same 15–20 total-call schedule. Its screen selected just one Down call,
but the final refit selected 133 Down calls in Validation, with 61 correct.
Total hits fell to 373/675 from 395/675 on Validation and 485/799 from
508/799 on Confirmation. Validation correctness AUC was 0.5047. The trained
mix is rejected; details are in `research/trained_directional_mix/README.md`.

A bounded weight search then compared calibrated Up and Down scores with
opposite-direction evidence and indicator-history priors. None of its 150
Tuning-screened candidates selected Down on the screen. The chosen weights
selected two Down calls in Validation (one correct), tied the active model at
395/675, and fell to 505/799 versus 508/799 on Confirmation. Its selected-score
correctness AUC fell from 0.5392 on screen to 0.4123 on Validation. The weights
are rejected as a confidence measure. See
`research/weighted_directional_mix/README.md`.

## Chronos-2 zero-shot trial (2026-09-22)

A pre-trained time-series foundation model (Chronos-2) forecast each indicator's
differenced series two steps ahead. Its calibrated Up-probability was blended with
the active model at a Tuning-selected weight of 0.75. At matched monthly coverage it
lost 15 hits on Tuning, 16 on Validation, and 12 on Confirmation, failing three of
five gate criteria. Better probability calibration on Validation (ECE 0.031 versus
0.059) did not produce better selected calls. It was not promoted. See
`docs/experiments/chronos2_zero_shot.md` and registry section 2.9.

## Calendar-seasonal prior trial (2026-09-23)

A pre-registered test of whether an indicator's Up-frequency in the target's calendar
month adds information beyond its trailing Up-rate. The existing `direction_lag_12`
feature is two months off season, so this had never been tested. Existence tests on
Tuning-era labels found nothing: split-half reliability between even and odd years was
r = 0.006. The primary overlay lost 9, 4, and 2 hits on Tuning, Validation, and
Confirmation at production's call count. Its control arm showed that production's
model stack beats its own prior by 13 hits on both Tuning and Confirmation, so the
stack carries real signal. A window-ensemble arm gained 11 on Validation but lost 20
on Tuning and 6 on Confirmation. See `research/seasonal_prior/README.md` and registry
section 2.10.

## Rules for future experiments

1. Test one bounded hypothesis per cycle, preferably with no more than three tunable parameters.
2. Select features and parameters only through origin 219 with genuine walk-forward evaluation.
3. Add a mutation test proving that changing labels at `t-1`, `t`, or `t+1` cannot affect a feature declared fit through `t-2`.
4. Compare candidates with the same monthly cap and the same 15–20 selection contract.
5. Promote only when Selection AUC is above 0.50 and improves by at least 0.02, Directional AUC does not decline by more than 0.002, accuracy does not decline, and Brier score does not materially worsen.
6. Treat Confirmation 220–266 as descriptive because it has been viewed repeatedly. A new future holdout is required for a final claim.
7. Do not reuse March–May 2026 for model or threshold selection.
8. Add every failed experiment to this registry before starting another one.

## Remaining research signals

`lead_negative_share` and recent causal miss history can change the ranking, but neither has improved Directional AUC and accuracy consistently. They are research observations, not production-ready solutions.
