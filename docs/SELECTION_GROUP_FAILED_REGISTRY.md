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
