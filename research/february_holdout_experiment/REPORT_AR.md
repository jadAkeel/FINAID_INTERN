# Detailed report: Regime Adaptive improvement experiment

Report date: 29 August 2026
Decision: **No generalizable improvement was found among the tested candidates.** The `Regime Adaptive Bidirectional Selector` remains active; no challenger was promoted.

This does not establish that improvement is impossible. The tested methods did not demonstrate an improvement outside tuning, and this experiment supplies no reliable evidence of a gain of 3 percentage points or more.

## 1. Evaluation protocol

- Tuning: origins 120–179, used to select candidate settings.
- Validation: origins 180–219, used to assess generalization.
- Development: origins 120–219, a summary of the preceding windows rather than another independent test.
- Confirmation: origins 220–266, an additional descriptive window.
- Terminal holdout: March, April, and May 2026, forecast together from the frozen February origin at position 313.

The final test generated all horizons together: H1 for March, H2 for April, and H3 for May. Actual March outcomes did not enter April forecasts, and actual March/April outcomes did not enter May forecasts. The group prior used available data through `origin - 2 - (horizon - 1)`, namely positions 311, 310, and 309 respectively.

After inspecting March–May outcomes, this holdout is consumed. It must not be reused to select thresholds or features and then described as an independent final test.

## 2. Active-model diagnosis

| Window | Calls | Hits | Accuracy | Directional AUC | Selection AUC | Up calls | Down calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tuning 120–179 | 1,071 | 686 | 64.05% | 0.5795 | 0.5413 | 1,067 | 4 |
| Validation 180–219 | 675 | 395 | 58.52% | 0.5094 | 0.4015 | 675 | 0 |
| Development 120–219 | 1,746 | 1,081 | 61.91% | 0.5533 | 0.5092 | 1,742 | 4 |
| Confirmation 220–266 | 799 | 508 | 63.58% | 0.5380 | 0.4411 | 792 | 7 |

Interpretation:

1. Directional AUC fell from 0.5795 on Tuning to 0.5094 on Validation: ranking rising indicators above falling ones became nearly random outside tuning.
2. Selection AUC fell from 0.5413 to 0.4015. The `selection_score` failed to rank correct selections above mistakes on Validation and was partly reversed.
3. Accuracy near 60% is consistent with weak AUC because most calls were Up. A model favoring the common direction can obtain acceptable accuracy without strongly discriminating indicators.
4. The Down sample is inadequate: 11 calls across 120–266, with none on Validation. It does not establish statistically persuasive Down forecasting quality.

## 3. Rank quality and the 15–20 cap

On Development 120–219:

| Rank | Accuracy | Cases |
|---|---:|---:|
| 1–5 | 64.00% | 500 |
| 6–10 | 62.40% | 500 |
| 11–15 | 60.60% | 500 |
| 16–20 | 59.35% | 246 |

Quality decreases as coverage expands, particularly comparing the first five ranks with 16–20. The difference is not stable across windows; Validation ranks were nearly equal. Dropping ranks 16–20 alone therefore does not solve the problem, and the project requires 15–20 indicators.

Development call counts were: 330 at cap=15, 192 at 16, 221 at 17, 342 at 18, 361 at 19, and 300 at 20. The 15–20 contract is enforced.

## 4. Did asset groups add value?

| Window | Raw `p_up` | After graph | After current group overlay | Group lift over graph |
|---|---:|---:|---:|---:|
| Tuning | 0.5684 | 0.5699 | 0.5795 | +0.0096 |
| Validation | 0.5127 | 0.5081 | 0.5094 | +0.0013 |
| Development | 0.5481 | 0.5476 | 0.5533 | +0.0057 |
| Confirmation | 0.5371 | 0.5394 | 0.5380 | -0.0014 |

The group overlay helped within Tuning, but its lift shrank to 0.0013 on Validation and reversed on Confirmation. This does not establish generalizable value.

The implementation uses a raw rolling 12-month prior with fixed weight 0.25 across groups, regardless of their size. Group strength also drifts: commodity effects ranged approximately from -1.07 to +0.56, and thematic effects from -0.47 to +0.32. Signals overlap substantially: `p_up` correlates 0.946 with its group-adjusted version, and `p_up_selection` correlates 0.924 with `risk_adjusted`.

## 5. Tested candidates

### A. Hierarchical Empirical Bayes group overlay

Replace the raw group prior with shrinkage toward the market:

`p_shrunk = (N × p_group + tau × p_market) / (N + tau)`

The best Tuning-only setting was `tau=10`, `weight=0.75`, without a stability penalty.

| Metric | Current | Challenger A | Difference |
|---|---:|---:|---:|
| Tuning AUC | 0.5795 | 0.5829 | +0.0034 |
| Validation AUC | 0.5094 | 0.5127 | +0.0033 |
| Tuning selected accuracy | 64.05% | 63.68% | -0.37 points |
| Development selected accuracy | 61.91% | 61.80% | -0.11 points |

Decision: rejected. AUC improvement was small; Development accuracy did not improve and Tuning accuracy declined. Removing the graph also produced a slightly higher Validation AUC of 0.5157, weakening the case for stable independent value from combining graph and group.

### B. Cross-sectional regularized ranker

A regularized Logistic Regression model used 19 existing Regime Adaptive signals.

- Tuning AUC: 0.6061 versus baseline 0.5795.
- Validation AUC: 0.4882 versus 0.5094.
- Validation accuracy: approximately 48.4% versus 58.5%.

Decision: rejected for clear overfitting; the Tuning gain became an out-of-tuning loss.

### C. Selection-quality correction

A model was trained to predict selection correctness and adjust `selection_score`.

- Baseline Selection AUC: 0.5413 on Tuning and 0.4015 on Validation.
- Correction model: 0.5783 in training and 0.4392 on Validation.
- Incorporating the correction into the score reduced Validation AUC to 0.3947.

Decision: rejected. Moving from 0.4015 to 0.4392 still leaves the result below 0.5, and the final score became worse.

### D. Causal reliability-gated group score

Regime Adaptive was retained while the fixed group weight was experimentally replaced with a causal rolling estimate of whether the group signal had added value beyond `p_up` in previously available months. Windows of 12/24/48/72 months, ridge settings, and weight bounds were searched on Tuning only.

The best setting failed the gates. Validation Selection AUC moved marginally from 0.4015 to 0.4026, while Directional AUC fell from 0.5094 to 0.5084 and accuracy from 58.52% (395/675) to 57.63% (389/675). It remained isolated from production. Details are recorded in [`candidate_d_report.json`](candidate_d_report.json).

## 6. Frozen final test: March–May 2026

Each model selected 17 indicators per month, all Up.

| Month | Baseline | Challenger A | Hit difference | Baseline Brier | Challenger Brier |
|---|---:|---:|---:|---:|---:|
| March 2026 | 1/17 = 5.88% | 1/17 = 5.88% | 0 | 0.3565 | 0.3850 |
| April 2026 | 15/17 = 88.24% | 16/17 = 94.12% | +1 | 0.1538 | 0.1325 |
| May 2026 | 13/17 = 76.47% | 12/17 = 70.59% | -1 | 0.1839 | 0.1892 |
| **Total** | **29/51 = 56.86%** | **29/51 = 56.86%** | **0** | — | — |

The challenger gained one April hit and lost one in May, tying overall. In March, only 5 of 50 indicators rose, while `forecast_market_breadth` was about 0.57 and the regime was classified calm with `regime_stress` near 0.30. No Down call fired: the model selected 17 Up calls and hit only one.

Monthly AUC within 17 selections is highly fragile. With all predictions Up and very few hits or mistakes, AUC can look high or unstable without useful directional ability. The firmer evidence is the tied hit count, absence of Down calls, and failure to anticipate the March reversal.

## 7. Promotion decision

| Acceptance gate | Result |
|---|---|
| Respect 15–20 selections | Passed; 17 each month |
| No holdout outcomes in training | Passed according to the implementation contract and report |
| Generalizable Directional AUC improvement | Failed |
| Stable Selection AUC above 0.5 | Failed |
| No accuracy deterioration | Final tie, with a slight Development loss |
| Independent group value established | Failed |
| Meaningful Down sample | Failed; no Down calls in the final test |

Reject the candidates and retain Regime Adaptive unchanged. The active configuration records `research_gate_passed: false` and owner-directed activation for bidirectionality; it must not be presented as demonstrated research superiority.

## 8. What this experiment establishes

The experiment supports:

- Methods A, B, and C did not produce generalizable improvement.
- No evidence here supports a stable 3–4-percentage-point gain.
- The main issue extends beyond the group formula to detecting sharp reversals and issuing reliable Down calls.
- Retuning on inspected March–May outcomes creates model-selection leakage even if those values do not enter training rows directly.

It does not establish:

- That future improvement is impossible.
- That 56.86% is the true expected performance; the terminal sample is only three months and 51 volatile calls.
- Down-call quality, because no Down call was issued on the holdout.

## 9. Methodologically permitted next study

A bounded next hypothesis is a small causal model of reversal or breadth-collapse risk, rather than another general ranker. A target such as `future breadth < 0.35` could use information through `t-2` only and serve as a guarded Down/abstention gate.

Choose all features and thresholds on 120–219 only, with walk-forward evaluation, block bootstrap, and paired hit deltas. Freeze the design before opening a new temporal holdout. Do not reuse March–May as an acceptance test or treat cap expansion as improvement without establishing marginal-rank quality.

## 10. Sources and verification

- Diagnosis: [`development_diagnosis.json`](development_diagnosis.json)
- Development and candidate report: [`development_report.json`](development_report.json)
- Experiment and decision manifest: [`run_manifest.json`](run_manifest.json)
- Detailed holdout results: [`march_may_holdout_report.json`](march_may_holdout_report.json)
- Hit comparison: [`model_comparison.csv`](model_comparison.csv)
- Group ablation: [`group_ablation_candidateA.csv`](group_ablation_candidateA.csv)
- Active settings: [`../../configs/active_model.yaml`](../../configs/active_model.yaml)
- Regime Adaptive settings: [`../../configs/regime_adaptive_selector.yaml`](../../configs/regime_adaptive_selector.yaml)

Related tests were rerun on 29 August 2026: `19 passed`. `ruff check` on `src/forecast_select` also passed. A future-configuration warning from `pytest-asyncio` did not affect the test result.
