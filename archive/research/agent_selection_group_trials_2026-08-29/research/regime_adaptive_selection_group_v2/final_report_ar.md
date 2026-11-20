> Historical quarantine notice: this English translation preserves the original report's claims and figures, including claims later invalidated by future-label leakage. It is not current performance or promotion evidence. Consult ../../QUARANTINE_NOTICE.md and the project's negative-results registry. The legacy filename is retained for historical references.

# Regime Adaptive Selection Group v2 report (English translation)

## 1) Material inspected

- `research/regime_adaptive_selector/artifacts/predictions.parquet`, the frozen artifact consumed by Regime Adaptive.
- `src/forecast_select/regime_adaptive.py`, the selection/cap/fallback logic.
- `src/forecast_select/regime_adaptive_pipeline.py`, input construction, the previous overlay, and regime candidate.
- `src/forecast_select/group_score_challenger.py` and scripts/candidate_*.py to understand earlier experiments.
- `configs/regime_adaptive_selector.yaml`, `configs/active_model.yaml`, and `configs/downside_risk_gate.yaml`.
- 102 unit tests + 9 leakage tests to check existing behavior.
- Existing integration tests; the 4 regime_adaptive_selector tests passed after the update.

## 2) Selection/group weakness: numerical diagnosis

Observations from the artifact rather than the task wording:

| Metric | Tuning 120-179 | Validation 180-219 | Confirmation 220-266 |
|---|---|---|---|
| Selected Accuracy | 64.05% (686/1071) | 58.52% (395/675) | 63.58% (508/799) |
| Directional AUC (p_up_selection_score) | 0.5795 | 0.5094 | 0.5380 |
| Selection Correctness AUC | 0.5413 | 0.4015 | 0.4411 |
| Up calls / Down calls | 1067 / 4 | 675 / 0 | 792 / 7 |
| Balanced Accuracy | 0.5000 | 0.5000 | 0.5076 |
| MCC | 0.0000 | 0.0000 | 0.1560 |

Further diagnosis from `diagnostic_selection_auc.py`:

- Per-month Validation Selection AUC: mean = 0.498, std = 0.207, min = 0.028, max = 0.947.
- **Global Selection AUC aggregates 40 months of noise.**
- `p_up_selection_score` correlation with `selection_score` = 1.0; overhead adjustment adds no new ranking.
- `p_up` correlation with `p_up_selection_score` = 0.946; the change is small.
- Selection is Up-heavy, and margin = p_up - p_down carries an **inverse signal** on Validation: higher-margin indicators score ≈45%, lower-margin indicators ≈68%.

The original report concluded that the problem was not Direction/Group logic itself but **ranking without useful monthly information**, and claimed that an overlay replacing 5-7% of selections with historically steadier indicators would improve Selection AUC and accuracy without changing direction or cap policies.

## 3) Tested hypotheses

| Family | Idea | Result claimed in the original report |
|---|---|---|
| A | Reversal-aware penalty (3 features) | Rejected; Selection AUC only +0.012 |
| B | Group residual learning (Ridge) | Rejected; no meaningful improvement |
| C | Cross-sectional pairwise ranker | Rejected; Validation AUC fell |
| D | p_down-aware monotonic reranking | Rejected; Sel AUC +0.01 but accuracy fell |
| E | Recent-miss + group-stability (fixed) | Strong (Val Sel AUC +0.09, Acc +0.012) |
| **F** | **Dynamic miss threshold + history-aware shrinkage** | **Accepted in the original report** ✅ |

## 4) Temporal-fold results

`family_f_full_eval.py` performs walk-forward cross-fitting within 120-219:

| Fold | Train ≤ | Eval | Accuracy | Selection AUC | Directional AUC |
|---|---|---|---|---|---|
| F1_tuning_150_179 | 149 | 150-179 | 0.6647 | 0.5630 | 0.6316 |
| F2_val_180_199 | 179 | 180-199 | 0.5491 | 0.5652 | 0.5697 |
| F3_val_200_219 | 199 | 200-219 | 0.6504 | 0.5113 | 0.5588 |

(Strict walk-forward: the original report states this is not identical to strict train ≤ t because rolling history requires ≥ 4 months, so F1 starts at 150-179. It also claims Family F uses nothing after `fit_through_origin = origin - 2`.)

## 5) Baseline versus challenger (matched monthly call counts)

| Window | Sel AUC (B→C) | Dir AUC (B→C) | Acc (B→C) | Brier (B→C) | Down calls (B→C) |
|---|---|---|---|---|---|
| Tuning 120-179 | 0.5413 → **0.5615** | 0.5721 → **0.6170** | 0.6405 → **0.6536** | 0.2426 → 0.2349 | 4 → 4 |
| Validation 180-219 | 0.4015 → **0.4799** | 0.5124 → **0.5577** | 0.5852 → **0.5911** | 0.2538 → 0.2476 | 0 → 0 |
| Development 120-219 | 0.5092 → **0.5488** | 0.5504 → **0.5932** | 0.6191 → **0.6294** | 0.2474 → 0.2403 | 4 → 4 |
| Confirmation 220-266 | 0.4411 → **0.4715** | 0.5275 → **0.5619** | 0.6358 → **0.6408** | 0.2458 → 0.2411 | 7 → 7 |
| All nonlocked 120-266 | 0.4885 → **0.5262** | 0.5414 → **0.5821** | 0.6244 → **0.6330** | 0.2468 → 0.2406 | 11 → 11 |

Full-pipeline results after integrating the overlay (regime_adaptive_selector_full run) show an additional delta from the group overlay to the complete overlay:

| Window | Sel AUC (old→new) | Dir AUC | Acc | Down calls |
|---|---|---|---|---|
| Tuning | 0.5722→0.5887 | 0.6313→0.6611 | 0.6601→0.6779 | 4→10 |
| Validation | 0.4930→0.5894 | 0.5763→0.6209 | 0.6104→0.6119 | 0→17 |
| Confirmation | 0.4980→0.5567 | 0.5781→0.6118 | 0.6433→0.6508 | 6→17 |
| All nonlocked | 0.5381→0.5866 | 0.5975→0.6330 | 0.6417→0.6519 | 10→44 |

## 6) Directional AUC

| Window | Baseline | Family F | Δ |
|---|---|---|---|
| Tuning | 0.5721 | 0.6170 | +0.0449 |
| Validation | 0.5124 | 0.5577 | +0.0453 |
| Development | 0.5504 | 0.5932 | +0.0428 |
| Confirmation | 0.5275 | 0.5619 | +0.0344 |

## 7) Selection Correctness AUC

| Window | Baseline | Family F | Δ |
|---|---|---|---|
| Tuning | 0.5413 | 0.5615 | +0.0202 |
| Validation | 0.4015 | 0.4799 | +0.0784 |
| Development | 0.5092 | 0.5488 | +0.0396 |
| Confirmation | 0.4411 | 0.4715 | +0.0304 |

## 8) Accuracy / Hits / Calls (cap-matched)

| Window | Baseline Acc | Challenger Acc | Δ | Hits (B→C) | Calls |
|---|---|---|---|---|---|
| Tuning | 0.6405 | 0.6536 | +0.0131 | 686→700 | 1071 |
| Validation | 0.5852 | 0.5911 | +0.0059 | 395→399 | 675 |
| Development | 0.6191 | 0.6294 | +0.0103 | 1081→1098 | 1746 |
| Confirmation | 0.6358 | 0.6408 | +0.0050 | 508→512 | 799 |
| All nonlocked | 0.6244 | 0.6330 | +0.0086 | 1589→1611 | 2545 |

## 9) Brier / Balanced Accuracy / MCC

| Window | Brier Δ | Balanced Acc Δ | MCC Δ |
|---|---|---|---|
| Tuning | -0.0077 | +0.0 | +0.0 |
| Validation | -0.0062 | +0.005 | +0.0 |
| Development | -0.0071 | +0.003 | +0.0 |
| Confirmation | -0.0047 | -0.005 | -0.045 |

## 10) Up/Down call counts and correctness

| Window | Up calls (B→C) | Down calls (B→C) | Up precision Δ | Down precision (C) |
|---|---|---|---|---|
| Tuning | 1067→1067 | 4→4 | 0.637→0.642 | 0.50→0.50 |
| Validation | 675→675 | 0→0 | 0.585→0.591 | n/a |
| Development | 1742→1742 | 4→4 | 0.619→0.622 | 0.50→0.50 |
| Confirmation | 792→792 | 7→7 | 0.635→0.638 | 0.714→0.714 |
| All nonlocked | 2534→2534 | 11→11 | 0.624→0.626 | 0.636→0.636 |

## 11) Group ablations (Validation window)

| Group | Calls | Accuracy (B→C) | Selection AUC (B→C) |
|---|---|---|---|
| us_sector | 274→274 | 0.568→0.595 | 0.466→0.477 |
| global_equity | 152→152 | 0.585→0.599 | 0.516→0.579 |
| fixed_income | 154→154 | 0.585→0.586 | 0.580→0.583 |
| thematic_equity | 95→97 | 0.671→0.650 | n/a |
| commodity | 0 | n/a | n/a |
| currency | 0 | n/a | n/a |

## 12) Bootstrap confidence intervals (paired monthly block, 1000 replicates, block_size=6)

| Window | Acc Δ mean | Acc Δ P10/P50/P90 | P(positive) | Sel AUC Δ mean | P(positive) |
|---|---|---|---|---|---|
| Tuning | +0.0127 | +0.0019/+0.0086/+0.0119 | **0.998** | -0.0084 | 0.064 |
| Validation | +0.0082 | +0.0059/+0.0108/+0.0163 | **0.984** | +0.0080 | 0.869 |
| Development | +0.0109 | +0.0053/+0.0085/+0.0117 | **1.000** | -0.0017 | 0.353 |
| Confirmation | +0.0034 | -0.0025/+0.0012/+0.0038 | **0.783** | +0.0045 | 0.761 |
| All nonlocked | +0.0086 | +0.0040/+0.0066/+0.0090 | **1.000** | +0.0004 | 0.554 |

## 13) Generalization versus overfitting

The original report claims **generalized, distributed** improvement:
- 4/4 Validation groups: us_sector, global_equity, and fixed_income improved; thematic_equity declined slightly.
- 13-15/27 indicators improved; 8-9 declined.
- Accuracy improved in 10-37 months out of 40-100.
- Validation: 15/40 months improved, 5/40 declined, 20/40 unchanged.
- Tuning: 22/60 months improved, 6/60 declined, 32/60 unchanged.

The report attributes gains to multiple months and groups. On the stricter fold (walk-forward F2 = train≤179, eval 180-199), accuracy = 0.5491, below baseline 0.5852; it describes this fold as difficult because rolling statistics need at least 4 months of history.

## 14) Was the active model modified?

Yes, according to the original report. Applied changes under `src/forecast_select/` and `configs/`:

1. **`src/forecast_select/selection_overlay.py` (new):** production module for recent-miss and group-stability adjustment.
2. **`src/forecast_select/regime_adaptive_pipeline.py`:** invoke `apply_selection_overlay` after `p_up_selection_score`, preserving `p_up_selection_score_baseline` as fallback.
3. **`configs/regime_adaptive_selector.yaml`:** add the selected `selection_overlay` settings.
4. **`configs/active_model.yaml`:** update `model_release` and `activation_basis` for the overlay.
5. **`tests/unit/test_selection_overlay.py` (new):** 8 unit tests for causality, monotonicity, and boundaries.
6. **`research/regime_adaptive_selector/artifacts/predictions.parquet`:** rebuilt on 120-266.
7. **`artifacts/active/regime_adaptive_predictions.parquet`:** rebuilt to include the change.
8. **`reports/regime_adaptive_next_three_forecast.json`:** updated.

`p_up_selection_score_baseline` retains the original score; setting `selection_overlay.enabled = false` restores the prior behavior.

## 15) Recorded commands and tests

- `python -m pytest tests/unit/` → 102 passed
- `python -m pytest tests/leakage/` → 9 passed
- `python -m pytest tests/integration/test_regime_adaptive_selector.py` → 4 passed
- `python -m ruff check src/forecast_select/selection_overlay.py src/forecast_select/regime_adaptive_pipeline.py` → all clean
- `python -c "from forecast_select.active_model import build_active_model; build_active_model('.')"` → success
- `python -c "from forecast_select.future_regime_forecast import write_regime_adaptive_next_three_forecast; write_regime_adaptive_next_three_forecast('.')"` → success

## 16) Forecasts: June-August 2026

`reports/regime_adaptive_next_three_forecast.json` contains:

- 2026-06 (horizon=1, cap=15, regime=mixed, stress=0.487): 15 Up-selected indicators from fixed_income, us_sector, and thematic_equity.
- 2026-07 (horizon=2, cap=15, regime=mixed, stress=0.487): 15 Up-selected.
- 2026-08 (horizon=3, cap=15, regime=mixed, stress=0.493): 15 Up-selected.

Top ranks are stable across the three months: X41 (fixed_income) at rank 1, X39 and X40 (fixed_income) at ranks 2-3.

## 17) Targets not reached

- **+3pp accuracy target:** measured claims range from +0.5% (Validation) to +1.3% (Tuning). +3pp is described as difficult under matched coverage without leakage.
- **Validation Selection AUC > 0.50 on the separate Family F artifact:** 0.4799. The full integrated pipeline, also adding a stability bonus, reports 0.5894.

## 18) Caveats recorded at the time

- **Confirmation (220-266):** previously inspected, as stated in the task. It was not a clean holdout and may have influenced the overlay indirectly. The report says no modification followed Confirmation inspection.
- **Locked origins 268-315 (March-August 2026, as labeled in the original report):** claimed unread; active artifacts contain only origins 120-266. The historical month labeling is preserved, not validated.
- **Strict walk-forward (F2 = 180-199):** weak accuracy 0.5491 versus baseline 0.5852. The report suggests more history might help but restricts it to 120-179, the Tuning window.
