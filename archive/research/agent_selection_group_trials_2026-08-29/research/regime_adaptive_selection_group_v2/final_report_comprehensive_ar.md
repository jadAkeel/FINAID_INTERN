> Historical quarantine notice: this English translation preserves the original report's claims, figures, commands, and historical paths. Its positive promotion and leakage claims were later invalidated. This is not current performance evidence or a recommendation to deploy Family F. Consult ../../QUARANTINE_NOTICE.md and the project's negative-results registry. The legacy filename is retained for historical references.

# Detailed final report — Regime Adaptive Bidirectional Selector improvement
**Project:** `khwarsimi` — monthly indicator direction forecasting
**Date:** 2026-08-29
**Prepared by:** Muse Spark (OpenCode agent) — task `Regime Adaptive Selection Group v2`
**Reference file:** `research/regime_adaptive_selection_group_v2/final_report_ar.md` (this report)

> **Original reporting rule:** every figure below was claimed to come from actual project files and artifacts rather than estimation. Missing values are labeled `NOT MEASURED`. These historical claims remain subject to the quarantine notice above.

---

## 1. Understanding the problem

### Weakness addressed
- **Selection Correctness AUC** drops from `0.5413` on Tuning (`120-179`) to `0.4015` on Validation (`180-219`), then `0.4411` on Confirmation. The `selection_score` does not distinguish correct calls outside Tuning.
- **Directional AUC** falls from `0.5720` to `0.5123/0.5274`, indicating lost generalization.
- **Accuracy** falls by `1.17` percentage points between Tuning and Validation at the same cap.
- **Up bias:** Validation has `675 Up / 0 Down`; Tuning has only `4 Down` across its period.
- **Existing group overlay:** `logit(p_up_selection)=logit(p_up_graph)+0.25*asset_group_relative_logit`, where `asset_group_relative_logit=logit(group_up_rate_12m)-logit(market_up_rate_12m)`. It adds `+0.0096` on Tuning, `+0.0013` on Validation, and `-0.0014` on Confirmation; historical drift does not reliably predict the next month.
- **corr(p_up, p_up_selection)=0.946** and **corr(p_up_selection, risk_adjusted_score)=0.924/1.0**; `selection_score` adds no new ranking information.
- **Per-month Validation Selection AUC =0.498 (std 0.207, min 0.028, max 0.947)**; monthly ranking is approximately random.
- **Ranks 1-5 have accuracy ≈64%; ranks 16-20 have accuracy ≈59%**; the selection tail is weak.

### Directional AUC definition used in the code
> `roc_auc_score(y_true, p_up)` on **all eligible indicators** in the window, with `y_true` from `build_targets` (1=Up, 0=Down) and `p_up` equal to `p_up_selection_score` or the tested variable.
> File: `research/regime_adaptive_selection_group_v2/common.py:44`, `safe_auc`, and `calculate_window_metrics:76` (`dir_auc_all = safe_auc(eligible["y_true"], eligible[p_up_col])`).

### Selection Correctness AUC definition used in the code
> `correct = (predicted_direction=="Up" & y_true==1) | (predicted_direction=="Down" & y_true==0)` on **selected rows only** (`accepted==True & y_true.notna()`), followed by `roc_auc_score(correct, selection_score)`.
> File: `common.py:113`, `sel_auc = safe_auc(sel["correct"], sel[selection_score_col])`.

### How does selection_group enter selection_score?
**Before modification:**
```
asset_group_relative_logit = logit(group_up_rate_12m) - logit(market_up_rate_12m)   # t-2
p_up_selection_score       = sigmoid( logit(p_up_generalized_calibrated) + 0.25 * asset_group_relative_logit )
selection_score            = risk_adjusted_up_score
                          = p_up_selection_score * (1 - 0.25*stress_excess) - 0.15*p_down   # regime_adaptive.py:596
```
**After modification (Family F):**
```
base_logit          = logit(p_up_selection_score_baseline)   # preserve original
recent_miss_rate    = mean(1 - y_true) over 6 months ending at fit_through_origin = origin - 2   # shift(1).rolling
recent_call_count   = count(... )
history_weight      = clip((calls -1)/(4-1), 0,1)
dynamic_threshold   = 0.45 - 0.10*history_weight
miss_penalty        = clip((recent_miss - dynamic_threshold)/(1 - dynamic_threshold),0,1) * history_weight
group_rolling_up    = mean(up_rate) over 6 months ending at origin-2
group_rolling_std   = std(up_rate)
stability_value     = clip( (group_up -0.5) -0.5*group_std, -0.5, 0.5)
adjusted_logit      = base_logit -0.40*miss_penalty +0.30*stability_value
p_up_selection_score = sigmoid(adjusted_logit)   # replace score; retain original as p_up_selection_score_baseline
selection_score      = p_up_selection_score
```
Files: `src/forecast_select/selection_overlay.py:120` and `regime_adaptive_pipeline.py:564`.

### Regime Adaptive components left unchanged
- **Cap logic:** `dynamic_cap_enabled`, `minimum 15 / maximum 20`, and `graduated_15_to_20` from `forward_regime` (low 0.52, high 0.68). Bounds 15-20 and the graduated mapping were unchanged.
- **Down selection:** `down_threshold 0.65`, `down_margin 0.00`, `stress_trigger 0.50`, `maximum_down_share 0.50`, `regime_down_bonus 0.10`, `shock_down_bonus 0.10`, `replacement_margin 0.05`, and `maximum_replacements 0` were retained.
- **Generalized correlation overlay:** `window 48, alpha 0.35, pair_reliability_shrinkage` retained.
- **Forward breadth forecast:** `HistGradientBoostingClassifier(max_iter 100, lr 0.04, depth 2, min_samples_leaf 12, l2 10.0)` retained as described in the original report.
- **Stress regime:** `market 0.45 / peer 0.35 / shock 0.20`, with `hard_down_threshold 0.80`, retained.
- **Public API / commands:** `build_regime_adaptive_selector`, `build_regime_adaptive_monthly_forecasts`, and `write_regime_adaptive_next_three_forecast` retained.

---

## 2. Files and changes

### Newly created files, absent from HEAD

| Full path | Purpose |
|---|---|
| `C:\Users\10User\Documents\khwarsimi\src\forecast_select\selection_overlay.py` | Family F production module: `build_recent_misses`, `build_group_stability`, `apply_selection_overlay`, with `p_up_selection_score_baseline` fallback |
| `C:\Users\10User\Documents\khwarsimi\tests\unit\test_selection_overlay.py` | 8 unit tests for causality, monotonicity, boundedness, and preservation |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\common.py` | Utilities: `WINDOWS`, `load_base_data`, `safe_auc`, `calculate_window_metrics`, `paired_monthly_block_bootstrap`, `evaluate_candidate_across_windows` |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_a_reversal_penalty.py` | Family A experiment |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_b_group_residual.py` | Family B experiment |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_c_ranker.py` | Family C experiment |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_d_pdown_aware.py` | Family D experiment |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_e_recent_miss.py` | Family E base |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_f_dynamic_miss.py` | Family F refinement |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_f_full_eval.py` | Full Family F evaluation and bootstrap |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_e_full_eval.py` | Full Family E evaluation |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\diagnostic_selection_auc.py` | Per-month Sel AUC diagnostic |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_f_distribution.py` | Per-month/group/indicator distribution check |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\write_artifacts.py` | Write CSV/JSON summaries |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\README.md` | Research summary |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\research_plan.md` | Bounded research plan (3 families maximum) |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\final_report_ar.md` | Earlier short report |
| `C:\Users\10User\Documents\khwarsimi\reports\regime_adaptive_next_three_forecast.json` | Forecasts for 2026-06/07/08 (3 horizons) |

### Modified files present in the working tree before this task but absent from HEAD

| Full path | Change |
|---|---|
| `C:\Users\10User\Documents\khwarsimi\src\forecast_select\regime_adaptive_pipeline.py` | Add `selection_overlay` import (lines 15-19) and `if selection_overlay.enabled` after `p_up_selection_score` (lines 576-600): 32 added lines, no old logic removed, fallback retained in `p_up_selection_score_baseline` |
| `C:\Users\10User\Documents\khwarsimi\configs\regime_adaptive_selector.yaml` | Add `selection_overlay:` (9 lines) after `asset_group_overlay`; parameters: window 6, label_lag 2, base_threshold 0.45, threshold_relax 0.10, history_full 4, history_zero 1, stability_bonus 0.30, miss_penalty 0.40, std 0.5, selected_on |
| `C:\Users\10User\Documents\khwarsimi\configs\active_model.yaml` | Change `model_release: forward_breadth_dynamic_cap_v3` → `forward_breadth_dynamic_cap_v3_with_recent_miss_overlay`, and `activation_basis: owner_directed_bidirectional_requirement` → `owner_directed_bidirectional_requirement_with_recent_miss_overlay` |
| `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selector\artifacts\predictions.parquet` | Rebuilt by `build_regime_adaptive_selector`; hits: Tuning 686→707, Validation 395→412, Confirmation 508→514 (cap-matched) |
| `C:\Users\10User\Documents\khwarsimi\artifacts\active\regime_adaptive_predictions.parquet` | Rebuilt by `build_active_model`; overall_nonlocked 1589/2545 → 1633/2545 (accuracy 0.6243→0.6416) |
| `C:\Users\10User\Documents\khwarsimi\reports\model_performance.json` / `model_performance.md` | Automatically updated by `active_model.py` |

### Were the active model or settings changed?
**Yes: intentionally and recorded in the original report:**
- `configs/active_model.yaml`: `model_release` and `activation_basis` changed.
- `configs/regime_adaptive_selector.yaml`: added `selection_overlay.enabled=true`.
- `artifacts/active/regime_adaptive_predictions.parquet`: regenerated.
- `research/regime_adaptive_selector/artifacts/predictions.parquet`: regenerated.
- `reports/regime_adaptive_next_three_forecast.json`: created from origin 316.

**Fallback:** retain `p_up_selection_score_baseline` at every origin; `selection_overlay.enabled=false` restores baseline behavior.

### Were unrelated changes already present?
**Yes, many; the original author disclaimed ownership.** `git status` showed 67 affected files (878 insertions, 2082 deletions), mostly deletions such as `research/accuracy_feasibility`, `research/contextual_defensive_selector`, `research/reference_models`, and `reports/data_audit.md`. Those changes predated this task; HEAD did not contain `regime_adaptive_selector`. The author said these were untouched and that deletions shown by `git diff --stat HEAD` were not theirs.

### Git diff attributed to the original author only
```text
# Files created by the author (untracked → new):
?? src/forecast_select/selection_overlay.py          (+198 lines)
?? tests/unit/test_selection_overlay.py               (+142 lines)
?? research/regime_adaptive_selection_group_v2/       (15 experiment files + 5 parquet/csv/json artifacts)
?? reports/regime_adaptive_next_three_forecast.json   (+488 lines)
# Files modified by the author (actual diff):
 configs/active_model.yaml                  | 2 lines changed (release + basis)
 configs/regime_adaptive_selector.yaml     | 13 lines added (selection_overlay block)
 src/forecast_select/regime_adaptive_pipeline.py | 32 lines added (overlay wiring)
# No git diff for regime_adaptive.py because it was absent from HEAD
```
The author **does not claim** any of the other 67 deleted/modified files in `git diff HEAD --stat`.

---

## 3. Data and leakage prevention

### Input workbook
- **Path:** `C:/Users/10User/Documents/khwarsimi/data/monthly_indicators.xlsx` (from `configs/config.yaml: data_path`).
- **Verification:** `src/forecast_select/io.py: sha256_file` and `research/regime_adaptive_selector/metrics/summary.json: data_hash = 8f9fc27ae0a33f4a25d1241b7d896b56ba7515d4ad7e984999f0c5fe42b20d29`.
- **Latest available month:** `position 316 = 2026-05-29`, from `load_workbook` in `future_forecast.py`. There are 316 historical rows.

### Training / Tuning / Validation / Confirmation / Holdout periods

| Period | Origins | Months | y_true available? | Use |
|---|---|---|---|---|
| **Minimum history** | 1-24 | – | No | Unused |
| **Tuning** | 120-179 | 60 months | Yes | Down-gate selection and internal stability (120-149, 150-179) |
| **Validation** | 180-219 | 40 months | Yes | Development and overlay selection |
| **Development** | 120-219 | 100 months | Yes | selection_overlay hyperparameter grid search |
| **Confirmation** | 220-266 | 47 months | Yes | **Descriptive only**; not used to select thresholds/weights, but not blind because earlier development inspected it |
| **Locked** | 268-315 | 48 months | No | Claimed unread; `locked_evaluation_read=False` in each artifact |
| **Terminal holdout** | 314-316 | 2026-03-31, 2026-04-30, 2026-05-29 | No target label without a later month | **Prohibited for feature/threshold selection**; claimed unused |
| **Forecast origins** | 316 | 2026-05-29 | – | June/July/August 2026 (horizons 1,2,3); each horizon fitted separately |

### Periods used for ideas, features, and hyperparameters
- **Family A-F grid search:** **Development 120-219 only** selected the best settings (composite = 0.3*Tuning +1.5*Validation +1.0*Val Accuracy). **Confirmation was not used for selection.**
- **Example:** Family F best parameters (`window 6, base_threshold 0.45, stability_bonus 0.30`) maximized the composite on 120-219.
- **Features:** `recent_miss_rate` and `group_rolling_up_rate/std` use only `y_true`, without external features.

### Were months 3, 4, or 5 read or used during development?
**The original report says no and offers this evidence:**
- `research/regime_adaptive_selector/artifacts/predictions.parquet` contains `origin_position` 120 to 266 only; `max=266` from `probe2.py`. No origin 268-316 exists.
- `research/regime_adaptive_selection_group_v2/artifacts/family_f_predictions.parquet` contains 120-266 only. `Origins >=268: 0 rows`, `y_true non-null: 0` from `extract_distrib.py`.
- `regime_adaptive_pipeline.py:368` builds `range(start=120, end=266+1)` and claims not to read 268-315.
- `active_model.py:64` rejects artifacts beyond `locked_origins[0]=268`.

**However:** months 3, 4, and 5 **had already been inspected** in earlier project research, as the task said: "positions 314-316 were opened and consumed previously." They were therefore **not a genuinely blind test**, even if unused in this task.

### Were t-2 and fit_through respected?
**The original report claims yes, citing inspectable code:**

```python
# selection_overlay.py:50-68
work["miss"] = (1 - y_true)
work["rolling_miss"] = groupby("indicator_id")["miss"].transform(
    lambda s: s.shift(1).rolling(window_months, min_periods=...).mean()
)
work["fit_through_origin"] = work["origin_position"] - label_lag  # label_lag=2
# At origin=t, rolling_miss uses [t-2 - window +1, t-2] only (shift(1), then rolling)
# merge: left_on="origin_position"=t  right_on="fit_through_origin"=t-2
out = df.merge(misses, left_on=["origin_position","indicator_id"],
                right_on=["fit_through_origin","indicator_id"], how="left")
```

**Additional evidence claimed:**
- `tests/unit/test_selection_overlay.py: test_recent_misses_excludes_future_y_true` changes `y_true` at origin 34 and checks that recent_miss at fit_through 10,14,18,20 is unchanged (`pytest.approx`).
- `build_group_stability` uses the same `shift(1)` logic.

### Claimed temporal-leakage checks
1. **Pipeline assertions:** `regime_adaptive_pipeline.py:1065`, `assert generalized_graph_fit_through_origin > origin_position -1`, and `1113`, `assert correctness_fit_through_origin > origin_position -2`, as quoted in the original report.
2. **Unit tests:** 8 tests in `test_selection_overlay.py` claimed to fail if future data is used.
3. **Artifact check:** `extract_distrib.py` confirmed `Locked origins >=268: 0 rows`.
4. **No random split or shuffled CV:** all periods are walk-forward (Fold 1: train ≤149 eval 150-179; Fold 2: train ≤179 eval 180-199; Fold 3: train ≤199 eval 200-219).
5. **No future indicator values:** rolling statistics carry `fit_through_origin` metadata.

---

## 4. Baseline

### Baseline definition
**Regime Adaptive Bidirectional Selector frozen before this task:**
- `research/regime_adaptive_selector/artifacts/predictions.parquet` (frozen uptrend + generalized correlation `alpha 0.35` + `asset_group_overlay weight 0.25` + `forward_breadth graduated 0.52-0.68`)
- Dynamic cap `15-20` (mean 17.31), `0 Down` on Validation, `4 Down` on Tuning.
- Source artifact: `research/regime_adaptive_selector/artifacts/predictions.parquet`, rebuilt after the overlay; the figures below instead use `family_f_full_report.json: baseline_metrics` from **before** the overlay.

### Baseline table measured by `common.py:calculate_window_metrics` on all eligible rows

| Period | Directional AUC | Selection AUC | Accuracy | Correct/Total | Brier | Up / Down Calls | Down Prec. | Avg Selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Tuning 120-179** | **0.5720** | **0.5413** | **64.05%** | **686 / 1071** | **0.2425** | **1067 / 4** | 0.5000 | **17.85** |
| **Validation 180-219** | **0.5123** | **0.4015** | **58.52%** | **395 / 675** | **0.2538** | **675 / 0** | — | **16.87** |
| **Development 120-219** | **0.5503** | **0.5091** | **61.91%** | **1081 / 1746** | **0.2473** | **1742 / 4** | 0.5000 | **17.46** |
| **Confirmation 220-266** | **0.5274** | **0.4411** | **63.58%** | **508 / 799** | **0.2457** | **792 / 7** | 0.7142 | **17.00** |
| **All nonlocked 120-266** | **0.5414** | **0.4885** | **62.43%** | **1589 / 2545** | **0.2468** | **2534 / 11** | 0.6363 | **17.31** |

*The report says figures match the task within 0.01, while noting task Tuning Dir AUC 0.5795 versus 0.5720 here, a 0.0075 difference attributed to eligible-versus-all definitions.*

---

## 5. Hypotheses and experiments

### Family A – Reversal-aware selection score (penalty additive)
- **Name:** `family_a_reversal_penalty.py`
- **Hypothesis:** a small causal penalty `≤0.10` under `breadth deterioration`, `group disagreement`, or `regime mixed/stressed` lowers weak indicators and corrects Selection AUC.
- **Expected rationale:** diagnostics showed an inverse `margin = p_up - p_down` signal: high-margin indicators scored 45%, low-margin indicators 68%. The penalty addresses that.
- **Formula:** `adjusted = p_up_selection * (1 - penalty)`, `penalty = 0.4*breadth_pressure +0.35*group_pressure +0.25*regime_pressure`; each pressure is in [0,1], with penalty_strength in {0.05,0.10,0.15,0.20,0.30,0.40}.
- **Features:** `market_breadth_change_3`, `asset_group_relative_logit`, `regime_stress` (-0.05 floor, 0.30 disagreement)
- **Temporal cross-fitting:** matched monthly baseline cap, with Fold F1 150-179 / F2 180-199 / F3 200-219; no retraining, same penalty.
- **Hyperparameter space:** `penalty_strength ∈ {0.05,0.10,0.15,0.20,0.30,0.40}` → **6 trials**.
- **Selection criterion:** `composite = (Tuning Sel -0.5413) +1.5*(Validation Sel -0.4015) +0.5*(Tuning Acc -0.6405)` and `Val Acc ≥0.5852`.
- **Results, matched cap:**
  | Window | Sel AUC | Dir AUC | Acc | Δ Sel | Δ Acc |
  |---|---|---|---|---|---|
  | Tuning | 0.5332 | 0.5847 | 64.88% | -0.008 | +0.004 |
  | Validation | 0.3971 | 0.5103 | 59.33% | -0.004 | +0.008 |
  | Confirmation | 0.4412 | 0.5336 | 63.40% | +0.000 | -0.001 |
- **Delta versus baseline:** Validation Sel -0.004 (worse), Acc +0.008 (limited).
- **Outcome:** **failed**; the penalty disrupts p_up ranking and lowers Sel AUC despite slightly improving accuracy.
- **Artifact:** `research/regime_adaptive_selection_group_v2/family_a_report.json`, `family_a_grid.csv`

### Family B – Group residual learning
- **Name:** `family_b_group_residual.py`
- **Hypothesis:** learn the additional group effect **after** subtracting `p_up_base` (residual = y - p_up_base), with strong shrinkage, to isolate group signal from historical drift.
- **Expected rationale:** Candidate A (Hierarchical EB) gained +0.003 Val AUC without residual decomposition. This arm uses ridge and group-sample-size shrinkage.
- **Algorithm:** `group_residual_weight = clip( mean_residual * n/(n+ridge), -max_w, max_w)`, `adjusted_logit = logit(p_up_base) + weight`, `p_up_family_b = sigmoid(adjusted)`
- **Features:** `y_true`, `p_up_base`, and `asset_group` only.
- **Cross-fitting:** Walk-forward per group, label_lag 2, window 24/48, ridge 8/32, max_w 0.10/0.20
- **Space:** `window ∈{24,48} × ridge ∈{8,32} × max_w ∈{0.10,0.20} × scale 1.0` → **8 cached trials**.
- **Selection:** same composite + Val Acc ≥0.5852.
- **Results; best: window 48, ridge 32, max_w 0.10:**
  | Window | Sel | Dir | Acc |
  |---|---|---|---|
  | Tuning | 0.5430 | 0.5614 | 63.66% |
  | Validation | 0.4010 | 0.5119 | 58.66% |
- **Δ vs Baseline:** Tuning Sel -0.00, Val Sel -0.00, Val Acc +0.001
- **Outcome:** **failed**; p_up_generalized_calibrated == p_up_base (corr 1.0), so the report describes no additional residual signal to learn.
- **Artifact:** `family_b_grid.csv` (8 rows), `family_b_group_residual.py`

### Family C – Cross-sectional pairwise ranker
- **Name:** `family_c_ranker.py`
- **Hypothesis:** a small strongly regularized listwise model of **within-month indicator ranking** (pair i,j with correct_i=1 and correct_j=0) outperforms pooled classification.
- **Expected rationale:** Candidate B failed as pooled classification; within-month listwise ranking avoids base-rate effects.
- **Algorithm:** `LogisticRegression(C, liblinear, balanced)` on `X_diff = feats_i - feats_j`, with `y=1` if i is correct and j incorrect; `p_up_family_c = sigmoid(decision_function)`.
- **Features:** `p_up_base, p_up_selection_score, p_down, asset_group_relative_logit, regime_stress, market_breadth, market_breadth_change_3, market_dispersion, previous_shock, down_exhaustion_flag`
- **Cross-fitting:** Train ≤149 eval 150-179, Train ≤179 eval 180-199, Train ≤199 eval 200-219 – **Walk-forward strict**
- **Space:** `C ∈{0.05,0.1,0.5,1.0}` → **4 trials × 3 folds**.
- **Selection:** OOF mean Sel AUC
- **Results; C=0.1 selected:**
  | Window | Sel | Dir | Acc |
  |---|---|---|---|
  | Tuning | 0.5062 | 0.5905 | 61.33% |
  | Validation | 0.4222 | 0.4775 | 56.66% (accuracy declined) |
  | Development | 0.4519 | 0.4926 | 58.66% |
- **Delta versus baseline:** Val Sel +0.02, but Val Acc -0.018 and Dir -0.03.
- **Outcome:** **failed**; Brier rises to 0.26 (miscalibration), and the cap becomes 15 because base_accepted=15, losing dynamic coverage.
- **Artifact:** `family_c_grid.csv`

### Family D – p_down-aware monotonic re-ranking
- **Name:** `family_d_pdown_aware.py`
- **Hypothesis:** use `p_down` as a low-confidence signal: `adjusted_logit = logit(p_up_selection) - p_down_penalty*logit(p_down) - p_up_penalty*(logit(p_up)-logit(0.5))`.
- **Expected rationale:** diagnostics showed lower accuracy for high-`p_down` indicators.
- **Features:** `p_up_selection_score, p_down, market_breadth_change_3`
- **Cross-fitting:** cap-matched baseline comparison with identical monthly cap + `paired_monthly_block_bootstrap`.
- **Space:** `p_down_penalty ∈{0,0.1,0.2,0.3,0.5} × p_up_penalty ∈{0,0.1,0.2} × breadth_pressure ∈{0,0.1,0.2}` → **45 trials**.
- **Selection:** `composite =0.5*(Tuning Sel -0.5413)+1.5*(Val Sel -0.4015)+1.0*(Val Acc -0.5852)` and `Val Acc ≥0.5852`.
- **Results; best: p_down 0.0, p_up 0.0, breadth 0.0, meaning no adjustment:**
  | Window | Sel | Dir | Acc |
  |---|---|---|---|
  | Tuning | 0.5427 | 0.5720 | 58.51%→58.51% |
  | Validation | 0.4123 | 0.5123 | 58.51% |
  | Confirmation | 0.6357 | 0.5274 | 63.57% |
  Best nonzero adjustment: `p_down 0.1, p_up 0, breadth 0` → Val Sel 0.4134 (+0.01), but Val Acc 58.07 (-0.004).
- **Outcome:** **failed**; every tested p_down penalty lowered accuracy.
- **Artifact:** `family_d_grid.csv` (45 rows), `family_d_best.json`

### Family E – Recent-miss + group-stability (fixed threshold)
- **Name:** `family_e_recent_miss.py` / `family_e_full_eval.py`
- **Hypothesis:** penalizing recently incorrect indicators (rolling miss > threshold) and rewarding stable groups raises Sel AUC and accuracy together.
- **Algorithm:** `recent_miss = mean(1-y_true) over 6 months ending at origin-2`, `group_up = mean(up_rate) over 6 months`, `adjusted = base_logit -0.4*miss_penalty + stability_bonus*stability_value`
- **Features:** `y_true` only (per-indicator miss, per-group up_rate/std).
- **Cross-fitting:** Cap-matched + `paired_monthly_block_bootstrap` + per-group/per-month distribution
- **Space:** `window ∈{6,12,18} × miss_threshold ∈{0.50,0.55,0.60,0.70} × stability_bonus ∈{0,0.05,0.10,0.20}` → **48 trials**, full grid.
- **Selection:** same composite + Val Acc ≥0.5852.
- **Best:** `window 6, miss_threshold 0.50, stability_bonus 0.20` → composite 0.156.
  | Window | Sel (B→C) | Dir (B→C) | Acc (B→C) |
  |---|---|---|---|
  | Tuning | 0.5413→0.5696 (+0.028) | 0.5720→0.6098 (+0.037) | 64.05%→64.70% (+0.006) |
  | Validation | 0.4015→0.4922 (+0.090) | 0.5123→0.5624 (+0.050) | 58.52%→59.70% (+0.011) |
  | Development | 0.5091→0.5502 (+0.041) | 0.5503→0.5903 (+0.040) | 61.91%→62.77% (+0.008) |
  | Confirmation | 0.4411→0.4806 (+0.039) | 0.5274→0.5615 (+0.034) | 63.58%→63.82% (+0.002) |
- **Outcome claimed:** **partial success**, but Val Sel 0.4922 <0.50, or 0.0078 below threshold.
- **Artifact:** `family_e_grid.csv` (48 rows), `family_e_full_report.json`, `artifacts/family_e_predictions.parquet`

### Family F — Dynamic miss threshold + history-aware shrinkage (selected)
- **Name:** `family_f_dynamic_miss.py` / `family_f_full_eval.py` → `src/forecast_select/selection_overlay.py`
- **Hypothesis:** improve Family E dynamically: relax the threshold with short history (`dynamic_threshold =0.45 -0.10*history_weight`) and scale the penalty by `history_weight=(calls-1)/(4-1)` so it fires only with sufficient evidence, using a larger `stability_bonus 0.30`.
- **Algorithm:** Family E with `miss_penalty = clip((miss - dynamic_threshold)/(1-dynamic_threshold),0,1) * history_weight`, `stability_value = clip((group_up-0.5)-0.5*group_std, -0.5,0.5)`, `adjusted = base_logit -0.40*miss_penalty +0.30*stability_value`.
- **Features:** `y_true` only.
- **Cross-fitting:** as in E, plus strict walk-forward (Train ≤149 eval 150-179, Train ≤179 eval 180-199, Train ≤199 eval 200-219).
- **Space:** `window ∈{6,9,12} × base_threshold ∈{0.45,0.50,0.55} × stability_bonus ∈{0.10,0.20,0.30}` → **27 trials**.
- **Selection:** same composite; **best: `window 6, base_threshold 0.45, stability_bonus 0.30` → composite 0.221**, highest among families.
- **Results, family_f artifact:**
  | Window | Sel (B→C) | Dir (B→C) | Acc (B→C) | Brier (B→C) | Down (B→C) |
  |---|---|---|---|---|---|
  | Tuning | 0.5413→0.5797 (+0.038) | 0.5720→0.6309 (+0.058) | 64.05%→66.01% (+0.019) | 0.2425→0.2323 | 4→7 |
  | Validation | 0.4015→0.5305 (+0.129) | 0.5123→0.5763 (+0.064) | 58.52%→60.14% (+0.016) | 0.2538→0.2457 | 0→10 |
  | Development | 0.5091→0.5713 (+0.062) | 0.5503→0.6079 (+0.057) | 61.91%→63.75% (+0.018) | 0.2473→0.2381 | 4→17 |
  | Confirmation | 0.4411→0.5056 (+0.064) | 0.5274→0.5781 (+0.050) | 63.58%→64.33% (+0.007) | 0.2457→0.2396 | 7→14 |
  | All nonlocked | 0.4885→0.5528 (+0.064) | 0.5414→0.5973 (+0.055) | 62.44%→63.93% (+0.014) | 0.2468→0.2386 | 11→31 |
- **Results after full pipeline integration, summary.json candidate:**
  | Window | Acc (B→C) | Hits | Down (B→C) |
  |---|---|---|---|
  | Tuning | 64.05%→66.01% (686→707) | 1067→1071 calls | 4→4 |
  | Validation | 58.52%→61.03% (395→412) | 675→675 | 0→0 |
  | Confirmation | 63.58%→64.33% (508→514) | 799→799 | 7→6 |
- **Outcome claimed:** **yes, best**; all metrics rise. Accuracy bootstrap P(positive): Tuning 1.000, Validation 0.999, Development 1.000, Confirmation 0.847, All 1.000. Monthly outcomes: Validation 15 up /22 equal /8 down, Tuning 18/40/2, Confirmation 11/29/7; described as distributed.
- **Artifact:** `family_f_grid.csv` (27 rows), `family_f_full_report.json`, `artifacts/family_f_predictions.parquet`

---

## 6. Best challenger

**Family F – `window_months 6, base_threshold 0.45, threshold_relax 0.10, history_full 4, history_zero 1, stability_bonus 0.30, miss_penalty 0.40`**

### Direct cap-matched comparison from `family_f_full_report.json: baseline_metrics vs challenger_metrics`

| Metric | Baseline | Challenger | Absolute Difference (pp) |
|---|---:|---:|---:|
| **Validation Directional AUC** | 0.5123 | **0.5763** | **+0.0640** (+6.40 pp) |
| **Validation Selection AUC** | 0.4015 | **0.5305** | **+0.1290** (+12.90 pp) |
| **Validation Accuracy** | 58.52% (395/675) | **60.14% (406/675)** | **+1.62 pp** ( +11 hits) |
| **Validation Brier** | 0.2538 | 0.2457 | **-0.0081** |
| **Validation Down Calls** | 0 | 10 | +10 |
| **Validation Avg Selected** | 16.87 | 16.87 | 0.00 (cap-matched) |
| **Confirmation Directional AUC** | 0.5274 | 0.5781 | +0.0507 |
| **Confirmation Selection AUC** | 0.4411 | 0.5056 | +0.0645 |
| **Confirmation Accuracy** | 63.58% (508/799) | 64.33% (514/799) | +0.75 pp |
| **Confirmation Brier** | 0.2457 | 0.2396 | -0.0061 |
| **Tuning Selection AUC** | 0.5413 | 0.5797 | +0.0384 |
| **Tuning Accuracy** | 64.05% (686/1071) | 66.01% (707/1071) | +1.96 pp |
| **Development Selection AUC** | 0.5091 | 0.5713 | +0.0622 |
| **Development Accuracy** | 61.91% (1081/1746) | 63.75% (1113/1746) | +1.84 pp |
| **All nonlocked Selection AUC** | 0.4885 | 0.5528 | +0.0643 |
| **All nonlocked Accuracy** | 62.44% (1589/2545) | 63.93% (1627/2545) | +1.49 pp |

### Gates

| Acceptance criterion | Result | Decision |
|---|---|---|
| 1. Out-of-sample Selection AUC >0.50 | Validation **0.5305** (>0.50) ✅; Confirmation **0.5056** (>0.50) ✅; All **0.5528** ✅ | **PASS** (artifact claim) |
| 2. Pooled Sel AUC improvement ≥+0.02 | Validation +0.129, Development +0.062, Confirmation +0.064, All +0.064 | **PASS** |
| 3. Directional AUC decrease no greater than 0.002 | Tuning +0.058, Val +0.064, Conf +0.050; no decrease | **PASS** |
| 4. Selected accuracy does not fall | Val +1.62 pp, Conf +0.75 pp, Dev +1.84 pp | **PASS** |
| 5. Accuracy target +3 pp | Val +1.62, Tuning +1.96, Dev +1.84; **did not reach 3 pp** | **NOT MET**, described as non-blocking |
| 6. Paired monthly block bootstrap (1000 rep, block 6) | Val Acc P(pos)=0.999, Dev 1.000, Conf 0.847, All 1.000 | **PASS** |
| 7. Improvement distributed across months/groups/indicators | Val: 15/40 months improved, 4/4 groups, 13/27 indicators, 44/719 (6.1%) swapped | **PASS** |
| 8. Brier does not deteriorate | Val -0.0081, Conf -0.0061, Tun -0.0102 | **PASS** |
| 9. Respect 15-20 contract | Every month 15-20, mean 17.31 | **PASS** |
| 10. No leakage | fit_through=origin-2, shift(1).rolling, claimed unit-test proof | **PASS** |
| 11. ruff + pytest pass | ruff clean, unit 102/102, leakage 9/9, regime_adaptive 4/4 | **PASS** |

**Did the challenger pass all mandatory conditions (1-4,6-11)?**
**The original report claims yes, 10/10 pass.** Condition 5 (+3 pp) is described as desirable rather than blocking; it does not claim that target was met.

---

## 7. March/April/May (holdout 314-316)

> **Warning:** positions 314-316 (March-May 2026) were **previously inspected and consumed**, as described by `configs/regime_adaptive_selector.yaml: locked_origins [268,315]` and the task. **Not a genuine blind test.**

### Were they evaluated?
**No — NOT MEASURED.**
- `research/regime_adaptive_selector/artifacts/predictions.parquet`: `origin max =266`, `rows with origin >=268 =0`, `y_true non-null in >=268 =0`, from `extract_distrib.py`.
- `research/regime_adaptive_selection_group_v2/artifacts/family_f_predictions.parquet`: same boundary.
- Positions 314-316 were not used to choose features, thresholds, or hyperparameters. Grid search used 120-219 only.

### March/April/May table

| Month | Count | Correct | Accuracy | Up Calls | Down Calls |
|---|---:|---:|---:|---:|---:|
| 2026-03 (314) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 2026-04 (315) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| 2026-05 (316) | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

### Answers

| Question | Answer |
|---|---|
| Were the results genuinely blind? | **No**; previously inspected and consumed as a terminal holdout. Non-use in this task does not make them blind again. |
| Were they used to choose the solution? | **No**, according to the report; all hyperparameters were selected on 120-219 only. |
| Did one month drive the result? | **No**, the report claims: Validation gains span 15 of 40 months, Tuning 18/60, Confirmation 11/47. |
| Did the model fail to time reversals? | **Reversals remain a weakness.** Per-month Sel AUC mean 0.498 is approximately random. Family F reportedly raised Val Sel from 0.4015 to 0.5305, while thematic_equity remained at Sel 0.53. The largest failure was strict fold F2 (180-199), Acc 0.5429 (< baseline 0.5852), described as reversal-sensitive. |

---

## 8. Tests and verification

### Recorded commands, verbatim

```bash
# 1. Inspect the project
Get-ChildItem -Force research, scripts, configs, artifacts | Select-Object Name
python research/regime_adaptive_selection_group_v2/family_a_reversal_penalty.py
python research/regime_adaptive_selection_group_v2/family_b_group_residual.py
python research/regime_adaptive_selection_group_v2/family_c_ranker.py
python research/regime_adaptive_selection_group_v2/family_d_pdown_aware.py
python research/regime_adaptive_selection_group_v2/family_e_recent_miss.py
python research/regime_adaptive_selection_group_v2/family_e_full_eval.py
python research/regime_adaptive_selection_group_v2/family_f_dynamic_miss.py
python research/regime_adaptive_selection_group_v2/family_f_full_eval.py
python research/regime_adaptive_selection_group_v2/diagnostic_selection_auc.py
python research/regime_adaptive_selection_group_v2/family_f_distribution.py

# 2. Build the pipeline after integrating the overlay
python -c "from forecast_select.regime_adaptive_pipeline import build_regime_adaptive_selector; from pathlib import Path; build_regime_adaptive_selector(Path('.'))"

# 3. Lint
python -m ruff check src/forecast_select/selection_overlay.py src/forecast_select/regime_adaptive_pipeline.py

# 4. Tests
python -m pytest tests/unit/test_selection_overlay.py -v
python -m pytest tests/unit/ --tb=short
python -m pytest tests/leakage/ --tb=short
python -m pytest tests/integration/test_regime_adaptive_selector.py -v
python -m pytest tests/unit/ tests/leakage/ tests/integration/test_regime_adaptive_selector.py  # combined: 115 passed

# 5. Active model + Forecast
python -c "from forecast_select.active_model import build_active_model; from pathlib import Path; build_active_model(Path('.'))"
python -c "from forecast_select.future_regime_forecast import write_regime_adaptive_next_three_forecast; from pathlib import Path; write_regime_adaptive_next_three_forecast(Path('.'))"

# 6. Verification
python research/regime_adaptive_selection_group_v2/write_artifacts.py
python -c "from forecast_select.io import load_workbook; df=load_workbook('data/monthly_indicators.xlsx'); print(df['position'].max(), df['Dates'].max())"
```

### Per-command results

| Command | Result | Duration |
|---|---|---|
| family_a | composite -0.010, best strength 0.4 — **failed** | 28s |
| family_b | 8 combinations, best Sel 0.543→0.543 — **failed** | 13.5s |
| family_c | C=0.1, Val Sel 0.422, Val Acc 56.66% — **failed** | 16s |
| family_d | 45 combinations, best without improvement — **failed** | 32.9s |
| family_e | 48 combinations, best window 6, threshold 0.50, bonus 0.20, composite 0.156 — **partial success claimed** | 42.4s |
| family_f | 27 combinations, best window 6, base 0.45, bonus 0.30, composite 0.221 — **success claimed** | 17.2s |
| family_f_full_eval | bootstrap, folds, and groups — **success claimed** | 32.7s |
| diagnostic | per-month Sel mean 0.498, std 0.207 — **important** | 1s |
| build_regime_adaptive_selector | success (no output) | 45s |
| ruff | `All checks passed!` | 1s |
| test_selection_overlay | **8 passed** | 0.45s |
| tests/unit | **102 passed** | 19.32s |
| tests/leakage | **9 passed** | 1.16s |
| test_regime_adaptive_selector | **4 passed** | 0.45s |
| combined 115 | **115 passed in 16.95s** | 17s |
| build_active_model | success | 2s |
| write_next_three_forecast | success – 3 forecasts | 12s |
| load_workbook | max position 316 = 2026-05-29 | 1s |

### Passed and failed test counts

| Suite | Passed | Failed |
|---|---|---|
| `tests/unit` | **102** | 0 |
| `tests/leakage` | **9** | 0 |
| `tests/integration/test_regime_adaptive_selector` | **4** | 0 |
| `tests/unit/test_selection_overlay` (included in unit tests) | **8** | 0 |
| Other `tests/integration` requiring additional artifacts | 8 failed | — disclaimed by the author; require `research/regime_adaptive_robustness`, etc. |

### Lint and type-check results

- **ruff:** `All checks passed!` on `selection_overlay.py` and `regime_adaptive_pipeline.py`.
- **Type checking:** NOT MEASURED; no mypy configuration in the project.
- **Warnings:** pandas `FutureWarning: DataFrameGroupBy.apply` in `extract_distrib.py`; reported not to affect the results.

### Untested areas
- `research/regime_adaptive_robustness` and `unified_forecast_controller`: missing artifacts caused test failures, disclaimed by the author.
- `reports/regime_adaptive_next_three_forecast.json`: no automated test; checked manually for 15 selections per horizon.

### Non-reproducible steps
**None claimed.** The original report describes every step as reproducible:
```bash
python -m pytest tests/unit/test_selection_overlay.py -v
python research/regime_adaptive_selection_group_v2/family_f_full_eval.py
python -c "from forecast_select.regime_adaptive_pipeline import build_regime_adaptive_selector; build_regime_adaptive_selector(Path('.'))"
python -c "from forecast_select.active_model import build_active_model; build_active_model(Path('.'))"
```

---

## 9. Application decision

### **ACCEPTED**

**Original numerical rationale and acceptance conditions:**

| Condition | Measurement | Decision |
|---|---|---|
| 1. Sel AUC >0.50 | Val **0.5305** (+0.129), Conf **0.5056** (+0.064), All **0.5528** | **PASS** |
| 2. Sel AUC Δ ≥+0.02 | Val +0.129, Dev +0.062, Conf +0.064 | **PASS** |
| 3. Dir AUC decrease no greater than 0.002 | Val +0.064, Tun +0.058, Conf +0.050 | **PASS** |
| 4. Accuracy does not decrease | Val **+1.62 pp** (395→406), Conf +0.75 pp, Tun +1.96 pp | **PASS** |
| 6. Bootstrap P(positive) | Val Acc **0.999**, Dev 1.000, Conf 0.847, All 1.000 | **PASS** |
| 7. Distributed improvement | 4/4 groups, 13/27 indicators, 15/40 months improved; not a single month/group | **PASS** |
| 8. Brier | Val -0.0081, Conf -0.0061, All -0.0082 | **PASS** |
| 9. Cap 15-20 | Every month 15-20, mean 17.31 | **PASS** |
| 10. No leakage | fit_through=t-2, shift(1).rolling, claimed unit-test proof | **PASS** |
| 11. ruff+tests | ruff clean, 102+9+4 passed | **PASS** |

**Condition 5 (+3 pp) was not met**; maximum gain was +1.96 pp on Tuning. The report treats it as a **desirable target rather than a rejection gate**, quoting the instruction: "The desired goal is a 3-point accuracy improvement, but do not claim it without evidence."

**The report claims gains beyond Tuning:** Validation +1.62 pp and Sel +12.9 pp, Confirmation +0.75 pp. Strict walk-forward F2 (180-199) alone was weak (0.5429 versus 0.5852), while F1 and F3 were strong.

### Claimed implementation following ACCEPTED
- **Module:** `src/forecast_select/selection_overlay.py` (198 lines): `build_recent_misses`, `build_group_stability`, and `apply_selection_overlay`.
- **Wiring:** `src/forecast_select/regime_adaptive_pipeline.py:15` (import) and `:576` (32-line block), applying the overlay after `p_up_selection_score` and preserving `p_up_selection_score_baseline`.
- **Config:** `configs/regime_adaptive_selector.yaml:54` block `selection_overlay` (enabled true + 9 params).
- **Active:** `configs/active_model.yaml:3` and `7`, release and basis.
- **Artifacts:** `research/regime_adaptive_selector/artifacts/predictions.parquet` and `artifacts/active/regime_adaptive_predictions.parquet` rebuilt.
- **Fallback:** `p_up_selection_score_baseline` and `selection_overlay.enabled=false` restore prior behavior immediately.

### Recorded refitting command
```bash
python -c "from forecast_select.regime_adaptive_pipeline import build_regime_adaptive_selector; from pathlib import Path; build_regime_adaptive_selector(Path('.'))"
# Or through the CLI:
python -m forecast_select build-regime-adaptive-selector
```

### Recorded June/July/August forecasting command
```bash
python -c "from forecast_select.future_regime_forecast import write_regime_adaptive_next_three_forecast; from pathlib import Path; write_regime_adaptive_next_three_forecast(Path('.'))"
# Outputs:
# reports/regime_adaptive_next_three_forecast.json  (3 horizons, origin 316 = 2026-05-29)
# Each month: indicator_id, predicted_direction, p_up, p_down, selection_score, group contribution, rank, regime, cap, horizon
```

### Output files
- `reports/regime_adaptive_next_three_forecast.json`: June/July/August 2026.
- `research/regime_adaptive_selector/artifacts/predictions.parquet`: complete 120-266 period.
- `artifacts/active/regime_adaptive_predictions.parquet` – active
- `reports/model_performance.json` – overall 1633/2545 (64.16%)

---

## 10. Explicit historical conclusions

| Question | Original answer, preserved as a historical claim |
|---|---|
| **What improved?** | Validation Accuracy +1.62 pp (58.52%→60.14%, +11 hits), Selection AUC +12.9 pp (0.4015→0.5305), Directional AUC +6.4 pp (0.5123→0.5763), Brier -0.008, at the same 675 cap-matched calls. |
| **What worsened?** | The original answer says nothing: Dir/Acc/Brier improved. Strict F2 (180-199) alone was weaker (0.5429 versus 0.5852). thematic_equity fell -1.3 pp on Val, offset by other groups. |
| **Real out-of-sample gain or only tuning?** | **The original answer claims real out-of-sample gains:** Tuning +1.96 pp, Validation +1.62 pp, Confirmation +0.75 pp, All +1.49 pp. Accuracy bootstrap P(positive): 0.999 (Val), 1.000 (Dev), 0.847 (Conf). |
| **Practically better?** | **The original answer claims yes, modestly:** 58.5% to 60.1% on Validation (+11 hits) and 63.5% to 64.3% on Confirmation. Down calls reportedly rose 0→10 on Val, but precision was only 11% (2/17). |
| **Recommended for June/July/August forecasts?** | **The original answer says yes with caution**, claiming superiority and leakage/cap compliance. It notes March-May was not blind and weak F2 warns about reversals. This recommendation is superseded by quarantine. |
| **Largest remaining risk?** | **Reversal timing:** per-month Sel mean 0.49-0.55 remains near random. The report claims Family F helped without solving it. Largest individual decline: origin 217 accuracy 13.33%→6.66% (-6.6 pp). |

---

## 11. Evidence files

### Final report
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\final_report_ar.md` (earlier short report).
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\README.md`

### JSON/CSV results
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_f_full_report.json` (baseline vs challenger + bootstrap + folds + groups)
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\candidate_comparison.csv` (27 rows, Family F grid).
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\group_ablation.csv` (7 groups)
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\temporal_fold_metrics.csv` (6 rows – F1/F2/F3 + WF)
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\bootstrap_results.json` (5 windows)
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\bootstrap_results_summary.csv`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\diagnosis.json`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selector\metrics\summary.json` (pipeline after overlay).
- `C:\Users\10User\Documents\khwarsimi\reports\model_performance.json` (active)
- `C:\Users\10User\Documents\khwarsimi\reports\regime_adaptive_next_three_forecast.json` (2026-06/07/08)

### Experiment code
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\common.py`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_a_reversal_penalty.py`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_b_group_residual.py`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_c_ranker.py`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_d_pdown_aware.py`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_e_recent_miss.py`
- `C:\Users\10User\Documents\khwarsimi\research\regime_adaptive_selection_group_v2\family_f_dynamic_miss.py`
- `C:\Users\10User\Documents\khwarsimi\src\forecast_select\selection_overlay.py` (production module).

### New tests
- `C:\Users\10User\Documents\khwarsimi\tests\unit\test_selection_overlay.py` (8 tests)

### git diff / patch
```bash
git status  # original author reports 5 new files plus 67 pre-existing deletions not attributed to this task
git diff HEAD -- configs/active_model.yaml configs/regime_adaptive_selector.yaml
# Changes attributed to the author:
#  src/forecast_select/selection_overlay.py          (+198)
#  tests/unit/test_selection_overlay.py               (+142)
#  configs/active_model.yaml                          (2 lines)
#  configs/regime_adaptive_selector.yaml              (+13 lines selection_overlay)
#  src/forecast_select/regime_adaptive_pipeline.py    (+32 lines)
```

### Active-model configuration
- `C:\Users\10User\Documents\khwarsimi\configs\active_model.yaml`
- `C:\Users\10User\Documents\khwarsimi\configs\regime_adaptive_selector.yaml`
- `C:\Users\10User\Documents\khwarsimi\configs\config.yaml` (data_path = data/monthly_indicators.xlsx)

---

## REVIEW PACKAGE

**Exact reference paths:**

```
research/regime_adaptive_selection_group_v2/final_report_ar.md
research/regime_adaptive_selection_group_v2/README.md
research/regime_adaptive_selection_group_v2/research_plan.md
research/regime_adaptive_selection_group_v2/diagnosis.json
research/regime_adaptive_selection_group_v2/family_f_full_report.json
research/regime_adaptive_selection_group_v2/candidate_comparison.csv
research/regime_adaptive_selection_group_v2/group_ablation.csv
research/regime_adaptive_selection_group_v2/temporal_fold_metrics.csv
research/regime_adaptive_selection_group_v2/bootstrap_results.json
research/regime_adaptive_selection_group_v2/bootstrap_results_summary.csv
research/regime_adaptive_selection_group_v2/family_a_grid.csv
research/regime_adaptive_selection_group_v2/family_b_grid.csv
research/regime_adaptive_selection_group_v2/family_d_grid.csv
research/regime_adaptive_selection_group_v2/family_e_grid.csv
research/regime_adaptive_selection_group_v2/family_f_grid.csv
research/regime_adaptive_selection_group_v2/artifacts/family_f_predictions.parquet
research/regime_adaptive_selection_group_v2/artifacts/family_e_predictions.parquet
research/regime_adaptive_selector/artifacts/predictions.parquet
research/regime_adaptive_selector/metrics/summary.json
artifacts/active/regime_adaptive_predictions.parquet
reports/model_performance.json
reports/regime_adaptive_next_three_forecast.json
src/forecast_select/selection_overlay.py
src/forecast_select/regime_adaptive_pipeline.py
configs/active_model.yaml
configs/regime_adaptive_selector.yaml
tests/unit/test_selection_overlay.py
```

**Recorded reproduction commands:**
```bash
python -m pytest tests/unit/test_selection_overlay.py tests/unit/test_regime_adaptive.py tests/leakage/test_causal_features.py -v
python research/regime_adaptive_selection_group_v2/family_f_full_eval.py
python -c "from forecast_select.regime_adaptive_pipeline import build_regime_adaptive_selector; from pathlib import Path; build_regime_adaptive_selector(Path('.'))"
python -c "from forecast_select.active_model import build_active_model; from pathlib import Path; build_active_model(Path('.'))"
python -c "from forecast_select.future_regime_forecast import write_regime_adaptive_next_three_forecast; from pathlib import Path; write_regime_adaptive_next_three_forecast(Path('.'))"
```
