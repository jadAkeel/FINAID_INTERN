# Experiment Registry

Last updated: 2026-08-30  
Authoritative Negative Results Companion: [`docs/SELECTION_GROUP_FAILED_REGISTRY.md`](SELECTION_GROUP_FAILED_REGISTRY.md)  
Quarantine Record: [`archive/research/agent_selection_group_trials_2026-08-29/QUARANTINE_NOTICE.md`](../archive/research/agent_selection_group_trials_2026-08-29/QUARANTINE_NOTICE.md)

This document is the comprehensive inventory of all active models, baselines, research candidates, diagnostic audits, and quarantined trials in the `forecast_select` repository.

---

## Schema & Status Definitions

Each experiment entry records:
- **Hypothesis**: The scientific/modeling claim being tested.
- **Status**:
  - `active`: The owner-promoted operational model generating current production forecasts.
  - `baseline`: The frozen, reproducible reference baseline.
  - `active_research`: An active research exploration, diagnostic tool, or experimental infrastructure under evaluation.
  - `rejected`: An experiment evaluated under non-locked walk-forward gates and formally rejected for promotion.
  - `contaminated`: An experiment invalid due to data leakage or methodological violation (quarantined).
  - `superseded`: An experiment whose holdout or evaluation window has been consumed or superseded.
  - `uncertain`: An experiment where historical evidence or outcomes are incomplete or unrecorded.
- **Code & Config Paths**: Primary implementation and configuration files.
- **Artifact & Report Paths**: OOF prediction parquet files, metrics summaries, and evaluation reports.
- **Observed Metrics**: Empirical results across development windows (Tuning 120–179, Validation 180–219, Development 120–219, Confirmation 220–266) or holdout periods.
- **Rejection / Decision Reason**: Explicit rationale for promotion, rejection, or retention.
- **Leakage & Holdout Warnings**: Explicit temporal boundaries, fit-through origin lags, and holdout preservation status.
- **Reproduction Command**: Safe command to run or verify the experiment where supported.

Path state is recorded as of 2026-08-30. A path marked **historical** is tracked
in Git but deleted from the current working tree; a path marked **planned
output** is defined by code but has not been materialized. Missing output is not
treated as evidence that an experiment passed or failed.

---

## 1. Production Models & Baselines

### 1.1 Regime Adaptive Bidirectional Selector (`forward_breadth_dynamic_cap_v3`)
- **Hypothesis**: Dynamic 15–20 monthly selection cap conditioned on walk-forward market-breadth forecast (breadth >= 0.65 expands selection from 15 to 20), combined with causal rolling 48-month signed correlation graph, 12-month asset-group relative-strength overlay (weight 0.25), and guarded Down fallback.
- **Status**: `active`
- **Code Paths**: `src/forecast_select/regime_adaptive.py`, `src/forecast_select/regime_adaptive_pipeline.py`, `src/forecast_select/active_model.py`
- **Config Paths**: `configs/active_model.yaml`, `configs/regime_adaptive_selector.yaml`
- **Artifact Paths**: `artifacts/active/regime_adaptive_predictions.parquet`, `research/regime_adaptive_selector/artifacts/predictions.parquet`
- **Report Paths**: `reports/model_performance.json`, `reports/model_performance.md`, `research/regime_adaptive_selector/metrics/summary.json`
- **Observed Metrics**:
  - Tuning (120–179): Accuracy 64.05% (686/1071), Selection AUC 0.5413, Directional AUC 0.5795
  - Validation (180–219): Accuracy 58.52% (395/675), Selection AUC 0.4015, Directional AUC 0.5094
  - Confirmation (220–266): Accuracy 63.58% (508/799), Selection AUC 0.4411, Directional AUC 0.5380
- **Rejection / Decision Reason**: Owner-promoted product decision to support bidirectional Up/Down forecasting. The research promotion gate did not formally pass (Validation accuracy < 65%, Selection AUC < 0.50).
- **Leakage & Holdout Warning**: Features use observations `<= t-1`; training labels stop at `t-2`. Locked evaluation origins 268–315 remain strictly unread.
- **Reproduction Command**: `python -m forecast_select build-model` / `python -m forecast_select show-results`

### 1.2 Uptrend Selector Baseline (`uptrend_logistic`)
- **Hypothesis**: Global regularized logistic regression on cross-sectional features + static frozen signed correlation graph (estimated through origin 119) + causal 48-month trailing target prior selection of top-15 Up calls.
- **Status**: `baseline`
- **Code Paths**: `src/forecast_select/uptrend_model.py`, `src/forecast_select/uptrend_pipeline.py`
- **Config Paths**: `configs/uptrend_model.yaml`, `configs/config.yaml`
- **Artifact Paths**: `artifacts/active/uptrend_predictions.parquet`
- **Report Paths**: `reports/model_performance.json`, `reports/model_performance.md`
- **Observed Metrics**:
  - Development (120–219): Accuracy 61.73% (926/1500), 100% Up calls.
- **Rejection / Decision Reason**: Retained reproducible baseline reference used throughout all research stages.
- **Leakage & Holdout Warning**: Features `<= t-1`, labels `<= t-2`. Locked origins 268–315 unread.
- **Reproduction Command**: `python -m forecast_select build-uptrend-model` / `python -m forecast_select show-uptrend-results`

---

## 2. Historical & Rejected Research Challengers

### 2.1 Directional Downside Selector (`directional_downside_selector`)
- **Hypothesis**: Direct Down target modeling (`Down = 1 - y_true`) blending global logistic, local per-indicator logistic, and rise-then-stall pattern priors with learned lead-lag peer correlations to admit top Down calls into the monthly top 15.
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/directional_downside.py`, `src/forecast_select/directional_downside_pipeline.py`
- **Config Paths**: `configs/directional_downside_model.yaml`
- **Artifact Paths**: `research/directional_downside_selector/artifacts/predictions.parquet`, `research/directional_downside_selector/artifacts/downside_probabilities.parquet`
- **Report Paths**: `research/directional_downside_selector/metrics/summary.json`, `research/directional_downside_selector/metrics/candidate_search.csv`
- **Observed Metrics**:
  - Validation: +5 hit improvement over baseline.
  - Confirmation: 0 net accuracy change (61.79% vs base 61.79%); Down precision 6/12 = 50.0%.
- **Rejection / Decision Reason**: Failed confirmation promotion gate; Down precision was too low to yield reliable net gains. Retained as unpromoted evidence.
- **Leakage & Holdout Warning**: Causal walk-forward, labels `<= t-2`. Locked origins 268–315 unread.
- **Reproduction Command**: `python -m forecast_select.research_cli build-directional-downside` / `python -m forecast_select.research_cli show-directional-downside`

### 2.2 Downside Risk Gate (`downside_risk_gate`)
- **Hypothesis**: Class-balanced logistic classifier estimating sudden extreme drop risk (>2 robust MAD + <5th percentile) to penalize Uptrend rankings without predicting Down directly.
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/downside_risk.py`, `src/forecast_select/downside_pipeline.py`
- **Config Paths**: `configs/downside_risk_gate.yaml`
- **Artifact Paths**: `research/downside_risk_gate/artifacts/gated_predictions.parquet` (**historical**)
- **Report Paths**: `research/downside_risk_gate/metrics/summary.json`, `research/downside_risk_gate/metrics/penalty_search.csv` (**historical**)
- **Observed Metrics**: Penalty search on Discovery 120–219 selected zero penalty (0.0).
- **Rejection / Decision Reason**: The selected penalty was zero, so the gate made no promoted change to the baseline ranking.
- **Leakage & Holdout Warning**: Excludes X16 due to historical scale inconsistencies. Labels `<= t-2`.
- **Reproduction Command**: `python -m forecast_select.research_cli build-risk-gate` / `python -m forecast_select.research_cli show-risk-gate`

### 2.3 Contextual Defensive Selector (`contextual_defensive_selector`)
- **Hypothesis**: During low market-breadth regimes (<0.45), substitute low-confidence selections with defensive neutral asset roles (X44, X49).
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/contextual_defensive.py`, `src/forecast_select/contextual_pipeline.py`
- **Config Paths**: `configs/contextual_defensive_selector.yaml`
- **Artifact Paths**: `research/contextual_defensive_selector/artifacts/predictions.parquet` (**historical**)
- **Report Paths**: `research/contextual_defensive_selector/metrics/summary.json`, `research/contextual_defensive_selector/metrics/candidate_search.csv` (**historical**)
- **Observed Metrics**: Discovery improved +8 hits; Confirmation showed zero net change.
- **Rejection / Decision Reason**: Confirmation generalization failed; roles were purely descriptive without persistent predictive power.
- **Leakage & Holdout Warning**: Causal breadth rolling average through `t-1`.
- **Reproduction Command**: `python -m forecast_select.research_cli build-context-selector` / `python -m forecast_select.research_cli show-context-selector`

### 2.4 Unified Forecast Controller (`unified_forecast_controller`)
- **Hypothesis**: Meta-layer combining downside risk penalty, contextual defensive role bonus, and directional downside probability bonus into a single ranking controller.
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/unified_controller.py`, `src/forecast_select/unified_pipeline.py`
- **Config Paths**: `configs/unified_controller.yaml`
- **Artifact Paths**: `research/unified_forecast_controller/artifacts/predictions.parquet` (**historical**)
- **Report Paths**: `research/unified_forecast_controller/metrics/summary.json`, `research/unified_forecast_controller/metrics/candidate_search.csv` (**historical**)
- **Observed Metrics**: Tuning lost 2 hits (578 vs 580), Validation gained 1 hit (353 vs 352), and Confirmation had zero net change (436 vs 436). Selected component bonuses/penalties were all zero.
- **Rejection / Decision Reason**: The small Validation change did not generalize to Confirmation, and the selected meta-layer weights were zero.
- **Leakage & Holdout Warning**: Non-promoting research layer; locked origins unread.
- **Reproduction Command**: `python -m forecast_select.research_cli build-unified-controller` / `python -m forecast_select.research_cli show-unified-controller`

### 2.5 Selection Score v2 Meta-Ranker (`selection_score_v2`)
- **Hypothesis**: Train a bounded regularized logistic meta-ranker on `p_up_base`, `p_down_base`, `up_down_margin`, and `indicator_prior` to improve selection ranking AUC.
- **Status**: `uncertain` (implementation exists; result is not materialized)
- **Code Paths**: `src/forecast_select/selection_score_v2.py`, `src/forecast_select/selection_score_v2_runner.py`
- **Config Paths**: Grid internal to runner (`C in [0.01, 0.1, 1.0]`)
- **Artifact Paths**: `reports/selection_score_v2/scored_candidates.parquet` (**planned output; absent**)
- **Report Paths**: `reports/selection_score_v2/summary.json` (**planned output; absent**)
- **Observed Metrics**: Unknown; no summary or scored-candidate artifact exists in the current working tree.
- **Rejection / Decision Reason**: Not recorded for this exact runner. The broader negative-results registry rejects related pointwise/selection-correctness rankers, but that evidence is not substituted for a missing v2 result.
- **Leakage & Holdout Warning**: Fitted on Tuning 120–179; locked evaluation origins unread.
- **Reproduction Command**: `python -m forecast_select.research_cli build-selection-score-v2`; `show-selection-score-v2` is valid only after the build materializes its summary.

### 2.6 Directional Ranker v1 (`directional_ranker_v1`)
- **Hypothesis**: Direct Up/Down pointwise regularized logistic ranker using z-scored returns, momentum, and baseline probabilities.
- **Status**: `uncertain` (implementation exists; result is not materialized)
- **Code Paths**: `src/forecast_select/directional_ranker_v1.py`, `src/forecast_select/directional_ranker_v1_runner.py`
- **Config Paths**: Grid internal to runner (`C in [0.01, 0.1, 1.0]`)
- **Artifact Paths**: `reports/directional_ranker_v1/scored_candidates.parquet` (**planned output; absent**)
- **Report Paths**: `reports/directional_ranker_v1/summary.json` (**planned output; absent**)
- **Observed Metrics**: Unknown; no summary or scored-candidate artifact exists in the current working tree.
- **Rejection / Decision Reason**: Not recorded for this exact runner. Related pointwise and pairwise rankers are rejected in the negative-results registry, but this runner has no materialized decision record.
- **Leakage & Holdout Warning**: Fitted on Tuning 120–179; locked evaluation origins unread.
- **Reproduction Command**: `python -m forecast_select.research_cli build-directional-ranker-v1`; `show-directional-ranker-v1` is valid only after the build materializes its summary.

### 2.7 Group Prior & Reliability Challengers (`group_score_challenger`)
- **Hypothesis**: Adaptive group-residual shrinkage and reliability-gated weighting of asset groups over trailing rolling windows.
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/group_score_challenger.py`, `research/regime_adaptive_selection_group_v2/`
- **Config Paths**: N/A
- **Artifact Paths**: `research/regime_adaptive_selection_group_v2/candidate_comparison.csv`, `research/regime_adaptive_selection_group_v2/group_ablation.csv`
- **Report Paths**: `docs/SELECTION_GROUP_FAILED_REGISTRY.md`
- **Observed Metrics**: Validation Selection AUC reached 0.4026; accuracy fell by 0.89 pp (-6 hits).
- **Rejection / Decision Reason**: Failed promotion gate; gains did not persist out of sample.
- **Leakage & Holdout Warning**: Causal residuals stop at `t-2`.
- **Reproduction Command**: No full experiment command is recorded. `python -m pytest tests/unit/test_group_score_challenger.py` verifies the retained helper logic only.

---

## 3. Active Research Studies & Diagnostic Tools

### 3.1 Regime Adaptive Robustness Study (`regime_adaptive_robustness`)
- **Hypothesis**: Stress-test the active Regime Adaptive model under varied replacement caps (0–3) and conservative Down abstention thresholds across calm, mixed, and stressed regimes.
- **Status**: `active_research` (diagnostic implementation; result not materialized)
- **Code Paths**: `src/forecast_select/robustness_pipeline.py`
- **Config Paths**: `configs/regime_adaptive_robustness.yaml`
- **Artifact Paths**: `research/regime_adaptive_robustness/metrics/scenarios.csv` (**planned output; absent**)
- **Report Paths**: `research/regime_adaptive_robustness/metrics/summary.json` (**planned output; absent**)
- **Observed Metrics**: Unknown. The configuration defines 4 replacement caps and 3 abstention policies, but no scenario table or summary is present.
- **Rejection / Decision Reason**: No promotion decision is recorded; the code is a non-mutating diagnostic around the active policy.
- **Leakage & Holdout Warning**: Evaluates through origin 266; locked origins 268–315 unread.
- **Reproduction Command**: `python -m forecast_select.research_cli build-regime-robustness`; `show-regime-robustness` requires the build output.

### 3.2 Correctness Calibration Audit (`calibration_audit`)
- **Hypothesis**: Audit OOF prediction scores with Isotonic and Platt logistic regressors to test whether individual correctness probabilities can be legitimately claimed.
- **Status**: `active_research` (production-support audit)
- **Code Paths**: `src/forecast_select/calibration_audit.py`, `src/forecast_select/correctness_calibration.py`
- **Config Paths**: Evaluates active model artifact
- **Artifact Paths**: `research/correctness_calibration_audit/artifacts/` (**planned output; absent**)
- **Report Paths**: `research/correctness_calibration_audit/metrics/summary.json` (**planned output; absent**)
- **Observed Metrics**: The existing negative-results registry records Validation correctness AUC 0.4392 and corrected selection score 0.3947. No materialized calibration-audit summary is present, so no additional metric is claimed here.
- **Rejection / Decision Reason**: Existing evidence rejects an individualized correctness calibrator; the active release therefore leaves individual correctness probability unavailable. The audit implementation remains reproducible but its current output is not materialized.
- **Leakage & Holdout Warning**: Fit through origin <= `t-2`.
- **Reproduction Command**: `python -m forecast_select.research_cli build-correctness-audit`; `show-correctness-audit` requires the build output.

### 3.3 Extreme Down Sensing Gate (`down_sensing_gate`)
- **Hypothesis**: Detect extreme down tail risks (trailing 60m shock events) and apply guarded replacement to swap weakest Up calls for Down only during low market breadth (<0.50) within strict conviction ceilings.
- **Status**: `active_research`
- **Code Paths**: `src/forecast_select/down_sensing.py`, `src/forecast_select/down_sensing_pipeline.py`
- **Config Paths**: `configs/down_sensing_gate.yaml`
- **Artifact Paths**: `research/down_sensing_gate/artifacts/extreme_scores.parquet` (**planned output; absent**)
- **Report Paths**: `research/down_sensing_gate/metrics/summary.json` (**planned output; absent**)
- **Observed Metrics**: Unknown. The configuration defines evaluation origins 120–266 and a promotion gate requiring non-negative Validation hit delta plus block-bootstrap p10 at least -0.02; no result artifact is present.
- **Rejection / Decision Reason**: No promotion decision is materialized; keep classified as active research.
- **Leakage & Holdout Warning**: Train lag = 2 months; locked origins 268–315 unread.
- **Reproduction Command**: `python -m forecast_select.research_cli build-down-sensing`; `show-down-sensing` requires the build output.

### 3.4 2026-08-07 Non-Locked Research Suite (Replay Cache, Expansion Quality, Ablations, Downside Challengers)
- **Hypothesis**: Provenance-isolated replay caching separating replay inputs from outcomes to evaluate dynamic cap expansion quality, single-family feature ablations, and normalized percentile Down challengers.
- **Status**: `active_research` (infrastructure) / `rejected` (specific candidates)
- **Code Paths**: `src/forecast_select/experiment_cache.py`, `src/forecast_select/regime_experiment_runner.py`, `src/forecast_select/expansion_quality.py`, `src/forecast_select/expansion_experiment_runner.py`, `src/forecast_select/feature_ablation_runner.py`, `src/forecast_select/downside_challengers.py`, `src/forecast_select/downside_challenger_runner.py`
- **Config Paths**: Evaluates regime adaptive research settings
- **Artifact Paths**: `research/regime_adaptive_selector/cache/...`, `research/regime_adaptive_selector/metrics/experiment_ledger.csv`
- **Report Paths**: `research/regime_adaptive_selector/metrics/phase1_summary.json`, `docs/methodology.md` lines 131–137
- **Observed Metrics**: `phase1_summary.json` records three cap policies: the current-binary reference was retained, while the graduated and quality-gate challengers failed their marginal and bootstrap gates. `docs/methodology.md` separately records three feature ablations and two Down challengers, none promoted against that follow-up's 60.00% Validation reference.
- **Rejection / Decision Reason**: Specific candidates rejected; caching and replay infrastructure retained.
- **Leakage & Holdout Warning**: Cache keys reject locked origin ranges; replay inputs are strictly causal.
- **Reproduction Command**: No single full-suite command is recorded. `python -m pytest tests/integration/test_feature_ablation_artifacts.py tests/integration/test_downside_challenger_artifacts.py` validates retained artifact contracts only.

### 3.5 Signal Ceiling Audit (`signal_ceiling_audit`)
- **Hypothesis**: Measure the empirical upper bound and date-block bootstrap distribution of active model accuracy on non-locked OOF predictions.
- **Status**: `active_research` (diagnostic implementation; result not materialized)
- **Code Paths**: `src/forecast_select/signal_ceiling_audit.py`
- **Config Paths**: N/A
- **Artifact Paths**: `research/signal_ceiling_audit/metrics/window_baselines.csv`, `rank_and_coverage.csv`, `temporal_drift.csv`, `block_bootstrap.csv` (**planned outputs; absent**)
- **Report Paths**: `research/signal_ceiling_audit/metrics/summary.json` (**planned output; absent**)
- **Observed Metrics**: Unknown; the output directory is not materialized. The code defines a six-month, 5,000-replicate date-block bootstrap, but generated values are not claimed as observed evidence.
- **Rejection / Decision Reason**: No materialized audit decision. The implementation's declared classification is inconclusive unless and until its non-locked output is generated and reviewed.
- **Leakage & Holdout Warning**: Evaluates only origins <= 266; locked evaluation strictly unread.
- **Reproduction Command**: `python -m forecast_select.signal_ceiling_audit` (writes non-locked diagnostic outputs); `python -m pytest tests/unit/test_signal_ceiling_audit.py` verifies safeguards.

---

## 4. Quarantined Trials & Consumed Holdouts

### 4.1 Quarantined Selection-Group Trials: Family F (Recent-Miss + Group Stability)
- **Hypothesis**: Penalize indicators with a high recent 6-month miss rate while rewarding asset groups with high recent Up stability: `adjusted_score = sigmoid(logit(base) - 0.40 * recent_miss_penalty + 0.30 * group_stability_value)`.
- **Status**: `contaminated` (original implementation) / `rejected` (clean causal rerun) / **QUARANTINED**
- **Code Paths**: `archive/research/agent_selection_group_trials_2026-08-29/src/forecast_select/selection_overlay.py` (contaminated)
- **Config Paths**: `archive/research/agent_selection_group_trials_2026-08-29/configs/regime_adaptive_selector.yaml`
- **Artifact Paths**:
  - Contaminated: `archive/research/agent_selection_group_trials_2026-08-29/artifacts/active/regime_adaptive_predictions.parquet`
  - Clean rerun: `archive/research/agent_selection_group_trials_2026-08-29/corrected_family_f/predictions.parquet`
- **Report Paths**: `archive/research/agent_selection_group_trials_2026-08-29/QUARANTINE_NOTICE.md`, `corrected_family_f/results.json`, `docs/SELECTION_GROUP_FAILED_REGISTRY.md`
- **Observed Metrics (Clean Causal Rerun)**:
  - Tuning (120–179): -6 hits (Accuracy 63.49%, Selection AUC 0.5334, Directional AUC 0.5657)
  - Validation (180–219): +2 hits (Accuracy 58.81%, Selection AUC 0.4222, Directional AUC 0.5146)
  - Development (120–219): -4 hits (Accuracy 61.68%, Selection AUC 0.5066, Directional AUC 0.5484)
  - Confirmation (220–266): -4 hits (Accuracy 63.08%, Selection AUC 0.4497, Directional AUC 0.5296)
- **Rejection / Decision Reason**:
  1. **Contamination Root Cause**: The original implementation used a reversed merge alignment (`shift(1).rolling(...)` merged on `fit_through_origin = origin - 2`), causing labels at `t+1` to leak into the recent-miss penalty for origin `t`. Mutation tests proved mutating origin 21 altered origin 20's miss rate from 1.0 to 0.8333.
  2. **Clean Evaluation**: When corrected so origin `t` strictly uses labels through `t-2`, Family F failed to generalize: Selection AUC remained < 0.50 and accuracy declined in Tuning, Development, and Confirmation.
- **Leakage & Holdout Warning**: **STRICTLY QUARANTINED**. Do not copy code or artifacts from `archive/` into production.
- **Reproduction Command**: None (quarantined).

### 4.2 February Holdout Experiment (March–May 2026 Terminal Holdout)
- **Hypothesis**: Evaluate 4 candidate models (Candidate A: Hierarchical EB prior, Candidate B: Cross-sectional ranker, Candidate C: Selection correction, Candidate D: Reliability-gated group) against the frozen baseline on the terminal holdout (origins 314=2026-03, 315=2026-04, 316=2026-05).
- **Status**: `superseded` / `rejected` (terminal holdout consumed)
- **Code Paths**: `research/february_holdout_experiment/`
- **Config Paths**: N/A
- **Artifact Paths**: `research/february_holdout_experiment/development_report.json`, `march_may_holdout_report.json`, `development_diagnosis.json`
- **Report Paths**: `research/february_holdout_experiment/README.md`, `REPORT_AR.md`, `docs/SELECTION_GROUP_FAILED_REGISTRY.md`
- **Observed Metrics**: Frozen baseline scored 29/51 = 56.86% Up calls (March 2026: 1/17 = 5.88%, April 2026: 15/17 = 88.24%, May 2026: 13/17 = 76.47%). Challengers failed to improve over baseline out of sample.
- **Rejection / Decision Reason**: Terminal holdout was inspected and is now consumed; cannot be reused for parameter or model selection.
- **Leakage & Holdout Warning**: **Consumed Holdout**. March–May 2026 data must not be used for threshold tuning.
- **Reproduction Command**: Historical script names are recorded in `research/february_holdout_experiment/README.md`, but those scripts are absent from the current working tree; no currently runnable full reproduction command is claimed.
