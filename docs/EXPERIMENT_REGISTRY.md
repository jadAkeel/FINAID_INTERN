# Experiment Registry

Last updated: 2026-09-21
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
- **Report Paths**: `reports/uptrend_model_performance.json`, `reports/uptrend_model_performance.md`
- **Observed Metrics**:
  - Development (120–219): Accuracy 61.73% (926/1500), 100% Up calls.
- **Rejection / Decision Reason**: Retained reproducible baseline reference used throughout all research stages.
- **Leakage & Holdout Warning**: Features `<= t-1`, labels `<= t-2`. Locked origins 268–315 unread.
- **Reproduction Command**: `python -m forecast_select build-uptrend-model` / `python -m forecast_select show-uptrend-results`

---

## 2. Historical & Rejected Research Challengers

### Down V2 and V3 research (2026-09-21)

- **Status**: `rejected` for promotion; research artifacts retained for reproducibility.
- **Hypothesis**: A calibrated Down model, improved Down features, or a limited bidirectional replacement policy can increase the hit rate over the current Up-first selector.
- **Code & artifacts**: `research/down_v2/` and `research/down_v3/` (including the V3 row-level audit and frozen parquet selections).
- **Reports**: `research/down_v2/FINAL_REPORT.md`, `research/down_v3/FINAL_REPORT.md`, `research/down_v3/audit/AUDIT_REPORT.md`, and [`RESEARCH_OUTCOMES.md`](RESEARCH_OUTCOMES.md).
- **Observed metrics**: V2 standalone Validation AUC 0.5256 (isotonic); V3 standalone Validation AUC 0.5314. V3 insertion lost 4 hits on Validation (391/675 vs production 395/675) and 5 on Confirmation. Production has 4 / 0 / 7 Down calls in Tuning / Validation / Confirmation; the earlier V3 0 / 0 / 0 claim was corrected by the audit.
- **Decision**: Neither Down version passes the replacement gate. Keep `maximum_replacements: 0` and the owner-promoted active model. Calibration and feature simplification are useful research findings, not approved production changes.
- **Reproduction**: `python research/down_v3/audit/reconcile_audit.py`; `python research/down_v3/v3_selection.py`; `python research/down_v3/final_comparison.py`; `python -m pytest research/down_v3/audit/test_reconciliation.py -q`. See the final reports for the complete, compute-intensive rebuild order.
- **Boundary**: This study used origins 120–266. Treat any previously inspected terminal holdout separately; the consumed March–May 2026 holdout in §4.2 cannot be reused for tuning or promotion.

The Uptrend baseline writes `reports/uptrend_model_performance.*`; the active
model owns `reports/model_performance.*`. These separate paths prevent a baseline
rebuild from overwriting the public active-model report.

### Flexible joint Up/Down selection trial (2026-09-21)

- **Hypothesis**: Rank each eligible indicator's best Up or Down call on a shared Tuning-calibrated scale, allow 0–5 Down calls, and admit optional positions 16–20 only when their score clears a floor.
- **Status**: `rejected` for production promotion; research implementation retained.
- **Code Path**: `research/flexible_directional_selection.py`
- **Artifacts & Report**: `research/flexible_directional_selection/predictions.parquet`, `summary.json`, `README.md`
- **Observed Metrics**: The Tuning-selected policy used floor 0.60, Down margin 0.08, and extra-position floor 0.60. Validation was 357/600 versus 351/600 for the active model at matched monthly coverage, but Confirmation was 445/709 versus 447/709. At the original monthly caps the challenger was 394/675 versus 395/675 on Validation and 505/799 versus 508/799 on Confirmation. It selected five Down calls on Validation and none on Confirmation; its new overlay never fired.
- **Decision**: The apparent 59.50% versus 58.52% raw Validation accuracy difference is confounded by coverage (600 versus 675 calls). Matched-coverage Confirmation and equal-cap comparisons do not support promotion. The active model remains unchanged.
- **Boundary**: Thresholds and Platt calibration were fit on Tuning 120–179. Validation 180–219 was used only for evaluation, Confirmation 220–266 is descriptive, and locked origins were not read. Tuning metrics are in-sample for policy selection.
- **Reproduction**: `python research/flexible_directional_selection.py`; `python -m pytest tests/unit/test_flexible_directional_selection.py`.

### Correctness-trained directional mix trial (2026-09-21)

- **Hypothesis**: Train a joint model of Up-call and Down-call correctness, then choose a variable 0–5 Down mix within the existing 15–20 monthly total.
- **Status**: `rejected` for promotion; research code and row-level results retained.
- **Code & report**: `research/trained_directional_mix.py`, `research/trained_directional_mix/README.md`, `summary.json`, and `predictions.parquet`.
- **Observed metrics**: The Tuning screen gained one hit on 519 calls with one Down call. The frozen final policy selected 133 Down calls in Validation, 61 correct, and scored 373/675 versus 395/675 for the active model. Confirmation was 485/799 versus 508/799. Validation correctness AUC across both directional options was 0.5047.
- **Decision**: The correctness scores and Down-call volume were unstable after refitting. The candidate lost 22 Validation and 23 Confirmation hits and cannot support confidence-based direction choices.
- **Boundary**: Screen model labels stopped at origin 148; the final model stopped at 178 before Validation origin 180. Raw V3 Down scores are walk-forward, but their feature family had already been selected using Tuning 120–179 in prior research, so the screen is exploratory. Locked origins were excluded. Validation and Confirmation are previously inspected, not fresh blind holdouts.
- **Reproduction**: `python -m research.trained_directional_mix`; `python -m pytest tests/unit/test_trained_directional_mix.py`.

### Weighted Up/Down confidence trial (2026-09-21)

- **Hypothesis**: Calibrated weights across Up, Down, opposite-direction evidence, and indicator-history priors can rank the most reliable 15–20 monthly calls and choose a variable Down mix.
- **Status**: `rejected` for promotion; research artifacts retained.
- **Code & report**: `research/weighted_directional_mix.py`, `research/weighted_directional_mix/README.md`, `summary.json`, `weight_screen.csv`, and `predictions.parquet`.
- **Observed metrics**: A bounded 150-candidate screen selected weights 1.0/1.0; no candidate selected Down on screen origins 150–179. The frozen policy selected two Down calls in Validation, one correct, and tied the active model at 395/675; Confirmation was 505/799 versus 508/799. Selected-score correctness AUC was 0.5392 on the screen and 0.4123 on Validation; observed accuracy dropped from 68.15% in the lowest score quintile to 48.15% in the highest.
- **Decision**: No tested weights produced a validated confidence ranking or reliable mixed-direction improvement. Production stays unchanged.
- **Boundary**: Calibrators used labels through origin 148, weights were selected on 150–179, and later windows were only evaluated. Existing upstream model components were themselves developed with previously viewed windows; locked origins were excluded.
- **Reproduction**: `python -m research.weighted_directional_mix`; `python -m pytest tests/unit/test_weighted_directional_mix.py`.

### Open joint-direction competition check (2026-09-22)

- **Hypothesis**: Calibrate Up and Down onto one complementary score, let every eligible indicator's stronger direction compete for the same 15–20 monthly slots, and admit any number of Down calls when they rank highest.
- **Status**: `rejected` for promotion; the fixed-rule research implementation and row-level selections are retained.
- **Code and report**: `research/joint_directional_competition.py`, `research/joint_directional_competition/README.md`, `summary.json`, and `predictions.parquet`.
- **Observed metrics**: Validation 396/675 versus active 395/675, with 0 Down calls and correctness AUC 0.4542. Confirmation 501/799 versus active 508/799, with 0 Down calls and correctness AUC 0.4587. Higher score quintiles were not consistently more accurate.
- **Decision**: The rule allows unrestricted Down competition in code but provides no reliable Down evidence or confidence ranking on this replay. It fails the non-locked gate; production remains unchanged.
- **Boundary**: Arm calibrators use labels through origin 148; the selection rule has no outcome-tuned threshold or weight. Validation and Confirmation and upstream features were viewed in prior research, so this is exploratory rather than a fresh blind test. Locked origins were not read.
- **Reproduction**: `python -m research.joint_directional_competition`; `python -m pytest tests/unit/test_joint_directional_competition.py -q`.

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

### 2.8 Local / Indicator-Specific Logistic Regression (`local_logistic_selector`)
- **Hypothesis**: The production global logistic regression shares every slope across indicators and can only express indicator-specific *intercepts* through its one-hot `indicator_id` block. Allowing indicator-specific *slopes* — via per-indicator local models, Global/Local shrinkage (partial pooling), or `indicator_id x feature` interaction terms — should improve the Up Selector.
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/local_logistic.py`, `src/forecast_select/local_logistic_pipeline.py`, `src/forecast_select/local_logistic_metrics.py`, `src/forecast_select/local_logistic_runner.py`, `src/forecast_select/local_logistic_report.py`, `research/local_logistic/diagnostics.py`
- **Config Paths**: `configs/local_logistic_experiment.yaml`
- **Artifact Paths**: `research/local_logistic/artifacts/predictions.parquet`, `full_signals.parquet`, `tuning_signals.parquet`, `local_coefficients.parquet`, `interaction_coefficients.parquet`
- **Report Paths**: `docs/research/local_logistic_experiment.md`, `research/local_logistic/README.md`, `research/local_logistic/metrics/summary.json`, `metrics/tables.md`, `metrics/tuning_search.csv`
- **Frozen Configuration**: Pure Local = `core8` (8 predictors), `C = 0.01`, minimum 60 local rows. Fixed shrinkage = `core8`, `C = 0.25`, minimum 60 rows, `w_local = 0.40`. Sample-aware shrinkage = `w_max = 0.10`, `n_ref = 180`. Interaction = 6 features x `indicator_id`, `C = 0.01`. All selected on Tuning 120–179 only, then frozen.
- **Observed Metrics** (top-15 hit delta versus the Global Logistic baseline; baseline 579/900, 347/600, 436/705):
  - Pure Local: Tuning −2, Validation +7, Confirmation 0
  - Global + Local fixed shrinkage: Tuning −1, Validation +8, Confirmation −1
  - Global + Local sample-aware shrinkage: Tuning −5, Validation +3, Confirmation −1
  - Global + indicator interactions: Tuning −9, Validation +4, Confirmation +2
  - Raw directional AUC (Global / Pure Local / Interaction): Tuning 0.5538 / 0.5264 / 0.5150; Validation 0.4963 / 0.5292 / 0.4891; Confirmation 0.5198 / 0.5308 / 0.5153
- **Rejection / Decision Reason**: `GLOBAL LOGISTIC REMAINS PREFERRED`. Zero of 220 tuning configurations beat the baseline on the very window that selected their hyperparameters, and no candidate is positive in two consecutive evaluation windows; every block-bootstrap interval that excludes zero is contradicted by an adjacent window. Local modelling is retained as research evidence only.
- **Secondary Findings**: Indicator-specific structure is genuine — per-indicator accuracy spread exceeds a matched coin-flip null (p ≈ 0.000) and local slopes reproduce at r = 0.63–0.87 between origin 179 and origin 266 — but it concentrates on indicators the selector never picks (+2.87 pp on never-selected indicators, −0.51 pp on selected ones). Local models are better calibrated than the global model in all three windows. A held-fixed-selector ablation shows the trailing 48-month prior dominates: replacing `p_up` with a constant scores 570/900, 352/600, 438/705, matching or beating the production model out of sample. Flagged for separate work; no production change proposed.
- **Leakage & Holdout Warning**: Features `<= t-1`, training labels `<= t-2`; imputers, scalers, encoders and every model refit per origin. The workbook is read with `nrows` capped at position 267 and `assert_origins_unlocked` rejects any origin `>= 268`, so locked origins 268–315 were not read or used.
- **Reproduction Command**: `python -m forecast_select.local_logistic_runner tune` / `python -m forecast_select.local_logistic_runner evaluate` / `python -m forecast_select.local_logistic_report` / `python research/local_logistic/diagnostics.py`

### 2.9 Chronos-2 Zero-Shot Directional Forecasting (`chronos_zero_shot`)
- **Hypothesis**: A pre-trained probabilistic time-series foundation model (Chronos-2, `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`) can forecast indicator direction zero-shot, either on its own or blended with the active model, without any task-specific training.
- **Status**: `rejected`
- **Code Paths**: `src/forecast_select/chronos_protocol.py`, `src/forecast_select/chronos_cache.py`, `src/forecast_select/chronos_distribution.py`, `src/forecast_select/chronos_evaluation.py`, `src/forecast_select/chronos_pipeline.py`, `research/chronos_zero_shot/run_chronos_evaluation.py`
- **Config Paths**: `configs/chronos_zero_shot.yaml`, `requirements-pretrained.lock` (`chronos-forecasting==2.3.2`, installed via the optional `pretrained` extra)
- **Artifact Paths**: `research/chronos_zero_shot/artifacts/predictions.parquet`, `matched_coverage.parquet`
- **Report Paths**: `docs/experiments/chronos2_zero_shot.md`, `research/chronos_zero_shot/readme.md`, `research/chronos_zero_shot/metrics/summary.json`, `window_metrics.csv`, `matched_coverage.csv`, `diversity_metrics.csv`
- **Frozen Configuration**: Two-step differenced formulation. At origin `t` the series is differenced only within the causal cutoff `t-1`; Chronos-2 forecasts 2 steps, and step-2 quantiles `[0.10 … 0.90]` are interpolated piecewise-linearly to give `P(Up) = 1 - F(0)`, clipped to `[0.001, 0.999]`. Platt calibration is prequential on Tuning 120–179 (fit strictly through `t-2`, minimum 12 origins, both classes required) and then frozen at origin `178` for Validation and Confirmation. The blend weight `w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` minimizes Brier on Tuning out-of-fold rows only, ties preferring `w = 1.0`; the selected weight was **0.75**, favoring the active model. Seed `20260727`.
- **Observed Metrics**: Matched-coverage hit deltas versus the active model are negative in every window — Tuning −15 (671 vs 686), Validation −16 (379 vs 395), Confirmation −12 (496 vs 508) — with origin-blocked bootstrap p10 of −23, −28, and −23. On common eligible rows the calibrated Chronos probability beats the active model on Validation accuracy (0.5355 vs 0.5246) and calibration (ECE 0.0314 vs 0.0589), but that advantage does not survive selection at matched coverage.
- **Rejection / Decision Reason**: The non-locked gate fails three of five criteria: `validation_hit_delta_positive`, `validation_bootstrap_p10_nonnegative`, and `confirmation_hit_delta_nonnegative`. Only `validation_blend_brier_improves_active` and `no_locked_reads` pass. `promotion_eligible` is `False` and `active_model_changed` is `False`.
- **Secondary Findings**: Better probability calibration did not convert into better selected calls. Chronos-2 is pre-trained on broad public corpora that may include these macro series, so historical backtest strength cannot by itself support promotion; genuine out-of-sample evidence requires live forward tracking or post-release paper trading. Promotion additionally requires a locked evaluation that has not been run.
- **Leakage & Holdout Warning**: Differencing and context stop at `t-1`, calibration labels at `t-2`. `maximum_origin_position` is 266 and `locked_evaluation_read` is `False`; any origin `>= 267` or an attempted locked read raises. Origin 267 is an excluded buffer and locked origins 268–315 were not read. Validation and Confirmation were viewed in prior research, so this is an exploratory replay rather than a fresh blind test.
- **Reproduction Command**: `python -m pip install -e ".[pretrained]"` then `python -m forecast_select build-chronos-zero-shot` / `python -m forecast_select show-chronos-zero-shot`, or `python research/chronos_zero_shot/run_chronos_evaluation.py`. Tests: `python -m pytest tests/unit/test_chronos_*.py tests/integration/test_chronos_pipeline.py` (the Chronos package is optional; torch-dependent cases skip without it).

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
- **Status**: `active_research` — **materialized 2026-09-22**; classification `inconclusive` for any stable > 65% claim.
- **Code Paths**: `src/forecast_select/signal_ceiling_audit.py` (per-window evidence is now computed from the audited rows; the earlier hardcoded 381/635 literals were removed)
- **Config Paths**: N/A
- **Artifact Paths**: `research/signal_ceiling_audit/metrics/window_baselines.csv`, `rank_and_coverage.csv`, `temporal_drift.csv`, `block_bootstrap.csv`
- **Report Paths**: `research/signal_ceiling_audit/README.md`, `research/signal_ceiling_audit/metrics/summary.json`
- **Observed Metrics**: 2,545 selected calls over 147 non-locked months, pooled accuracy 62.44% (Tuning 686/1071 = 64.05%, Validation 395/675 = 58.52%, Confirmation 508/799 = 63.58%). Six-month block bootstrap, 5,000 replicates: median 62.60%, p10–p90 **59.96%–65.11%**, p95 65.81%, P(≥ 62%) = 61.6%, **P(≥ 65%) = 11.3%**. Ranks 1–2 hit 66–69%, ranks 3–15 hit 54–65%, ranks 16–19 hit 59–62%; a fixed 15-call policy on the same rows scores 63.13% versus the delivered 62.44%.
- **Rejection / Decision Reason**: Not a promotion decision. The audit's central finding is that accuracy tracks the realized Up-prevalence of the selected set year by year almost exactly (2015: 46.7% / 46.7%; 2017: 74.8% / 74.8%; 2019: 76.2% / 76.7%; Jan–Apr 2022: 30.2% / 30.2%), because 2,534 of 2,545 calls are Up. Year-to-year accuracy swings (rolling-12 range 42.5%–76.2%) are market breadth, not model skill.
- **Leakage & Holdout Warning**: Evaluates only origins ≤ 266; locked evaluation strictly unread. Learning curves, null-signal tests, error overlap, and oracle bounds are recorded as unavailable because their prerequisite artifacts were not pre-registered.
- **Reproduction Command**: `python -m forecast_select.signal_ceiling_audit`; `python -m pytest tests/unit/test_signal_ceiling_audit.py`.

### 3.6 Prior-Only Selector Diagnostic (`prior_only_selector`)
- **Hypothesis**: Quantify how much of the active model's accuracy a zero-feature rule — top-15 by causal trailing Up-rate, all called Up — already explains, across every window length, with nothing selected.
- **Status**: `active_research` (diagnostic; not a candidate)
- **Code Paths**: `research/prior_only_selector/prior_only_diagnostic.py`
- **Config Paths**: N/A (window lengths 24/36/48/60/96/all, k ∈ {15, 20}, modes `up_only` and `majority` are all reported)
- **Artifact Paths**: `research/prior_only_selector/metrics/prior_only_windows.csv`
- **Report Paths**: `research/prior_only_selector/README.md`
- **Observed Metrics**: Production 64.05% / 58.52% / 63.58% (Tuning / Validation / Confirmation). Best zero-feature rule per window: N=48 63.33% on Tuning, N=60 60.33% on Validation, N=96 62.41% on Confirmation — a different window wins each time, spread ≈ ±2 pp. Fixed N=48: 63.33% / 57.67% / 61.70%. Majority-direction (Down allowed) is worse in every window; k=20 is below k=15 everywhere; excluding X16 changes nothing.
- **Rejection / Decision Reason**: Not a promotion decision. Production's edge over the best zero-feature rule is about one point on Tuning and Confirmation and negative against N=60 on Validation. That one point sits inside the ±2 pp window-to-window spread, which is the same magnitude as every challenger improvement recorded in Section 2. Promoting the N=60 rule for its Validation win would be the exact window-picking the table exposes.
- **Leakage & Holdout Warning**: Labels used only through `t-2`; no origin above 266 read; the loader asserts that no locked row is loaded. Confirms `research/accuracy_feasibility/key_results.csv` (rolling 60-month Up-rate 62.40% Discovery, 60.69% Confirmation) and the local-logistic constant-`p_up` ablation.
- **Reproduction Command**: `python research/prior_only_selector/prior_only_diagnostic.py`.

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
