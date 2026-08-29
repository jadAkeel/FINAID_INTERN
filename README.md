# Forecast Select — Regime Adaptive Bidirectional Selector

A leakage-safe, reproducible research pipeline for forecasting the next-month direction (Up/Down) of 50 anonymous monthly indicators (X1..X50). The system is designed for monthly asset selection under strict causal constraints: **no future information is used at forecast time**.

**Active Model:** `Regime Adaptive Bidirectional Selector` (`forward_breadth_graduated_15_to_20`) — the owner-promoted production policy. It selects **15–20 indicators per month** with a guarded Up/Down decision, using only observations through `t-1` and labels through `t-2`.

---

## 1. Problem and Objectives

**Task:** Each month, select 15–20 indicators most likely to go Up next month, with the option to predict Down when justified. Evaluate on **Directional AUC** (ranking of Up vs Down across all eligible indicators) and **Selection Correctness AUC** (ranking of correct vs incorrect predictions after selection) and **Accuracy** (hits/calls).

**Constraints:**
- Causal: features `<= t-1`, labels `<= t-2` (`fit_through = origin - 2`), no shuffled CV, only walk-forward.
- Cap: `15 <= selected <= 20` per month, cap-matched comparison (same cap for Baseline vs Challenger).
- Horizons: 1, 2, 3 months ahead are trained separately with `effective_lag = 1 + (h-1)`.

**Current Data:** `data/monthly_indicators.xlsx` — 316 positions (Feb 2000 to May 2026, last date `2026-05-29`).

---

## 2. Evolution — What We Did and How We Developed

### Phase 0 — Baseline (Proven Reference)
Built the reproducible baseline from `research/regime_adaptive_selector/artifacts/predictions.parquet` (6349 rows, origins 120–266):
- **Group overlay:** `relative_logit = logit(group_up_rate_12m) - logit(market_up_rate_12m)` (trailing 12, lag 2)
- **Selection:** `logit(p_up_selection) = logit(p_up_graph) + 0.25 * relative_logit` (`src/forecast_select/regime_adaptive_pipeline.py:572`)
- **Ranking:** `selection_score = p_up_selection * (1 - 0.25*stress_excess) - 0.15*p_down` (`src/forecast_select/regime_adaptive.py:596`)
- **Graph:** 48-month signed correlation over percentage returns (`alpha 0.35`)
- **Cap:** graduated `0.52 -> 15 / 0.68 -> 20` via forward market-breadth forecast.

**Baseline Results (cap-matched):**

| Period | DirAUC | SelAUC | Accuracy | Hits/Calls | Cap |
|---|---:|---:|---:|---:|---:|
| Tuning 120–179 | 0.5795 | 0.5413 | 64.05% | 686/1071 | 15-20 |
| Validation 180–219 | 0.5094 | 0.4015 | 58.52% | 395/675 | 15-20 |
| Development 120–219 | 0.5533 | 0.5092 | 61.91% | 1081/1746 | 17.46 |
| Confirmation 220–266 | 0.5380 | 0.4411 | 63.58% | 508/799 | 17.0 |

*Observation:* Tuning SelAUC `0.5413` collapses to `0.4015` in Validation; `corr(p_up, p_up_selection)=0.946` (group adds little independent signal); Validation is `675 Up / 0 Down` (strong Up bias); per-month SelAUC `mean 0.498 std 0.207` (high monthly noise).

### Phase 1 — Diagnostic (Feb–May Holdout)
Consumed positions 314–316 (Mar–May 2026) as a one-time terminal holdout from Feb origin 313:
- Mar 2026: `1/17 (5.88%)`, Apr: `15/17 (88.24%)`, May: `13/17 (76.47%)` — total `29/51 (56.86%)`.
- Finding: the model captures trend continuation (April) but fails on sharp reversal (March 10% Up base rate).

### Phase 2 — Focused Overlay Research (Current)
We kept the **Regime Adaptive core unchanged** (stress, forward breadth, p_down, cap logic) and only varied the **group overlay and selection_score**. Three hypothesis families (max 3, ≤3 params each, temporal folds 120–149 / 150–179 / 180–199 / 200–219, selection on 120–219 only, Confirmation 220–266 evaluated once):

1. **Group Residual with Reliability Shrinkage** — `feat = relative_logit * reliability(N, sign_stability, months, std)`, ridge `alpha 10`. Rationale: down-weight small/noisy groups.
2. **Reversal-Aware Hierarchical Penalty** — `penalty = (0.03*dis + 0.02*breadth_deterioration + 0.02*uncertainty) * regime_factor` (calm 0.8 / mixed 1.0 / stressed 1.1), subtract only. Rationale: penalize disagreement when market deteriorates.
3. **Pairwise Ranker (Logit Lead)** — `new_p = inv_logit(logit(p_graph) -0.5*dis +1.0*lead)`, `lead = peer_nonselected_lead_negative_share`, pairwise logistic `C=0.1` with blocked time folds + isotonic. Rationale: SelAUC is pairwise ranking of correct vs incorrect after selection.

**Exhaustive Follow-up:** 120 linear combos `wg(0-0.6) * wl(0-1.0) * wd(-0.5-0) * wp(0-0.2)` in logit, plus 30 `recent_miss` variants and 1 non-linear conditioned variant. All cap-matched, same temporal protocol. Detailed results in `docs/SELECTION_GROUP_FAILED_REGISTRY.md`.

---

## 3. Results

**Best Challenger in This Cycle (Family3 LogitLead):**

| Metric | Baseline | Challenger | Δ |
|---|---:|---:|---:|
| Validation SelAUC | 0.4015 | **0.4628** | **+0.061** |
| Validation DirAUC | 0.5094 | 0.4932 | -0.016 |
| Validation Accuracy | 58.52% | 58.37% | -0.15pp |
| Development SelAUC | 0.5092 | 0.5284 | +0.019 |
| Confirmation SelAUC | 0.4411 | 0.4563 | +0.015 |

*Exhaustive linear max:* Val Sel `0.4689 (+0.067)` stayed `<0.50` (0/120 crossed 0.50). *Exploratory best* `0.5113 (+0.11)` crossed 0.50 but lost Accuracy `-0.59pp`. `recent_miss` max `0.4720 (+0.07)` also `<0.50`.

**Interpretation:** Consistent improvement in SelAUC (+0.06 to +0.11) was achievable, but always at the cost of DirAUC or Accuracy; no configuration satisfied all gates (`Sel>0.50` & `Δ≥0.02` & `Dir≥-0.002` & `Acc≥0`) cap-matched. The linear ceiling with current 50 indicators is approximately `~0.47` Validation SelAUC.

**Decision:** **No promotion.** Active model stays `forward_breadth_dynamic_cap_v3`. All experiments remain isolated in `research/` and did not modify `configs/active_model.yaml` or `artifacts/active/`.

---

## 4. How to Run (CLI)

```powershell
# Install
python -m pip install -e ".[dev]"

# Audit data (validates 316 positions, last 2026-05-29)
python -m forecast_select.cli audit-data

# Build active Regime Adaptive model (thru May 316, t-2)
python -m forecast_select.cli build-regime-adaptive

# Inspect
python -m forecast_select.cli show-regime-adaptive

# Forecast next three months — June (H1), July (H2), August (H3)
# Each horizon trains Up/Down separately with effective_lag = 1+(h-1)
python -m forecast_select.cli forecast-regime-next-three
# alias:
python -m forecast_select.cli forecast-next-three

# Verify
python -m forecast_select.cli check-project
python -m pytest tests/unit/test_regime_adaptive.py -q  # 19 passed
```

Output: `reports/regime_adaptive_next_three_forecast.json` — for each horizon: `forecast_month`, `horizon_months`, `regime_label`, `regime_stress`, `selection_cap`, `forecast_market_breadth`, and 15–20 selections with `rank`, `indicator_id`, `direction`, `selection_score`, `p_up`, `p_down`, `asset_group`.

**To change the number of selected indicators (cap):**

```powershell
# Fixed 15 per month
python -m forecast_select.cli build-regime-adaptive --cap 15
python -m forecast_select.cli forecast-regime-next-three

# Fixed 20 per month
python -m forecast_select.cli build-regime-adaptive --cap 20
python -m forecast_select.cli forecast-regime-next-three

# Dynamic 15-20 via forward breadth (default, 0.52->15 / 0.68->20)
python -m forecast_select.cli build-regime-adaptive
python -m forecast_select.cli forecast-regime-next-three
```
Valid range: `15 <= cap <= 20` (`configs/regime_adaptive_selector.yaml:13`). Comparison must be cap-matched.

---

## 5. Current Forecast Output (June–August 2026)

Executed on `2026-05-29` Origin 316 (feature information through `2026-04-30`):

```
Generated from: 2026-05-29 | Origin: 316 | Through: 2026-04-30
Method: direct_multi_horizon_frozen_regime_adaptive | Cap mode: guarded_bidirectional_fallback

=== 2026-06 | Horizon 1m | mixed stress 0.487 cap 15 breadth 0.531 ===
   1. X41 Up score 0.715 p_up 0.748 p_down 0.367 group fixed_income
   2. X39 Up score 0.600 p_up 0.649 p_down 0.506 group fixed_income
   3. X40 Up score 0.590 p_up 0.627 p_down 0.438 group fixed_income
   4. X24 Up score 0.552 p_up 0.625 p_down 0.404 group us_sector
   5. X9  Up score 0.546 p_up 0.621 p_down 0.523 group thematic_equity
   6. X11 Up score 0.547 p_up 0.614 p_down 0.470 group thematic_equity
   7. X10 Up score 0.546 p_up 0.610 p_down 0.456 group thematic_equity
   8. X43 Up score 0.560 p_up 0.601 p_down 0.468 group fixed_income
   9. X38 Up score 0.539 p_up 0.593 p_down 0.554 group fixed_income
  10. X30 Up score 0.497 p_up 0.591 p_down 0.539 group us_sector
  11. X3  Up score 0.512 p_up 0.576 p_down 0.455 group thematic_equity
  12. X32 Up score 0.515 p_up 0.574 p_down 0.532 group global_equity
  13. X33 Up score 0.510 p_up 0.567 p_down 0.514 group global_equity
  14. X37 Up score 0.493 p_up 0.557 p_down 0.560 group global_equity
  15. X34 Up score 0.487 p_up 0.544 p_down 0.516 group global_equity

=== 2026-07 | Horizon 2m | mixed stress 0.487 cap 15 ===
   1. X41 Up score 0.708 p_up 0.734 ...
  ... (15 each month)

=== 2026-08 | Horizon 3m | mixed stress 0.493 cap 15 ===
   1. X41 Up score 0.698 p_up 0.743 ...
  ... (15 each month)
```

Full JSON: `reports/regime_adaptive_next_three_forecast.json`.

---

## 6. Limitations

1. **Up bias:** Validation 675 Up / 0 Down; Down is only emitted when `p_down >= 0.65` and `stress > 0.50` — insufficient Down sample to learn a reliable Down policy.
2. **Redundancy:** `corr(p_up, p_up_selection)=0.946`; group overlay adds little independent signal beyond the base model.
3. **Monthly noise:** per-month SelAUC `mean 0.498 std 0.207` — monthly rank noise is high; improvements of `+0.06` are within noise.
4. **Reversal risk:** March 2026 (10% Up base rate) was a sharp reversal the 12-month group drift did not predict.
5. **Linear ceiling:** Exhaustive linear search (120 combos) max Val Sel `0.4689 <0.50`; even the best non-linear conditioned variant reached `0.4672`. Breaking `0.50` cap-matched requires a new causal feature or external data, not just re-weighting current 50 indicators.
6. **Calibration:** Brier on selected is `0.25` — `selection_score` is a ranking utility, not a calibrated correctness probability (see `research/correctness_calibration_audit/`).

---

## 7. For the Reviewer (Dr.)

- **Reproducibility:** All metrics are cap-matched 15–20, `fit_through <= origin-2`, `locked 268-315` never read. March-May 314–316 were consumed once as a terminal holdout in `research/february_holdout_experiment/` and are documented as **not blind** for the current cycle.
- **Evidence:** Baseline artifact `research/regime_adaptive_selector/artifacts/predictions.parquet` (6349 rows); all challengers in `research/regime_adaptive_selection_group_v2/` with `candidate_comparison.csv`, `group_ablation.csv`, `temporal_fold_metrics.csv`, `bootstrap_results.json` (block 6, 500 replicates). No challenger modified `configs/active_model.yaml` or `artifacts/active/`.
- **Statistical claim:** We do **not** claim statistical superiority or +3pp Accuracy. The best observed Validation Accuracy gain was `+0.29pp` (58.52% -> 58.81%) and did not generalize (Confirmation `+0.015` Sel). Block bootstrap `p10` intervals overlap.
- **Next step (if pursued):** A single non-linear, regime-conditioned hypothesis using `lead_negative_share` only when `breadth_change_3 < 0` and `uncertainty > 0.6` with hierarchical shrinkage and isotonic calibration — not yet tried as a standalone promotion candidate.

## 8. Delivery Package

```
research/regime_adaptive_selector/artifacts/predictions.parquet
research/regime_adaptive_selection_group_v2/  # 9 clean files
research/february_holdout_experiment/          # 6 files
docs/SELECTION_GROUP_FAILED_REGISTRY.md        # internal registry (not for README)
configs/active_model.yaml
reports/regime_adaptive_next_three_forecast.json
```
