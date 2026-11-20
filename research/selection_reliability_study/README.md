# Indicator-ranking reliability and top-15–20 selection study (Up-only first)

Research only. The active model was unchanged during this study; no automatic replacement is authorized.

Local evidence note: the study runner and generated artifacts are not included
in this documentation-only update. The commands below describe the retained
local study; this commit alone is not a complete runnable experiment.

- Active: `regime_adaptive_selector` (`forward_breadth_dynamic_cap_v3`) — unchanged.
- Baseline: `uptrend_logistic` — reference only.
- Run: `python research/selection_reliability_study/run_study.py` from the repository root.
- Outputs: `summary.json`, `diagnosis.json`, `monthly_*.csv`, `calibration_quintiles.csv`,
  `swaps_*.csv`, `predictions_*.parquet`, `scored_panel.parquet`, `weights.json`.
- Windows: Tuning 120–179 (decision selection), Validation 180–219 (decision gate),
  Confirmation 220–266 (descriptive). Locked 268–315 was not read by this study, enforced in code.
- **Disclosure:** Validation and Confirmation were inspected in earlier research;
  these results are an **exploratory replay**, not a fresh blind test. Neither
  March–May 2026 nor Locked is used for tuning or promotion.

---

## 1. Existing-work review before implementation

| Component | Status | Retained result | Failure or limitation | Specific difference in this study |
|---|---|---|---|---|
| Uptrend Selector (`uptrend_model.py`, `uptrend_pipeline.py`) | Frozen baseline and `uptrend_predictions.parquet` (120–219: 926/1500=61.73%) | Available | Reference only; fixed Top-15 | Unequal-coverage reference; primary fair comparison is Regime-Up at identical caps |
| Regime Adaptive (`regime_adaptive.py`, `regime_adaptive_pipeline.py`) | Owner-promoted active model; research gate did not pass | Tuning 686/1071=64.05%; Val 395/675=58.52%; Conf 508/799=63.58% | Validation AUC 0.4015; rare Down calls (11/147 months) | Isolate Up-only at identical caps/directions to measure ranking |
| `correctness_calibration.py` + audit | Supporting research; no valid individual calibrator | Correctness AUC 0.4392; corrected score 0.3947 | No individual-confidence claim; cohort monitoring only | First test whether even global univariate calibration fails monotonicity (it does) |
| Correctness-trained directional mix (`trained_directional_mix.py`) | Rejected; results retained | Screen +1 on 519 (1 Down); Val 373/675 versus 395 (133 Down, 61 correct); Conf 485/799 versus 508; AUC 0.5047 | Down volume exploded after refitting (1→133) and was unstable | **Up-only first**; no Down score in primary ranking; at most 4 features; no variable 0–5 Down mix |
| Weighted directional mix (`weighted_directional_mix.py`) | Rejected; results retained | Weights 1.0/1.0; 0 screen Down; Val tied at 395/675; Conf 505/799 versus 508; AUC 0.5392 (screen) to 0.4123 (Val); highest quintile 48.15% versus lowest 68.15% | Confidence inversion: higher scores were less accurate | Explicitly diagnose inversion, then test group shrinkage on final scores rather than directional-arm blending |
| Joint directional competition (`joint_directional_competition.py`) | Rejected; results retained | Val +1 (396/675, 0 Down, AUC 0.4542); Conf −7 (501/799); monotonicity failed | Free Down competition without reliable evidence | Fixed rule without tuned thresholds versus this study's Tuning-frozen settings; no Down competition here |
| Group prior/reliability (`group_score_challenger.py`, `regime_adaptive_selection_group_v2/`) | Rejected | Val Selection AUC 0.4026; −6 hits (−0.89pp) | Reweighting the group signal within selection_score did not generalize | **Different source:** post-calibrate final scores by group with shrinkage; target correctness rather than residuals |
| Local Logistic (`local_logistic.py` + pipeline) | Rejected research | Pure −2/+7/0; Fixed −1/+8/−1; Sample-aware −5/+3/−1; Interaction −9/+4/+2 (Top-15 deltas); the retained report describes no candidate as positive across two consecutive windows | Real indicator structure (spread exceeds null, p≈0.000; slope stability r=0.63–0.87) is concentrated outside selected rows (+2.87pp unselected, −0.51pp selected); the 48-month prior dominates | No base-feature retraining; meta model uses OOF scores + prior + counts only; base retraining requires a separate experiment |
| selection_score_v2 (`selection_score_v2.py`) | **Code without results** (`uncertain`; `reports/selection_score_v2/` absent) | Unknown | 14 features; C∈[0.01,0.03,0.1,0.3] | **Distinct study:** ≤4 features, C∈{0.1,1.0,10.0}, Up-correctness (y==1) rather than Down-aware decision correctness, with leakage guards, ablation, and paired bootstrap |
| directional_ranker_v1 (`directional_ranker_v1.py`) | **Code without results** (`uncertain`) | Unknown | 20 numerical features + indicator/group one-hot, predicting y directly | Predict Up-selection correctness from OOF without one-hot, with causal shrinkage and uncertainty |

Non-duplication rule: these candidates differ in hypothesis (Up-only reliability rather than Down mix),
information source (final score + prior + counts rather than raw features or Down scores),
and evaluation design (matched caps/directions + paired bootstrap + swaps + ablation).

---

## 2. Separate three questions: scaling, calibration, and uncertainty

1. **Feature scaling before fitting:** `StandardScaler` is inside the base pipeline
   (`uptrend_model.py`: median-impute → scale → Logistic, refitted per origin
   on `causal_training_rows` only), `local_logistic`, and the small meta model here
   (median → scale → Logistic, fitted only on training data). Scaling remains causal.
2. **Cross-indicator score calibration:** the current score is a `ranking utility`
   (`selection_score_is_probability=False`), not a correctness probability. Evidence: score quintiles
   reverse on Val/Conf (lowest accuracy 68.9%/72.5%; highest 48.9%/57.5%).
3. **Prediction uncertainty and stability:** use only causal history counts and binomial SE,
   with penalty k frozen on the screen. Lower volatility is not assumed to mean easier direction:
   `corr(acc, market_dispersion)≈−0.07`, and `corr(score_mean, acc)≈−0.07`,
   while accuracy follows `market_up_rate` (corr≈0.93), not mean score.

Scaling does not turn ranking utility into a valid correctness probability. It standardizes
feature units before Logistic; mapping scores to P(correct) requires causal post-calibration,
whose monotonicity failed even at the global level here.

---

## 3. Current-ranking diagnosis (Up-only, matched caps, using OOF predictions)

Fair reference: `regime_up` = identical eligible rows (`level_c_ready` and not excluded) +
identical monthly `regime_cap` (15–20) + identical directions (Up) + `p_up_selection_score` ranking.

| Window | Calls | Hits | Acc | Correctness AUC (score→correct) | Brier | ECE10 | Quintile monotonicity |
|---|---:|---:|---:|---:|---:|---:|---|
| Tuning 120–179 | 1071 | 686 | 64.05% | 0.542 | — | — | Failed (0.586/0.645/0.607/0.636/0.729: not fully monotonic) |
| Validation 180–219 | 675 | 395 | 58.52% | **0.412** | — | — | **Reversed:** 0.689/0.637/0.630/0.481/0.489 |
| Confirmation 220–266 | 799 | 505* | 63.20%* | **0.439** | — | — | **Reversed:** 0.725/0.644/0.629/0.606/0.575 |

\* Up-only: 505/799 versus 508/799 for the original bidirectional policy (7 original Down calls, 5 correct;
removing Down loses 3 Conf hits and ties Tuning/Val, which contain 4/0 Down calls).
`p_up_selection_score` gives the same pattern: Val quintiles
0.681/0.644/0.607/0.511/0.481 — higher scores are less accurate.

- **By indicator (α=10 shrinkage toward mean 62.44%):** raw spread is real, but
  some indicator samples are small; shrinkage narrows `max−min` and prevents treating 70 versus 75
  as evidence (for example, n<60 indicators shrink strongly; see `diagnosis.json:per_indicator`).
- **By group**, after shrinkage: fixed_income 64.5% (614), thematic 63.4% (358),
  us_sector 61.9% (1060), global_equity 60.6% (376), commodity 58.0% (88),
  currency 57.1% (49). Small-sample Wilson intervals overlap; arbitrary weights are unsupported.
- **By market**, with sufficient observations: accuracy closely follows `market_up_rate` (2015:46.7/46.7,
  2017:74.8/74.8...); `mixed` 65.9% (85 months) > `stressed` 59.8% (35) > `calm` 53.9% (27).
- **Selected versus selection boundary** (Top-cap versus the next 5 at the same score definition):
  mean +8.4pp overall (Tuning +12.2pp, Val +2.7pp, Conf +8.1pp).
  Ranking partly separates selected from unselected cases but **does not reliably order the selected cases**.
- **Indicator volatility versus prediction instability:** market dispersion is unrelated; instability
  (per-indicator score variance, median ≈0.05–0.10) does not improve ranking; the evidence is that
  the highest scores deteriorate outside Tuning.

Diagnosis: **cross-indicator/group score reliability differs outside
Tuning (AUC<0.5 and reversal)**, but one 75-versus-70 case is not evidence;
shrinkage, adequate samples, and monotonicity across slices are needed.

---

## 4. Bounded plan fixed before fitting

- **Start Up-only** to isolate ranking; bidirectionality is separate and was not implemented after Up-only failed.
- **3 alternatives only**, plus a calibration reference:
  - `global_platt`: univariate Platt `logit(p_up)→P(correct)` (C=1e6). Monotonic, so ranking is unchanged; calibration only.
  - `group_shrunk`: Platt plus empirical-Bayes group correction: `sigmoid(logit(p_glob)+[logit(shrunk_g)−logit(glob)])`, `k∈{10,50,200}` (one parameter).
  - `meta_small`: regularized Logistic with 4 features (`logit_p, prior, log_hist, group_logit`) and internal StandardScaler, `C∈{0.1,1.0,10.0}` (one parameter).
  - `uncertainty_penalized`: `p_grp − k·SE`, `SE=√(p(1−p)/n)`, causal `n=history_rows` (≥5), `k∈{0,0.5,1.0}` (one parameter). `k=0` is an allowed explicit negative result.
- **Fitting:** expanding, not rolling. Frozen rule: screen fit through 148,
  screen evaluation 150–179, final refit through 178 for Val/Conf. Group rates
  are recomputed per origin using `labels≤t−2` (expanding from 120, ≥12 months); weights
  and calibrators stay frozen. Rolling fitting was not tested.
- **Prohibited:** arbitrary score multipliers or unrestricted search until gains appear. A hypothesis needing
  base-model retraining or new features requires a separate experiment, not performed here.

---

## 5. Leakage prevention

- Features through `t−1` (causal OOF features in the artifact).
- Training outcomes through `t−2`: fits use `origin≤148` for screening and `≤178` for the final model;
  expanding group rates use `origin−2` per forecast origin;
  `assert max_origin≤266`, rejecting any `≥268`.
- Fit scaling, imputation, and calibration on training only (`SimpleImputer(median)` +
  `StandardScaler` inside a Pipeline fitted on `train_early`/`train_full` only).
- Meta fitting uses earlier OOF predictions outside the base model's fitting sample (120–149/120–178 OOF),
  not in-sample predictions.
- Hyperparameters, thresholds, and calibration rules are selected within Tuning 120–179 using temporal
  internal splits (120–149 versus 150–179). The update policy above was fixed before evaluation.
- Tests: `tests/unit/test_selection_reliability_study.py` checks temporal boundaries
  (`fit_through≤origin−2`), no Locked reads, and identical caps/eligibility/directions.

---

## 6. Fair evaluation

The primary comparison uses identical eligible rows, monthly caps, and Up directions.

| Window | Original Uptrend (fixed 15) | Original Regime (15–20, bidirectional) | Fair Regime-Up (same caps, Up) | New candidates (same caps, Up) |
|---|---:|---:|---:|---|
| Tuning 120–179 | 579/900=64.33% | 686/1071=64.05% | 686/1071=64.05% | global 686 (+0); group 687 (+1); meta 672 (−14) |
| Validation 180–219 | 347/600=57.83% | 395/675=58.52% (0 Down) | 395/675=58.52% | global 395 (+0); group 386 (**−9**); meta 391 (**−4**) |
| Confirmation 220–266 | — (not covered by the registered Uptrend artifact) | 508/799=63.58% (7 Down, 5 correct) | 505/799=63.20% | global 505 (+0); group 496 (**−9**); meta 492 (**−13**) |

- Changes to selection counts or directions are reported separately. The only difference here is
  Regime-Up versus original Regime on Conf (−3 from removing correct Down calls), documented as
  a separate control rather than a ranking improvement.
- **Holdout-consumption ledger:** Uptrend/Regime were developed using inspected Tuning/Val/Conf;
  March–May 2026 is consumed (29/51=56.86%, not reused); Locked 268–315 was read once on
  2026-09-23 (60.97% versus Uptrend 61.81%) and is not touched here. No unused holdout is claimed.

---

## 7. Auditable results

- **Per-window/month hits, calls, and accuracy:** `monthly_*.csv` (reference + each candidate); `summary.json:challengers.*.windows`.
- **Hit/accuracy delta against each baseline:** above, with paired monthly comparisons below.
- **Calibration and accuracy by slice:** `calibration_quintiles.csv`; Val group example:
  67.4%/57.8%/55.6%/53.3%/51.9% from low to high (not monotonic); meta Val AUC 0.430,
  Brier 0.251, ECE≈0.07.
- **Metrics:** probabilities → Brier/ECE/AUC; ranking → correctness AUC and quintile monotonicity.
- **Uncertainty:** 6-month block bootstrap ×2000 (seed 20260727), paired on the same months.

| Candidate | Val paired mean | p10–p90 | P(>0) | Val B/E/W months | Conf paired mean | P(>0) |
|---|---:|---|---|---|---|---|
| group_shrunk | −1.30pp | −1.98/−0.62pp | 1.1% | 6/20/14 | −1.13pp* | ~0%* |
| meta_small | −0.59pp* | — | — | 9/18/13 | −1.63pp* | — |
| global | 0 | — | — | 0/40/0 | 0 | — |

\* Calculated consistently in `summary.json`, with paired bootstrap retained for every window.
- **Entries/exits and swap contributions:** `swaps_*.csv`. Val group example: 20 months without swaps,
  14 worse versus 6 better; net swaps lose −9 hits on Val and −9 on Conf.
  Meta: net −4/−13. Global: 0 swaps, as expected for a monotonic transform.
- **Separate ablation:** global alone 0 (ranking unchanged by construction); +group −9/−9;
  small meta −4/−13; +uncertainty (screen-selected k=0) equals group exactly — **no component helps alone**.
- **Monthly weights/calibrators and reproduction:** `weights.json`
  (Platt coefficient/intercept, `k=200`, `C=0.1`, meta coefficients and means/scales, `k_unc=0.0`) +
  `scored_panel.parquet` (all scores for each eligible case) + the command above.
- **One successful and one unsuccessful month**, group versus Regime-Up, from input to outcome:
  - **Success: 217 (Val)** — broad-down market (`market_up=34.9%`, cap 15). The `p_up` reference
    selects X32/X37/X35/X36/X40... (2/15 correct: X40+X38 only). Group shrinkage lowers
    thematic/global (X11 0.625→0.570, X32 0.685→0.650) and raises fixed_income
    (X43 0.561→0.579, X38 0.601→0.608), inserting X43/X24 (y=1) for X11/X3 (y=0):
    4/15 versus 2/15 (**+2**). A small group adjustment happened to match a weak thematic month;
    this did not generalize (7 better versus 15 worse on Conf).
  - **Failure: 195 (Val)** — mixed market (`market_up=46.3%`, cap 17). Reference 12/17.
    Group shrinkage lowers thematic strongly (X11 0.585→0.518, X3 0.585→0.518), removing
    X11/X3 (both y=1) and inserting X42/X22 (y=0): 10/17 versus 12/17 (**−2**).
    The expanding shrunk thematic rate penalized two indicators that actually rose that month,
    illustrating that shrinkage does not distinguish call correctness from a group prior.

---

## 8. Delivery and adoption

**Do cross-indicator score-reliability differences exist?**
Yes outside Tuning: AUC 0.412 (Val), 0.439 (Conf), reversed quintiles, and failed monotonicity
even after global Platt calibration. The magnitude is entangled with market effects (accuracy follows breadth at 0.93);
a single 70-versus-75 observation is insufficient. Shrinkage and adequate samples are needed.

**Did the correction improve selection?**
No: global 0 (expected), group −9/−9, meta −4/−13, uncertainty = group (k=0).
Val paired-bootstrap intervals are below zero (group p90 −0.62pp).

**Is the improvement stable or concentrated in a few months/indicators?**
There is no improvement. Even scattered successes (6/40 Val months for group) are outnumbered by failures
(14/40), with a negative net result and no stability.

**What remains unproven?**
- Individual-indicator calibration with sufficient samples (most indicators have n<150 selected observations;
  the current shrinkage was insufficient).
- Causal uncertainty estimation more useful than `SE=√(p(1−p)/n)` (failed here; k=0).
- Rolling versus expanding fitting, not tested.
- Bidirectional Down-mix claims, deferred because Up-only failed first.
- Final promotion requires later data excluded from development, such as a new Locked
  period or live forward evaluation, rather than reused Val/Conf.

**Tests:** `tests/unit/test_selection_reliability_study.py` checks
temporal boundaries (`fit_through≤origin−2`, `max≤266`), leakage (no Locked; scaler fitted on
training only), matched caps/eligibility/Up directions, and no duplicates.

**Adoption:** `active_model` and the baseline were unchanged (`git status` clean for those files).
The negative result is recorded explicitly. Final adoption needs later data excluded
from development; that forward evaluation was not performed in this study.

---

## Appendix: reproduction and files

```powershell
python research/selection_reliability_study/run_study.py
python -m pytest tests/unit/test_selection_reliability_study.py -q
```

- `summary.json`: all windows, deltas, paired bootstrap, and B/E/W months.
- `diagnosis.json`: score, indicator/group (shrunk), market, near-miss, volatility, and stability slices.
- `monthly_*.csv`, `calibration_quintiles.csv`, `swaps_*.csv`, `predictions_*.parquet`,
  `scored_panel.parquet`, `weights.json`.
- `run_study.py`: the single frozen rule; writes neither `artifacts/active` nor `configs/active_model.yaml`.
