# Down V3 — Bidirectional Selection Research Report

**Goal:** every month, select the strongest 15–20 directional forecasts from all
eligible indicators, with a flexible Up/Down mix capped at **0–5 Down calls**,
where a Down call enters only when it is genuinely stronger than the Up slot it
displaces — and a Down overlay remains available to correct Up candidates.

**Method:** walk-forward, causal, leakage-safe, same availability lag as
production. Tuning 120–179 (screen) → Validation 180–219 (gate) → Confirmation
220–266 (descriptive). Locked origins 268–315 never read. **No production code
modified.** Everything lives in `research/down_v3/`.

Reproduce:
`build_v3_panel.py` → `v3_ablation.py` → `v3_models.py` → `up_v3.py` →
`v3_selection.py` → `final_comparison.py`.

---

## 1. Current architecture

Verified from code, configs, and the frozen artifact (`artifacts/active/
regime_adaptive_predictions.parquet`):

1. `p_down` **is** computed on all eligible indicators
   (`src/forecast_select/directional_downside_pipeline.py`, walks all
   `down_eligible` rows).
2. The Base Pool is created by `p_up_selection_score` alone
   (`src/forecast_select/regime_adaptive.py:681-684`).
3. `maximum_replacements = 0` (`configs/regime_adaptive_selector.yaml`) blocks
   any out-of-pool indicator from entering. P1 proved 0/147 months, 0
   replacements.
4. The Down model is a **Direction Override**, not an independent selector.
   Jaccard(Up top-15, Down top-15) = 0.2154 — the two models genuinely disagree,
   but only one is allowed to admit members.

So the current system is exactly: all indicators → Up ranking → Up base pool →
Down overlay inside that pool. Confirmed.

## 2. Problems and constraints

Carried from the V2 research, all re-verified:

- **Architectural:** no out-of-pool indicator can ever be selected. 230 rows
  with `p_down ≥ 0.65` across 65/147 months were ignored.
- **Calibration:** raw `p_down` predicted 0.90 → observed 0.52.
- **Comparability:** `max(p_up, p_down)` AUC 0.5365; margin scores below chance
  (0.4541). No raw cross-direction comparison is legitimate.
- **Modeling:** the full 28-feature Down model is the *worst* config on
  Validation (0.5015); the local per-indicator model is noise (~0.50).
- **Evaluation:** thresholds tune well and collapse out of sample.

One correction to the earlier V2 report: `lead_peer` features are **37%**
missing, not "mostly NaN" — the accurate statement is that they need origin 100
for >50% coverage and are the weakest family standalone.

## 3. Why the current Down model is weak

Ranked by evidence: the signal itself is weak (best Validation AUC 0.5314 after
all fixes); thresholds do not transfer; complexity actively overfits; the local
model is noise; `class_weight="balanced"` breaks calibration. See the V2 report
for the full ranking — V3 addressed the top item by changing the *feature space*.

## 4. Down V3 feature ablation

V3 replaces absolute-price features with **relative** ones: "is this indicator
weaker than everything else, and is it breaking down?" 42 features in 5 families,
all built from `frame[indicators].shift(availability_lag=1)` — identical causal
boundary to production.

Full walk-forward refit (147 origins each), C screened on Tuning (selected
C=1.0):

| Config | n feat | Tuning AUC | **Validation AUC** | Confirmation AUC | V top-5 |
|---|---|---|---|---|---|
| only_mkt | 5 | 0.5222 | **0.5362** | 0.5211 | 0.4750 |
| only_int | 2 | 0.5600 | 0.5302 | **0.5429** | 0.4900 |
| only_group | 8 | 0.5562 | 0.5295 | 0.5427 | 0.4900 |
| only_rel | 13 | 0.5567 | 0.5272 | 0.5418 | 0.4750 |
| drop_break | 28 | 0.5232 | 0.5328 | 0.5159 | 0.4650 |
| drop_rel | 29 | 0.5267 | 0.5140 | 0.5080 | 0.4900 |
| drop_group | 34 | 0.5261 | 0.5086 | 0.5117 | 0.4800 |
| only_break | 14 | **0.5647** | 0.5057 | 0.5280 | 0.5000 |
| drop_int | 40 | 0.5258 | 0.5108 | 0.5065 | 0.4850 |
| drop_mkt | 37 | 0.5533 | 0.5002 | 0.5228 | 0.4800 |
| **full** | **42** | 0.5246 | **0.5099** | 0.5071 | 0.4850 |
| best2 (T-selected) | 16 | 0.5622 | 0.5047 | 0.5292 | — |
| best3 (T-selected) | 29 | 0.5568 | 0.4983 | 0.5282 | — |
| **v2_reference** | 10 | 0.5423 | 0.5252 | 0.5306 | 0.5300 |

**Three findings:**
1. The new relative families **do** beat V2: only_mkt 0.5362, only_int 0.5302,
   only_group 0.5295, only_rel 0.5272 all exceed V2's 0.5252, and three of the
   four also beat it on Confirmation.
2. **Complexity is still harmful.** The full 42-feature model is 0.5099 — worse
   than every single family. Greedy Tuning-selected combinations (best2, best3)
   are worse still. This reproduces the V2 finding on a brand-new feature space.
3. The family with the best *Tuning* AUC (only_break, 0.5647) is the second
   *worst* on Validation (0.5057). Tuning AUC is a poor selector — which is why
   the model stage below screens with a bootstrap-SE parsimony rule instead.

**Feature set used downstream:** `only_int` — the two market-interaction
features (`v3_int_weak_in_strong`, `v3_int_weak_in_weak`). Selected by a
Tuning-only rule: among configs within one bootstrap SE (SE = 0.0118) of the
best Tuning AUC, take the fewest features. Validation was never consulted.

## 5. Model comparison

Same data, same walk-forward, 2 interaction features:

| Model | Tuning AUC | Validation AUC | Confirmation AUC | V top-1 |
|---|---|---|---|---|
| **Logistic** | **0.5585** | **0.5314** | **0.5421** | 0.4750 |
| HistGradientBoosting | 0.4846¹ | 0.4846 | 0.4890 | 0.5250 |
| LightGBM | 0.4599 | 0.4788 | 0.4870 | 0.5250 |
| ensemble (log+lightgbm) | 0.5033 | 0.5102 | 0.5227 | 0.5500 |

¹HGB/LightGBM are **below chance** on all three windows. With ~9k rows, ~43%
base rate, and a near-noise signal, gradient boosting has nothing to learn from
and memorises noise instead. Tuning model correlations: logistic↔hgb 0.14,
logistic↔lightgbm 0.14, hgb↔lightgbm 0.90 — the tree models are highly
correlated with each other and carry almost no independent information relative
to logistic. **The ensemble is therefore not complementary evidence; it is a
dilution of the only model that works.** Per the brief's rule, no ensemble.

**Decision: regularized Logistic Regression.** No new production dependency is
justified — LightGBM/XGBoost are available in the environment (4.6.0 / 3.1.3)
but are not used because they fail on this data, not because they are absent.

## 6. Calibration

Platt and Isotonic fit on Tuning only, frozen before Validation:

| Variant | Window | AUC | Brier | ECE | mean pred | observed | top-1 |
|---|---|---|---|---|---|---|---|
| raw | tuning | 0.5585 | 0.2422 | 0.0217 | 0.4239 | 0.4302 | 0.6500 |
| raw | validation | 0.5314 | 0.2500 | 0.0392 | 0.4423 | 0.4626 | 0.4750 |
| **platt** | tuning | 0.5585 | 0.2421 | 0.0229 | 0.4302 | 0.4302 | 0.6500 |
| **platt** | validation | 0.5314 | 0.2498 | 0.0377 | 0.4484 | 0.4626 | 0.4750 |
| **platt** | confirmation | 0.5421 | 0.2437 | 0.0299 | 0.4423 | 0.4271 | 0.6596 |
| isotonic | tuning | 0.5678 | 0.2406 | **0.0000** | 0.4302 | 0.4302 | 0.6833 |
| isotonic | validation | 0.5298 | 0.2514 | 0.0387 | 0.4519 | 0.4626 | 0.4500 |
| isotonic | confirmation | 0.5396 | 0.2447 | 0.0297 | 0.4423 | 0.4271 | 0.5957 |

**Platt is the choice.** ECE holds at 0.023–0.038 across all three windows;
isotonic is perfect in-sample (0.0000 by construction) and no better out of
sample (0.0387). Mean prediction tracks observed within 0.015 on Validation and
0.015 on Confirmation. Raw `p_down` is not used for any cross-direction
comparison — only as a control, per the brief.

**Answer to the specific question — "does p_down = 0.80 mean ~80% Down?":**
No. In the raw production model the [0.8, 1.01) bucket contains 25 cases with
mean prediction 0.9022 and observed Down rate **0.5200** — an overconfidence of
+0.38. After V3 + Platt the equivalent top bucket is well calibrated, but the
production number is close to a coin flip with a confident label.

## 7. Down V3 standalone

Standalone ranking of all eligible indicators per origin (logistic + Platt):

| Window | AUC | Brier | ECE | top-1 | top-2 | top-3 | top-4 | top-5 | top-10 |
|---|---|---|---|---|---|---|---|---|---|
| tuning | 0.5585 | 0.2421 | 0.0229 | 0.6500 | 0.6083 | 0.5722 | 0.5542 | 0.5233 | 0.4883 |
| validation | 0.5314 | 0.2498 | 0.0377 | 0.4750 | 0.5000 | 0.4917 | 0.5125 | 0.5000 | 0.5050 |
| confirmation | 0.5421 | 0.2437 | 0.0299 | 0.6596 | 0.5532 | 0.5106 | 0.5160 | 0.5021 | 0.4723 |

vs V2 (platt): Validation AUC 0.5252→0.5314, Confirmation 0.5306→0.5421,
top-3 0.4750→0.4917, top-5 0.4900→0.5000. **V3 wins on every standalone
measure that matters out of sample**, with 2 features instead of 10.

There is **no depth decay problem**: top-5 (0.5000) ≈ top-10 (0.5050) on
Validation. The signal is uniformly weak rather than concentrated, so a
"sparse high-confidence selector" hypothesis is not supported either — the
quality at top-1 is not reliably better than at top-10 out of sample
(0.4750 vs 0.5050, i.e. top-1 is *worse*).

**Overlay never fires.** Across all designs and windows, overlay flips = 0.
The reason is structural: an Up candidate is in the pool precisely because
`p_up_selection_score` ranked it top-20, so `p_up_cal` is high; the Down model
rarely beats it by any margin, let alone a strict one. The overlay path exists
in the code and is tested, but the data does not supply candidates for it.

## 8. Overlay performance

Zero flips in every window for both V2 and V3, at every margin tested (0.00
through 0.08) and every floor (0.50 through 0.65). **Path B is inert on this
data** — not because it is broken, but because the Up pool's weakest member
still outranks the Down model's read of it. Keeping the overlay in the design is
correct (it costs nothing and would fire in a stressed regime), but it produced
no calls and no evidence of value.

## 9. External Down insertion performance

Path A does fire. With margins/floors screened on Tuning and frozen:

| Variant | Window | final down calls | down prec | replacement delta |
|---|---|---|---|---|
| V3 (margin 0.00, floor 0.55) | tuning | 5 (1 insert + 4 prod) | 0.6000 | **+1** |
| V3 | validation | 34 (34 inserts) | 0.4118 | **−4** |
| V3 | confirmation | 20 (13 inserts + 7 prod) | 0.5500 | **−5** |
| V2 (margin 0.08, floor 0.65) | tuning | 4 (0 inserts + 4 prod) | 0.5000 | 0 |
| V2 | validation | 0 | — | 0 |
| V2 | confirmation | 7 (0 inserts + 7 prod) | 0.7143 | 0 |

V3's Tuning delta is +1 on 1 event — one net hit gained. On Validation the same
frozen policy admits 34 calls at 0.4118 precision against Up victims that were
correct 52.9% of the time (18/34), for a net **−4**. Confirmation agrees: −5.
This is the honest core result: **the Down model can find candidates that look
strong on Tuning, but out of sample those candidates are worse than the Up call
they displace.**

Note the subtlety worth reporting honestly: on Validation the total hit rate
*fell* (391 vs A's 395) by exactly the replacement delta. An earlier version of
this paragraph said the hit rate "rose slightly (397 vs 395) because 16 of the
34 displaced Up victims would have been wrong anyway" and called it base-rate
noise. That was wrong twice over: a removed row cannot add to a hit count, and
the apparent +2 was a scoring bug (`hits` summed the Up-correct label `y_true`
over all rows, which scores a Down call as a hit exactly when it is wrong). See
`audit/AUDIT_REPORT.md` for the full row-level reconciliation. Every number in
this report is now direction-aware and reconciles exactly.

## 10. Dynamic 0–5 Down policy

The cap is enforced and working: monthly Down distribution on Validation is
0 calls × 10 months, 1 × 26, 2 × 4 — never above 2, well inside the ≤5 limit.
The policy is genuinely dynamic (no forced quota) and returns 0 Down in 10 of
40 Validation months. **The mechanism works; the signal does not.**

## 11. Replacement delta analysis

| Variant | Tuning | Validation | Confirmation |
|---|---|---|---|
| V2 | 0 (4 calls) | 0 (0 calls) | 0 (7 calls) |
| V3 | +1 (5 calls) | −4 (34 calls) | −5 (20 calls) |

Delta by source: all recorded calls are external inserts (overlay = 0
everywhere). Delta is not segmentable by regime with any confidence at these
sample sizes — 34 calls over 40 months is too thin to split further.

## 12. Down V2 vs V3 decision

| Measure | V2 | V3 | Winner |
|---|---|---|---|
| Validation AUC | 0.5252 | **0.5314** | V3 |
| Confirmation AUC | 0.5306 | **0.5421** | V3 |
| Validation top-3 | 0.4750 | **0.4917** | V3 |
| Validation top-5 | 0.4900 | **0.5000** | V3 |
| Tuning Brier | 0.2435 | **0.2421** | V3 |
| Validation ECE | **0.0335** | 0.0377 | V2 (marginal) |
| n features | 10 | **2** | V3 |
| **Validation replacement delta** | **0** | **−4** | **V2** |

**V3 is the better Down model; V2 is the safer Down policy.** V3 wins every
standalone quality measure, but its higher AUC translates into *worse*
replacement value out of sample — it is confident more often, and that
confidence is misplaced. Under the brief's promotion rule (positive
replacement delta on Validation, bootstrap p10 ≥ 0), **neither promotes**:
V3's bootstrap P(Δ>0) = 0.064 with p10 = −0.1088 (direction-aware; the pre-audit values 0.363 / −0.0408 came from the same scoring bug).

## 13. Up V3 research

Four families tested with the same walk-forward protocol (C screened on Tuning,
selected C=1.0):

| Config | Tuning AUC | Validation AUC | V top-5 | V top-20 | **V ranks 11–15** | **V ranks 16–20** |
|---|---|---|---|---|---|---|
| current_up | 0.5795 | 0.5094 | 0.5600 | 0.5775 | **0.6350** | 0.5300 |
| only_persistence | 0.5560 | 0.5296 | 0.5500 | 0.5675 | — | — |
| only_quality | 0.5623 | 0.5244 | 0.5850 | 0.5737 | 0.5300 | 0.5800 |
| only_relative | 0.5659 | **0.5337** | 0.5700 | 0.5587 | 0.5500 | 0.4900 |
| only_breadth | 0.5526 | 0.5277 | 0.5500 | 0.5675 | — | — |
| full | 0.5488 | 0.5240 | 0.5950 | 0.5725 | 0.5250 | 0.5850 |

**Up V3 improves list quality but not the marginal slots.** only_relative beats
the current Up on Validation AUC (0.5337 vs 0.5094) and top-5 (0.5700 vs 0.5600),
but at ranks 11–15 — exactly the slots a Down candidate would displace — the
current Up is markedly better (0.6350 vs 0.5500). Since the whole point of a
stronger Up here is to make the marginal slots harder (or easier) to displace,
Up V3 does not help the bidirectional question. It may be worth pursuing for its
own sake, but that is a separate decision.

## 14. Current Up vs Up V3

Not promoted. The current Up has the better marginal-slot accuracy, which is the
criterion that matters for this experiment, and its Tuning AUC (0.5795) is the
best of any config in this entire study. Up V3 stays as research.

## 15. Final architecture comparison

All designs: 15–20 calls (frozen per-origin regime cap), 0–5 Down dynamic, no
forced quota. Parameters screened on Tuning, frozen, Validation as gate,
2000-replicate origin-block bootstrap vs A.

| Design | Window | calls | hits | rate | final down | down prec | flips | ext. inserts | prod down | delta |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** production | T/V/C | 1071/675/799 | 686/395/**508** | .6405/.5852/**.6358** | 4/0/7 | .5000/—/.7143 | 0 | 0 | 4/0/7 | 0 |
| B (Up + V2 overlay) | T/V/C | 1071/675/799 | = A | = A | 4/0/7 | .5000/—/.7143 | 0 | 0 | 4/0/7 | 0 |
| C (Up + V2 + insert) | T/V/C | 1071/675/799 | = A | = A | 4/0/7 | .5000/—/.7143 | 0 | 0 | 4/0/7 | 0 |
| D (Up + V3 overlay) | T/V/C | 1071/675/799 | = A | = A | 4/0/7 | .5000/—/.7143 | 0 | 0 | 4/0/7 | 0 |
| E (Up + V3 + insert) | tuning | 1071 | 687 | .6415 | 5 | 0.6000 | 0 | 1 | 4 | +1 |
| E | validation | 675 | **391** | **.5793** | 34 | 0.4118 | 0 | 34 | 0 | **−4** |
| E | confirmation | 799 | **503** | **.6295** | 20 | 0.5500 | 0 | 13 | 7 | **−5** |
| F (Up V3 + V2) | T/V/C | 1071/675/799 | 659/391/488 | .6153/.5793/.6108 | 25/29/26 | .5200/.5172/.3077 | 0 | 0 | 25/29/26 | 0 / 0 / 0 |
| G (Up V3 + V3) | tuning | 1071 | 662 | .6181 | 56 | 0.6250 | 0 | 31 | 25 | +3 |
| G | validation | 675 | 381 | .5644 | 105 | 0.4857 | 0 | 76 | 29 | **−10** |
| G | confirmation | 799 | 480 | .6008 | 57 | 0.3509 | 0 | 31 | 26 | **−8** |

`final down` = selected rows whose final direction is Down = external inserts +
overlay flips + production Down rows already in the base pool. F and G carry an
additional `up_ranking_delta` (they replace the Up ranking itself), which is
measured separately in `audit/AUDIT_REPORT.md` §7: F −4, G −4 on Validation. For
every design and window, `arch_hit_delta == replacement_delta +
up_ranking_delta` with residual 0.

Bootstrap vs A (Validation+ origin-level, direction-aware):

| Design | Δ hit rate | p10 | p90 | P(Δ>0) |
|---|---|---|---|---|
| B / C / D | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| E | −0.0544 | −0.1088 | −0.0068 | 0.064 |
| F | −0.3469 | −0.4898 | −0.2041 | **0.002** |
| G | −0.4490 | −0.6122 | −0.2925 | **0.000** |

**B, C, and D are exactly A.** This is not a bug — it is the result: with the
Tuning-screened margin/floor, V2 admits zero Down calls out of pool and V3's
overlay never fires, so those designs reduce to production. They are
architecturally different and behaviourally identical on this data.

## 16. Winning monthly mix policy

**A — current production**, with a 15–20 total-call cap and 4 / 0 / 7 Down
calls in Tuning / Validation / Confirmation. It was not beaten. Every design that actually admitted Down
calls lost hits out of sample, and the two designs that replaced the Up ranking
lost catastrophically (F: −0.35, G: −0.45, both with bootstrap confidence).

> **Corrigendum.** An earlier version of this report reported A's Confirmation
> as 505 hits / .6320 and E's Validation as 397 / .5881 (+2 vs A). Both were
> wrong: `hits` was computed as `sum(y_true)` over all rows, and `y_true` is the
> **Up-correct** label, so a Down call was scored as a hit exactly when it was
> wrong. Corrected: A Confirmation 508 / .6358, E Validation 391 / .5793 (−4,
> matching the replacement delta). Design A also showed 0/0/0 Down calls because
> `pred_dir` was never propagated to the selector, hiding production's real
> 4/0/7. Full row-level reconciliation, per-origin ledger, and regression tests
> in **`audit/AUDIT_REPORT.md`** and **`audit/test_reconciliation.py`**. No
> production code or parameter was changed; the decision is unchanged and, on
> corrected arithmetic, stronger.

The 0–5 dynamic Down mechanism itself behaves exactly as specified — it fires
0, 1, or 2 times per month, never forces a quota, and returns 0 Down in 10 of 40
Validation months. The policy is sound; the Down signal is not strong enough to
populate it. **0 Down is the correct output of a working policy, not a failure.**

## 17. What failed

1. **Gradient boosting.** HGB and LightGBM are below chance (0.48) on all three
   windows. Not a tuning issue — there is not enough signal or sample.
2. **Ensembling.** Tree models correlate 0.90 with each other and only 0.14
   with logistic; the "ensemble" dilutes the only working model.
3. **Feature-count scaling.** Full 42-feature V3 (0.5099) is worse than every
   2–13 feature family. Replicates the V2 finding on a new feature space.
4. **Greedy family combination.** best2/best3 chosen on Tuning are the two
   worst Validation configs (0.5047/0.4983).
5. **Up V3 as a marginal-slot strengthener.** Better AUC, worse ranks 11–15.
6. **The overlay path.** Zero flips at every margin/floor, both models, all
   windows.
7. **External Down insertion.** Tuning delta +1 → Validation −4, Confirmation
   −5. The single clearest negative result in the study.
8. **Threshold/margin transfer.** V3's Tuning-optimal loose policy admits 34
   Validation calls; its Tuning-strict policy admits 0. The setting that looks
   best on Tuning is the one that loses the most out of sample.

## 18. What remains uncertain

- **Whether the Down signal is genuinely absent or merely too weak at this
  sample size.** V3's Validation AUC of 0.5314 is real and above chance, but
  0.53 does not beat a 0.58 Up pool at the marginal slot. The distinction
  matters: "no signal" says stop; "weak signal" says the bar is 0.55+ AUC and
  this data cannot reach it.
- **Whether a longer Confirmation window would change the replacement delta.**
  20 calls over 47 months is thin; −5 is directionally consistent with
  Validation's −4 but not decisive alone.
- **Regime-conditional Down selection.** Not re-tested under V3. The V2
  evidence (best regime 0.5458 against a 0.5852 bar) was negative, but V3 is a
  different model and the question is technically open. Low priority: the
  V2 evidence was strongly negative and V3's replacement delta is worse.
- **Whether Up V3's top-of-list gain (top-5 0.5850 vs 0.5600) is real.** It is
  the one positive Up result, but it does not address the marginal slots.

## 19. Exact production changes if a winner exists

**No winner exists, so no production change is proposed.** The fallback below
applies. For completeness, the changes that *would* be made if a future Down
model clears the bar (kept here as the pre-registered specification):

1. `configs/regime_adaptive_selector.yaml` — set
   `maximum_replacements_grid: [5]`, add `maximum_down_share_grid: [5]` as an
   absolute monthly Down cap (not a share), and add `down_margin_grid` /
   `down_floor_grid` parameters screened on Tuning only.
2. `src/forecast_select/regime_adaptive.py` — in
   `apply_regime_adaptive_selector` (line 457), build the replacement pool from
   `ready` minus `base_pool` ranked by a *calibrated* Down score, and admit a
   candidate only when `p_down_cal >= p_up_cal + margin` against the weakest
   remaining Up slot, subject to the ≤5 combined cap.
3. `src/forecast_select/directional_downside.py` — replace the blended Down
   model with the 2-feature logistic + Platt calibration fit on Tuning only;
   remove `class_weight="balanced"`; drop the local and pattern components.
4. `tests/test_down_replacement_delta.py` (new) — assert any promoted
   margin/floor has non-negative replacement delta on Validation, with
   bootstrap p10 ≥ 0. Today's V3 would fail this gate.
5. `docs/methodology.md` — document that `p_down` requires Platt calibration
   before any cross-direction comparison, and that `max(p_up, p_down)` is not a
   valid unified score (AUC 0.5365, margin 0.4541).

## 20. Exact fallback if no V3 wins

**This is the operative section.** No V3 wins, so:

- **Production stays exactly as-is:** Up-first selection with the current Down
  overlay, `maximum_replacements: 0`. On this evidence that setting is
  *correct* rather than merely conservative — it is what kept the system at
  0.5852/0.6358 while every alternative lost hits.
- **Do not adopt Down V3 as a selector.** It is a better standalone model
  (V AUC 0.5314) and a worse selector (delta −4). Standalone quality does not
  imply replacement value; that is the central lesson of this study.
- **Do not adopt Up V3.** Better Validation AUC, materially worse at the
  marginal ranks that matter here.
- **Keep both as monitoring signals**, not selection paths. V3's calibrated
  Down probability and Up V3's relative-strength ranking are cheap, causal, and
  worth re-evaluating when more data accumulates past origin 266.
- **`configs/active_model.yaml` stays** `owner_promoted` /
  `research_gate_passed: false`.
- **Re-run this study unchanged** when ~24 more origins are available. The
  pre-registered bar: Validation replacement delta > 0 with bootstrap p10 ≥ 0
  and Confirmation agreement, at ≥20 Down calls. V3 today is −4 at 34 Validation calls and −5 at 13 Confirmation events.

---

## Answer to the final question

> *What is the best system for selecting 15–20 monthly forecasts with a flexible
> 0–5 Down mix, where Down enters only when stronger than the Up alternative,
>  and a Down overlay remains available?*

**A — current production.** Its Up-first ranking, existing Down direction
overlay, and zero out-of-pool replacements were not beaten. A separate V3
dual-path policy (overlay correction plus external insertion) was built and
tested with a calibrated common scale, a margin requirement, a ≤5 Down cap,
and no forced quota. It worked mechanically but admitted 0–2 Down calls per
Validation month and lost hits on net. Its overlay never fired. Up V3 made the
marginal slots worse.

The binding constraint is not the architecture — it is that a Down model with
Validation AUC 0.53 cannot outperform an Up pool whose marginal slots are
correct 63.5% of the time. Three different feature spaces (V2 absolute, V3
relative, V3 breakdown), three model classes (logistic, HGB, LightGBM), four
architectures (B/C/D/E), and two Up variants all converged on the same answer.
**The honest conclusion is that this data does not currently support an
independent Down selection path; zero out-of-pool Down replacements is the
supported choice, and `maximum_replacements: 0` should stay.**
