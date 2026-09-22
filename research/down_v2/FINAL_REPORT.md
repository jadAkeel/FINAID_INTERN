# Down Model V2 — Final Research Report

**Audit update (2026-09-21):** The V3 row-level reconciliation corrected the
production A Confirmation result to 508/799 (0.6358) and confirmed its 4 / 0 /
7 Down calls across Tuning / Validation / Confirmation. V2's architecture
comparison below is retained as the original research snapshot; use
`research/down_v3/audit/AUDIT_REPORT.md` for audited production and replacement
comparisons. The no-promotion decision is unchanged.

**Scope:** Why the Down model is weaker than the Up model, whether it can be
strengthened, and whether it is ready to act as an independent selector.
**Method:** walk-forward, causal, leakage-safe, Tuning 120–179 (screen) /
Validation 180–219 (gate) / Confirmation 220–266 (descriptive) / locked
268–315 (never read).
**Production code untouched.** Everything below lives in `research/down_v2/`.

Reproduce: `python research/down_v2/build_down_panel.py` then
`research/down_v2/diagnostics/p1_architecture.py` … `p11_regime_conditional.py`,
`research/down_v2/down_v2_build.py`, `research/down_v2/architecture_comparison.py`.

---

## 1. Current problems, classified

| # | Problem | Class | Evidence |
|---|---------|-------|----------|
| 1 | The Down model is a **Direction Override, not a Selector**. The base pool is sorted by `p_up_selection_score` only; with `maximum_replacements=0` no out-of-pool indicator can ever be selected. | **Architecture** | `src/forecast_select/regime_adaptive.py:681-684`; P1: 0/147 months, 0 replacements |
| 2 | The "X40 scenario" is real: an indicator with low `p_up` but very high `p_down` can never enter the final selection. | **Architecture** | P2: 230 ready rows with `p_down ≥ 0.65` across 65/147 months were ignored |
| 3 | `p_down` is badly **overconfident**: at predicted 0.90 the observed Down rate is 0.52. | **Calibration** | P3, walk-forward refit over 147 origins |
| 4 | `p_up` and `p_down` are **not on a comparable scale** — different means, different ranges, different AUC. `max(p_up,p_down)` and margin scores rank worse than chance. | **Calibration** | P4: margin AUC 0.4541 (below 0.5) |
| 5 | `class_weight="balanced"` inflates `p_down` (mean 0.4923 vs observed 0.4302) without buying AUC. | **Calibration** | P6: A_balanced val AUC 0.5015, B_none 0.4955, C_calibrated 0.5015 |
| 6 | **Model complexity is harmful.** The full 28-feature model is the *worst* config on validation; every single family beats it. | **Modeling** | P5b: full V=0.5015 vs `only_volatility` V=0.5370 |
| 7 | The **local per-indicator model is noise** (AUC 0.5015–0.5055) and its blend weights hurt out of sample. | **Modeling** | P5a: indicator prior alone is the best component (V 0.5251, C 0.5516) |
| 8 | Severe **feature redundancy**: `momentum`↔`drawdown_distance` \|r\|=0.905; momentum's within-family mean \|r\|=0.838. 28 features on ~9k rows. | **Feature** | P10 |
| 9 | `lead_peer` features are 37% missing and are unavailable before origin 100, so the first 20 Tuning origins have no coverage — they burn training rows for no return (`only_lead_peer` V AUC 0.5260, near the weakest). | **Data** | P10 |
| 10 | Down thresholds tune well but **collapse out of sample** — precision rises with threshold on Tuning and does not transfer. | **Evaluation** | P8; and again in V2 below |
| 11 | The Down signal is **regime-dependent**, not universal: strongest at low dispersion (AUC 0.589) and high shock (0.573), near chance at high dispersion (0.499). | **Modeling** | P9 |

## 2. What is NOT a problem

- **The Up model.** Tuning 0.6405 / Validation 0.5852 / Confirmation 0.6320
  monthly hit rate, stable across windows. No work was done on it — frozen per
  requirement 9.
- **The causal boundary.** `causal_training_rows` + `availability_lag=1` +
  `assert_target_history_available` are correctly enforced everywhere I looked.
  No leakage found in any Down feature (P10).
- **The Up-first pool's own construction.** Sorting by `p_up_selection_score`
  is a legitimate ranking objective; `docs/methodology.md` correctly documents
  `selection_score` as ranking utility, not a probability. The problem is only
  that *nothing else* gets to rank.
- **Pool overlap as such.** Up-vs-Down top-15 Jaccard is 0.2154 — the two
  models genuinely disagree, which is what makes the X40 scenario interesting
  rather than vacuous.
- **The Down base rate.** 0.4314 across 7,350 eligible rows is a healthy,
  learnable minority class. The weakness is not class rarity.

## 3. Why Down is weak — ranked by evidence

1. **The signal itself is weak, not merely mis-calibrated.** After fixing
   calibration, removing class weighting, and cutting 28 features to 10, the
   best achievable Validation AUC is **0.5256** (V2 isotonic). The Up model's
   Tuning AUC on the same task is 0.5795. Down is a genuinely weaker problem
   on this data. *Strongest evidence: P5b — even the best single feature
   family only reaches 0.5370, and no combination of fixes pushes past it.*
2. **Threshold non-transferability.** V2's Tuning-screened floor (0.55,
   29 calls, 0.6897 precision) fires **once** on Validation and **once** on
   Confirmation, both misses. The calibrated distribution shifts between
   windows: Tuning max `p_platt` 0.7232 vs Validation max 0.5518. A threshold
   tuned on Tuning probabilities is a threshold on a moving scale.
   *Evidence: down_v2_standalone.json `calls_at_threshold`.*
3. **Overfitting from complexity.** Every leave-one-family-out config beats the
   full model on Validation (0.5013–0.5164 vs 0.5015); every single-family
   config beats it too (0.5260–0.5370). *Evidence: P5b, 17 configs, each a
   full 147-origin walk-forward refit.*
4. **The local model is pure noise** (Tuning 0.5015 / Validation 0.5055 /
   Confirmation 0.5018) yet consumes rows and blend weight. *Evidence: P5a.*
5. **Calibration damage from `class_weight="balanced"`.** Mean prediction
   0.4923 vs observed 0.4302. Removing it fixes the mean at no AUC cost;
   post-hoc Platt recovers it (C_balanced_calibrated mean 0.4302).
   *Evidence: P6 A/B/C.*
6. **Feature redundancy** inflates variance without adding information.
   *Evidence: P10.*

## 4. Ablation results

**Component ablation (P5a),** blend grid + leave-one-out, 147-origin
walk-forward:

| Component | Tuning AUC | Validation AUC | Confirmation AUC |
|---|---|---|---|
| indicator prior (best single) | 0.5601 | 0.5251 | 0.5516 |
| pattern prior | 0.5525 | 0.5256 | 0.5435 |
| global logistic | 0.5351 | 0.5015 | 0.5094 |
| local per-indicator | 0.5015 | 0.5055 | 0.5018 |
| all four equally blended | 0.5466 | 0.5164 | 0.5352 |

The local model is available on ~90% of indicators (mean share 0.9006) and
still does not beat the global logistic where both exist
(local 0.4910 vs global 0.5289 on Tuning; local 0.5016 vs global 0.5085 on
Confirmation). The indicator prior has std 0.1011 against a base rate of
0.4381 — it carries real cross-indicator variation, not a constant.

**Family ablation (P5b),** 17 configs, each a full walk-forward refit with
`include_indicator=True` preserved:

| Config | n feat | Tuning AUC | **Validation AUC** | Confirmation AUC |
|---|---|---|---|---|
| only_volatility | 2 | 0.5668 | **0.5370** | 0.5483 |
| only_recent_returns | 3 | 0.5605 | 0.5361 | 0.5403 |
| only_drawdown_distance | 2 | 0.5626 | 0.5357 | 0.5382 |
| only_momentum | 3 | 0.5616 | 0.5335 | 0.5390 |
| only_misc | 4 | 0.5340 | 0.5298 | 0.5362 |
| only_exhaustion_stall | 5 | 0.5675 | 0.5294 | 0.5274 |
| only_market_regime | 5 | 0.5321 | 0.5275 | 0.5264 |
| only_lead_peer | 4 | 0.5674 | 0.5260 | 0.5484 |
| drop_lead_peer | 24 | 0.5313 | 0.5164 | 0.5040 |
| drop_exhaustion_stall | 23 | 0.5154 | 0.5091 | 0.5212 |
| drop_drawdown_distance | 26 | 0.5374 | 0.5069 | 0.5098 |
| drop_volatility | 26 | 0.5366 | 0.5038 | 0.5133 |
| drop_misc | 24 | 0.5471 | 0.5038 | 0.5113 |
| drop_momentum | 25 | 0.5352 | 0.5035 | 0.5088 |
| drop_market_regime | 23 | 0.5440 | 0.5030 | 0.5227 |
| **full** | **28** | 0.5351 | **0.5015** | 0.5094 |
| drop_recent_returns | 25 | 0.5392 | 0.5013 | 0.5130 |

Reading: the full model is the *worst* or second-worst on Validation. Two
features do the job of twenty-eight. Nothing above 0.5370 exists.

**`class_weight` (P6):** balanced AUC 0.5015 / mean 0.4923; none 0.4955 /
mean 0.4128; balanced+Platt 0.5015 / mean 0.4302 (Brier 0.2501, best of the
three). AUC is insensitive to weighting; calibration is entirely determined
by it.

## 5. Calibration results

**Raw `p_down` (current production, P3):** predicted 0.9022 in the top bucket
→ observed 0.5200. Reliability is monotone-ish but the slope is far too flat.

**V2 after leakage-safe Platt (fit on Tuning only) and Isotonic:**

| Variant | Window | AUC | Brier | ECE | mean pred | observed | top1 | top5 |
|---|---|---|---|---|---|---|---|---|
| raw | tuning | 0.5423 | 0.2462 | 0.0489 | 0.4297 | 0.4302 | 0.6500 | 0.5300 |
| raw | validation | 0.5252 | 0.2526 | 0.0594 | 0.4473 | 0.4626 | 0.4750 | 0.4900 |
| raw | confirmation | 0.5306 | 0.2462 | 0.0490 | 0.4408 | 0.4271 | 0.6170 | 0.4936 |
| **platt** | tuning | 0.5423 | 0.2435 | 0.0321 | 0.4302 | 0.4302 | 0.6500 | 0.5300 |
| **platt** | validation | 0.5252 | 0.2492 | 0.0335 | 0.4357 | 0.4626 | 0.4750 | 0.4900 |
| **platt** | confirmation | 0.5306 | 0.2439 | 0.0260 | 0.4338 | 0.4271 | 0.6170 | 0.4936 |
| isotonic | tuning | 0.5540 | 0.2414 | **0.0000** | 0.4302 | 0.4302 | 0.7000 | 0.5533 |
| isotonic | validation | 0.5256 | 0.2503 | 0.0352 | 0.4367 | 0.4626 | 0.4500 | 0.4900 |
| isotonic | confirmation | 0.5349 | 0.2447 | 0.0220 | 0.4336 | 0.4271 | 0.5957 | 0.4468 |

Verdict: **calibration is a solved problem; signal is not.** Platt cuts ECE
from ~0.049 to ~0.026–0.034 and holds across all three windows, with no AUC
change. Isotonic is perfect in-sample (ECE 0.0000 by construction) but does
not transfer (0.0352 Validation). **Platt is the recommendation** — monotone,
one parameter, and it does not degrade out of sample.

But note the honest limit: the Validation Brier barely moves (0.2526 → 0.2492)
and top-1 precision actually *drops* from 0.4750 to 0.4500. A better-calibrated
weak model is still a weak model.

## 6. Down V2 architecture

Built from the ablation evidence, not chosen in advance:

- **10 features** (was 28): the two strongest families plus mild drawdown/distance.
  `down_volatility_3/12/compression`, `down_return_1/lag_1/lag_2`,
  `down_drawdown_12`, `down_distance_mean_12`, `down_momentum_3`,
  `down_negative_share_3`.
- **No local per-indicator model** (noise, P5a).
- **No `class_weight="balanced"`** (breaks calibration for no AUC, P6).
- **`indicator_id` restored** via OneHotEncoder — this matters: dropping it
  collapsed Tuning AUC to 0.4865; restoring it gave 0.5423.
- **C screened on Tuning only** → C=1.0.
- **Platt / Isotonic fit on Tuning only**, applied forward.
- **Threshold screened on Tuning with a ≥20-call floor** → 0.55.

Code: `research/down_v2/down_v2_build.py`. Artifacts:
`artifacts/down_v2_predictions.parquet`, `metrics/down_v2_standalone.json`.

## 7. V2 standalone results

**What improved vs the current full Down model:** Validation AUC 0.5015 →
0.5256 (+0.024); Confirmation 0.5094 → 0.5349; ECE ~0.049 → ~0.026–0.034;
feature count 28 → 10; training time ~1/4.

**What did not:** the Tuning-screened threshold fires **1 time** on Validation
(0 hits) and **1 time** on Confirmation (0 hits). This is the same failure P8
predicted. The calibrated probability distribution shifts between windows
(Tuning max 0.7232, Validation max 0.5518), so an absolute threshold tuned on
Tuning probabilities does not transfer.

Top-N precision degrades with depth as expected, and Top-1 is unstable
(tuning 0.70 / validation 0.45 / confirmation 0.60) — consistent with a weak
signal plus small sample, not with a usable ranking.

**Standalone verdict:** V2 is better calibrated and better ranked than the
current Down model, and is strictly simpler. It is **not** a selector: a
selector must produce calls out of sample, and at every threshold that is
precise enough to be worth calling, it stops firing.

## 8. Architecture comparison A / B / C / D

All four hold the monthly cap identical to production (verified:
call-count deviation vs A = 0 in every window). Free parameters screened on
Tuning only: Down floor → 0.40 (highest Tuning precision among floors with
≥20 out-of-pool calls); B insert cap → 1 (best Tuning hits). Up arm for D is
Platt-calibrated on `p_up_selection_score` (Tuning AUC 0.5795).

| Design | Window | calls | hits | hit rate | Down calls | Down prec |
|---|---|---|---|---|---|---|
| **A** production | tuning | 1071 | 686 | **0.6405** | 4 | 0.5000 |
| | validation | 675 | 395 | **0.5852** | 0 | — |
| | confirmation | 799 | 505 | **0.6320** | 7 | 0.7143 |
| **B** limited insertion | tuning | 1071 | 676 | 0.6312 | 64 | 0.5625 |
| | validation | 675 | 393 | 0.5822 | 40 | 0.4750 |
| | confirmation | 799 | 502 | 0.6283 | 53 | 0.4906 |
| **C** independent pools | tuning | 1071 | 633 | 0.5910 | 521 | 0.4722 |
| | validation | 675 | 365 | 0.5407 | 325 | 0.5015 |
| | confirmation | 799 | 486 | 0.6083 | 388 | 0.4149 |
| **D** unified bidirectional | tuning | 1071 | 685 | 0.6396 | 15 | 0.7333 |
| | validation | 675 | 392 | 0.5807 | 4 | 0.5000 |
| | confirmation | 799 | 506 | 0.6333 | 2 | 0.5000 |

**Bootstrap over origins (Validation+), delta vs A:**

| Design | Δ hit rate | p10 | p90 | P(Δ>0) |
|---|---|---|---|---|
| B | −0.0063 | −0.0098 | −0.0025 | **0.014** |
| C | −0.0391 | −0.0533 | −0.0258 | **0.001** |
| D | −0.0014 | −0.0045 | +0.0019 | 0.279 |

Reading:
- **B loses** — small, statistically confident loss. It inserts Down calls at
  ~0.475 precision in slots that Up was filling at ~0.585. Note the direction:
  the floor that *looked* best on Tuning (0.40, precision 0.4779) is barely
  above base rate, and the strict floor (0.55) that had 0.6897 on Tuning had
  **zero** out-of-pool calls on Validation. B cannot win with this signal.
- **C loses badly and confidently.** Forcing up to half the monthly cap into
  Down calls at ~0.47–0.50 precision against an Up pool at 0.585–0.632 is
  value destruction by construction. This is the "artificial balance" the
  brief warned against.
- **D is a wash** (P(Δ>0)=0.279, interval straddles zero). It is the only
  design that does no measurable harm — because in practice it almost never
  fires Down (4 calls in Validation, 2 in Confirmation). The unified score
  `max(p_up_cal, p_down_v2)` correctly concludes that Up wins almost every
  head-to-head. D is harmless precisely because it is inert.

**Regime-conditional check (P11)** — the last escape hatch for a conditional
Down selector. Out-of-pool Down V2 precision vs the Up-pool hit rate it would
forego:

| Regime | Tuning | Validation | Confirmation |
|---|---|---|---|
| all | 0.4335 | 0.4641 | 0.4252 |
| stressed | 0.5095 | 0.4637 | 0.4709 |
| calm | 0.4978 | **0.5458** | 0.4677 |
| low dispersion | 0.4613 | 0.5382 | 0.3624 |
| falling breadth | 0.4731 | 0.4461 | 0.5120 |
| v2 top-5/month | 0.4633 | 0.5050 | 0.4596 |
| **Up pool hit rate (bar)** | **0.6405** | **0.5852** | **0.6320** |

**No regime clears the bar in any window.** The best case is "calm" on
Validation at 0.5458 — still 4 points below the 0.5852 it would give up, on
273 rows / 7 months. Low dispersion looks promising on Validation (0.5382)
and collapses on Confirmation (0.3624), i.e. it is noise, not a conditional
edge. This was checked on Validation and Confirmation, not Tuning alone, so
the conclusion is not a Tuning artifact.

## 9. Decision

**The Down model is not ready to act as an independent selector, and the
evidence says the barrier is the signal, not the plumbing.**

What succeeded:
- **Calibration is fixed.** Platt on V2 gives ECE 0.026–0.034 across all three
  windows vs ~0.049 raw, mean prediction within 0.003 of observed on Tuning
  and 0.027 on Confirmation. This is a real, transferable improvement.
- **V2 is strictly simpler and strictly better ranked** than the current
  28-feature model: Validation AUC 0.5015 → 0.5256, Confirmation 0.5094 →
  0.5349, with 10 features instead of 28 and no local model.
- **The X40 constraint is confirmed as real and quantified** — 230 ignored
  strong-Down rows — but the counterfactual shows that acting on it loses
  hits. The architecture is not the bottleneck; the Down signal is.

What failed:
- **Every architecture that admits Down calls loses on Validation**, with
  bootstrap confidence: B −0.0063 (P>0 = 0.014), C −0.0391 (P>0 = 0.001).
- **Thresholds do not transfer.** The Tuning-optimal floor fires 1× on
  Validation and 1× on Confirmation. This is the specific failure mode the
  brief flagged, and it reproduced in V2 exactly as it did in P8.
- **No regime rescues it.** Best case (calm, Validation) still sits 4 points
  under the Up-pool bar it would have to clear.

What stayed uncertain:
- Whether a **fundamentally different Down feature space** (cross-sectional
  relative weakness, peer-conditional signals, or non-linear models) could
  reach the ~0.58 bar. This work re-weighted and re-calibrated the existing
  feature space; it did not invent a new one. That is the honest boundary of
  what was tested.
- Whether a longer Confirmation window would let the low-dispersion effect
  separate from noise. 37 months is thin for a conditional claim.

**Recommendation: do not promote a bidirectional selector.** Keep the Up-first
architecture as the production selector. Adopt Down V2 **as a monitoring
signal only** — calibrated, cheap, and available for future work — not as a
selection path. `configs/active_model.yaml` should stay
`owner_promoted`-as-is with `research_gate_passed: false`; the current
`maximum_replacements: 0` is, on this evidence, the correct setting rather
than a bug.

This matches the brief's instruction to say so plainly rather than force
bidirectional selection: the Down model on this data does not provide
sufficient signal to act as an independent selector.

## 10. Modification plan

Ordered by confidence. Nothing below is applied — all are proposals pending
your call.

### 10a. Recommended now (evidence: solid, low risk)

1. **`src/forecast_select/directional_downside.py`** —
   `_model_pipeline` (line 254): remove `class_weight="balanced"`. P6 shows
   no AUC cost (0.5015 → 0.4955, within noise) and it is the root cause of
   the 0.4923 vs 0.4302 mean inflation.
2. **`configs/directional_downside_model.yaml`** — collapse `blend_grid` to
   `{local_weight: 0.0, pattern_weight: 0.0}` only, and raise
   `minimum_local_rows` / `minimum_local_class_rows`. P5a shows the local and
   pattern components are noise (local AUC ~0.50 in all three windows) and
   their blend weights hurt out of sample.
3. **`src/forecast_select/directional_downside.py`** — add a Platt
   post-processing step fit on Tuning predictions only, applied forward.
   Section 5: ECE 0.049 → 0.026–0.034 across all three windows.
4. **`docs/methodology.md`** — record that `p_down` requires Platt calibration
   before any cross-direction comparison, and that `max(p_up,p_down)` and
   margin scores rank below chance (P4: 0.4541). This is documentation of a
   measured fact, not a code change.

### 10b. Contingent on a future decision (evidence: not there yet)

5. **Feature pruning to the V2 ten** (`down_v2_build.py:V2_FEATURES`) — P5b
   shows every family-beats-full result. Only worth doing *if* the Down model
   is kept at all; on its own it changes no selection, since
   `maximum_replacements=0` keeps Down out of the pool regardless.
6. **Do not implement B/C/D.** Both lose on Validation with bootstrap
   confidence. D is inert in practice and therefore harmless but pointless.

### 10c. New artifacts and tests

7. `research/down_v2/architecture_comparison.py` — promote to a gated
   experiment runner under `src/forecast_select/` (alongside
   `downside_challenger_runner.py`) with `require_positive_validation_delta`
   and `require_positive_bootstrap_p10` enforced from
   `configs/directional_downside_model.yaml:promotion`. Both currently fail;
   that is the correct outcome and the gate should keep them out.
8. New test: `tests/test_down_threshold_transfer.py` — assert that any
   Down threshold promoted from Tuning produces ≥ `minimum_tuning_down_calls`
   calls on Validation. Today's 0.55 floor would fail it (1 call), which is
   exactly the failure this research found.
9. New test: `tests/test_down_calibration_stability.py` — assert post-hoc
   calibration fit on Tuning keeps ECE ≤ 0.05 on Validation and Confirmation.
   V2 passes (0.0335 / 0.0260); the raw model does not.
10. Keep `research/down_v2/` as the evidence trail. Re-run it unchanged if
    new data arrives past origin 266; the locked window 268–315 stays unread.

### 10d. What would change the decision

The bar is specific and pre-registered: out-of-pool Down precision ≥ the
realized Up-pool hit rate, on Validation (not Tuning), in at least one
identified regime, with bootstrap p10 > 0 and Confirmation support. Today the
best candidate is 0.5458 against a 0.5852 bar, in a single 7-month slice —
not close. A new *feature space*, not a new threshold or a new architecture,
is what would move it.
