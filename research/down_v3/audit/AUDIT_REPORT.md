# Down V3 — Row-Level Reconciliation Audit

**Status:** COMPLETE. Every reported hit difference is now exactly reproducible
from row-level selected calls. Prior to this audit it was not.

**Scope:** research/evaluation code and artifacts only. No production code,
config, threshold, margin, cap, model, feature, calibrator, or regime logic was
modified. `src/`, `configs/`, and `artifacts/active/` are untouched (verified:
142/142 production tests pass unchanged).

**Owner:** Senior ML Research Auditor / Quantitative Engineer.

---

## 0. Executive summary

The headline contradiction is resolved, and it was **not** base-rate noise. It
was two metric-definition bugs in the research evaluation code:

1. **Wrong-way Down scoring.** `hits` was computed as `y_true.sum()` over all
   selected rows. `y_true` is the **Up-correct label**, so on a Down-direction
   row `y_true == 1` means the Down call was **wrong**. Every Down call was
   scored as a hit exactly when it failed.
2. **Invisible production Down calls.** `final_comparison.month_select` read
   `base.get("pred_dir", …)`, but `design_selections` never set `pred_dir`. The
   fallback was an all-`"Up"` series, so design A's real Down calls (4 tuning /
   0 validation / 7 confirmation) were reported as `0/0/0`, and — worse —
   already-Down base-pool rows became eligible as *victims* for external
   insertion.

Consequence for the reported numbers:

| | Reported (pre-audit) | Corrected (row-level) |
|---|---|---|
| A, Confirmation hits | 505 (0.6320) | **508 (0.6358)** |
| E, Validation hits | 397 (0.5881) | **391 (0.5793)** |
| E, Validation delta vs A | **+2 hits**, delta −4 (contradiction) | **−4 hits**, delta −4 (consistent) |
| E, Confirmation delta | −4 | **−5** |
| E, Confirmation Down calls | 13 | **20** (13 inserts + 7 production) |

**The contradiction was real.** `397` came from `377 + 34 − 14`: 377 correct Up
calls, plus 34 Down calls of which 14 were right, plus 20 that were wrong and
counted as hits. The true total is `377 + 14 = 391`, i.e. `395 − 4`. The
arithmetic identity `arch_hit_delta == replacement_delta` now holds with
**zero residual** for every design in every window.

**The production decision is unchanged in direction, but its evidence base is
corrected:** A still wins. E's Validation replacement delta is −4 (not +2),
Confirmation is −5, bootstrap p10 = −0.1088 (worse than the reported
−0.0408). Nothing promotes. The prior verdict "UNVERIFIED" is now resolved to
**"VERIFIED — no promotion."**

---

## 1. Metric definitions (authoritative)

`y_true` in the frozen artifact is the **Up-correct label**: `1` iff the
realized direction of `(indicator, target_date)` was Up. This is proven, not
assumed: on production-accepted rows, `y_true` mean is 0.6243 for Up-direction
calls and 0.3636 for Down-direction calls — a Down row with `y_true == 1` was a
wrong Down call.

```python
call_correct   = y_true                      if final direction == Up
               = 1 - y_true                  if final direction == Down

hits           = sum(call_correct)           # over the selected set
hit_rate       = hits / calls
down_calls     = # selected rows with final direction == Down     # "final_down_calls"
down_hits      = sum(call_correct) over down rows
down_precision = down_hits / down_calls
external_down_inserts = down rows whose indicator was OUTSIDE the Up base pool
overlay_flips        = base-pool rows flipped Up -> Down
victim_correct  = call_correct of the displaced row, under ITS OWN direction
replacement_delta = down_correct - victim_correct     # per event
arch_hit_delta  = challenger_hits - baseline_hits     # direction-aware
```

**`replacement_delta` is per event** (one external insert or one overlay flip),
not per origin and not per row-difference. Base-pool Down rows that production
already selected carry no victim and contribute **zero** to the delta.

**`hits` is computed on exactly the rows in the architecture table** — same
selected set, same `y_true`, same windows. There is no subset mismatch; the only
error was the scoring formula.

### 1.1 Semantic collision that caused two report sections to disagree

The pre-audit report used the name "down calls" for two different quantities:

- §9 external-insertion table: `down_calls` = final Down-direction rows
  (production Down rows **+** new inserts) → Confirmation **20**
- §15 final architecture table: `down` = external inserts only → Confirmation **13**

Both numbers are correct for their own definition. Neither was wrong; the name
was. The corrected report uses explicit names: `final_down_calls`,
`external_down_inserts`, `prod_down_rows_in_base`, with the identity
`final_down_calls = external_down_inserts + overlay_flips + prod_down_rows_in_base`,
asserted as Test 3.

---

## 2. Artifacts and provenance

### 2.1 Frozen production prediction artifact

`artifacts/active/regime_adaptive_predictions.parquet` — 6,349 rows × 154 cols.

| Field | Value |
|---|---|
| model_id | `regime_adaptive_selector` |
| model_version | `forward_breadth_graduated_15_to_20_v1` |
| feature_version | `structured_causal_features` |
| config_hash | `bbcbcb114f8f98f821490965842bf87ba32313da723d597fe66a4cf82a61953f` |
| data_hash | `8f9fc27ae0a33f4a25d1241b7d896b56ba7515d4ad7e984999f0c5fe42b20d29` |
| seed | `20260727` |
| run_id | `regime_adaptive_selector_research` |
| active_model / activation_status | `True` / `owner_promoted` |
| parameters_selected_on | `tuning_120_179` |
| source_experiment_artifact | `research/regime_adaptive_selector/artifacts/predictions.parquet` |
| fit_window | walk-forward, `<=position_118` … `<=position_314` (147 origins) |

Calibration versions: `p_up_calibrated` (production calibrator), plus research
Platt calibrators fit on Tuning only in each script. Windows: Tuning 120–179,
Validation 180–219, Confirmation 220–266. Locked origins 268–315 never read.

### 2.2 Selection artifacts (regenerated by this audit)

`research/down_v3/artifacts/final_sel_{A..G}.parquet` — 2,545 rows each, one row
per selected call. Columns added by the audit: `final_direction`,
`call_correct`, `victim_dir`, `victim_correct`.

`research/down_v3/artifacts/v3_selections_{v2,v3}.parquet` — same correction.

### 2.3 Audit artifacts (new)

| Path | Contents |
|---|---|
| `audit/replacement_ledger.parquet` | one row per replacement event |
| `audit/architecture_reconciliation.csv` | per-origin reconciliation |
| `audit/set_classification_by_category.csv` | A-vs-E set-diff categories |
| `audit/corrected_window_table.csv` | corrected per-window metrics |
| `audit/audit_findings.json` | machine-readable summary |
| `audit/test_reconciliation.py` | 10 regression tests |
| `audit/backup_pre_audit/` | pre-audit scripts, metrics, artifacts |

### 2.4 Provenance gap found and fixed

`final_comparison.month_select` built each inserted row from an explicit
dict of ~11 columns. That dict **dropped** `regime_cap`, `origin_date`,
`target_date`, `config_hash`, `model_version`, `data_hash`, `run_id`,
`fit_window`, and 130 other provenance columns. The 48 inserted rows in the
pre-audit artifacts had `regime_cap = NaN`. Fixed: inserted rows now carry the
full candidate row. This is why the ledger's `monthly_cap` had to be backfilled
from sibling rows when reconstructing the pre-audit state.

---

## 3. Validation discrepancy: 395 vs 397 vs replacement delta −4

### 3.1 The exact reconstruction, from `final_sel_E.parquet` rows

Validation window (origins 180–219), all 675 calls, identical `y_true`:

```
A hits (direction-aware)        = 395
E: Up rows 641, Up hits         = 377
E: Down rows 34, Down hits      = 14
E total hits (direction-aware)  = 377 + 14 = 391
E reported hits (y_true sum)    = 377 + 34 − 14 = 397   ← BUG
replacement delta               = 14 − 18 = −4
identity check                  = 395 + (−4) = 391 ✓
```

The reported `397` decomposes as **377 correct Up calls + 20 wrong Down calls
counted as hits + 14 correct Down calls**. The bug added the 20 failures and
dropped the 14 successes relative to the correct 391. The `+2` was `20 − 14 + …`
— pure scoring error, zero statistical content.

### 3.2 Why "16 of 34 displaced victims were wrong anyway" cannot explain it

The pre-audit report offered: *"the total hit rate still rose slightly (397 vs
A's 395) because 16 of the 34 displaced Up victims would have been wrong
anyway. The hit-rate metric is flattered by base-rate noise."*

This is not an explanation, it is a coincidence. The 34 displaced victims were
**removed from E's selected set** — their outcomes cannot add to E's hits. A
removed row contributes zero to either design's total. The only quantity that
can change E's hit count is the *difference* between the inserted Down call and
the displaced Up call, which is precisely the replacement delta, which is −4.
Under identical rows and identical `y_true`, the total-hit difference is an
arithmetic constant, not a sample statistic. The report's §9 narrative has been
removed.

### 3.3 The identity, now asserted as a test

```
arch_hit_delta = replacement_delta + overlay_flip_delta
                 + up_ranking_delta + residual
```

For E/Validation: `−4 = −4 + 0 + 0 + 0`. For every design and window,
`residual == 0` (Test 1, Test `test_architecture_reconciliation_decomposition_json`).

---

## 4. Confirmation discrepancy: 20 vs 13 Down calls, −5 vs −4

Two independent causes, both real:

**Cause 1 — name collision (§1.1).** `v3_selection.py` scored base rows with
`predicted_direction` and counted production's 7 Down-direction rows; 7 + 13
inserts = 20. `final_comparison.py` had the `pred_dir` bug and saw 0 production
Down rows, so it reported 13. **Both were correct under their own definitions.**
With explicit names the two tables now agree row-for-row.

**Cause 2 — a real scoring bug, worth 1 hit.** Because `pred_dir` was missing,
already-Down base-pool rows were treated as ordinary Up victims. At **origin
226** the engine inserted X12 and displaced **X22** — but X22 was itself a
production Down call, `y_true = 0`, so it was **correct** (`victim_correct = 1`).
The engine recorded `victim_y = 0` and scored the replacement as `+1 − 0 = +1`
when the true replacement value was `+1 − 1 = 0`. Confirmation delta was
reported as −4; the correct value is **−5**.

After the fix, already-Down rows are structurally excluded from victim
candidacy (`remaining = base[~base["is_down_call"]]`), and Test 2 asserts that
no victim ever has `victim_dir == "Down"`.

---

## 5. Tuning discrepancy: 5 vs 1 Down calls

Same name collision. In `v3_selection.py`, Tuning reported 5 Down rows = **4
production Down rows + 1 external insert**. In `final_comparison.py`, the
`pred_dir` bug hid the 4 production rows, so it reported 1. The **delta of +1
is identical in both** because the 4 production rows carry no victim and
contribute zero to the replacement delta.

Corrected: `final_down_calls = 5` (4 production + 1 insert), `replacement_delta
= +1` on 1 event. Frozen screen parameters are unchanged: V2 margin 0.08 /
floor 0.65, V3 margin 0.00 / floor 0.55.

---

## 6. Production A "0/0/0 Down" semantic mismatch

The V2 report's design A correctly showed 4 / 0 / 7 Down calls. The V3 report's
§15 showed 0 / 0 / 0 — not because production changed, but because
`final_comparison.design_selections` never propagated `pred_dir`. Design A was
silently redefined from "production" to "production with its Down overlay
erased."

Verified against the frozen artifact: production's accepted set contains
**11 Down-direction rows** — 4 in Tuning, 0 in Validation, 7 in Confirmation.
Design A's selected set is set-equal to production's accepted set in **all 147
origins** (Test 7), and now reports 4 / 0 / 7.

This matters beyond cosmetics: A's Confirmation hits were understated as 505
when production actually delivered **508** (0.6358, not 0.6320). Production is
3 hits better than the report claimed.

---

## 7. Per-origin reconciliation

`audit/architecture_reconciliation.csv` — 882 rows (6 challengers × 147
origins). Columns: `baseline_calls/hits`, `challenger_calls/hits`,
`direct_hit_delta`, `sum_replacement_delta`, `unexplained_delta` where
`unexplained_delta = direct_hit_delta − sum_replacement_delta`.

| Design | Origins with unexplained ≠ 0 | Σ direct | Σ replacement | Σ unexplained |
|---|---|---|---|---|
| B | **0** | 0 | 0 | 0 |
| C | **0** | 0 | 0 | 0 |
| D | **0** | 0 | 0 | 0 |
| E | **0** | −8 | −8 | 0 |
| F | 98 | −51 | 0 | −51 |
| G | 98 | −66 | −15 | −51 |

**B, C, D, E reconcile exactly at every origin.** F and G show non-zero
`unexplained_delta` for a legitimate and important reason: they **replace the
Up ranking arm** (Up V3 instead of current Up), which is a selection change
outside the replacement ledger. This is the `other_selection_delta` the brief
asks about, and it is now measured rather than assumed — `final_comparison.py`
re-runs each design with both Down paths disabled and differences the result
against A:

| Design / Validation | arch | replacement | up-ranking | residual |
|---|---|---|---|---|
| B / C / D | 0 | 0 | 0 | **0** |
| E | −4 | −4 | 0 | **0** |
| F | −4 | 0 | −4 | **0** |
| G | −14 | −10 | −4 | **0** |

---

## 8. Selected-set comparison, A vs E

`audit/set_classification_by_category.csv`. Across all 147 origins:

| Category | Count | Σ baseline hits | Σ challenger hits | Σ delta |
|---|---|---|---|---|
| same indicator, same direction | 2,497 | 1,560 | 1,560 | **0** |
| same indicator, direction differs | **0** | — | — | 0 |
| in A only | 48 | 29 | — | — |
| in E only | 48 | — | 21 | — |

**This is the answer to §8 of the brief.** E differs from A in **exactly 48
rows**, all of them the 48 recorded external inserts. There is:

- **no re-ranking** of remaining Up calls (2,497 shared rows, identical
  direction, identical correctness, Σ delta = 0),
- **no direction change on any shared indicator** (0 disagreements),
- **no cap change** — every design fills its per-origin `regime_cap` exactly,
  in all 147 origins (Test 4),
- **no dropped or extra calls** — 2,545 rows in every design, 675 / 799 / 1,071
  per window,
- **no duplicate selections** (Test 5),
- **no unrecorded overlay flip** — `overlay_flips = 0` everywhere, consistent
  with §7–§8 of the report,
- **no change to `accepted`, rank cut, missing-data eligibility, calibrated Up
  score, candidate universe, or baseline artifact.**

E changes one thing and one thing only: 48 out-of-pool insertions. Every hit
difference is attributable to them.

---

## 9. Architecture B / C / D identity

Asserted exactly, not just on totals (Test 6):

```python
set((origin, indicator_id, final_direction)) for B, C, D == set(...) for A
```

2,545 / 2,545 rows match on `(origin, indicator, direction)` for all three.
The report's claim "B, C and D are exactly A" is **confirmed at row level**.
The reason stands: with the Tuning-screened margin/floor, V2 admits zero
out-of-pool calls and the V3 overlay never fires, so the architectures are
behaviourally identical on this data.

---

## 10. Corrected architecture table

All designs: 15–20 calls (frozen per-origin `regime_cap`), 0–5 Down dynamic, no
forced quota. Parameters screened on Tuning, frozen, Validation as gate,
2,000-replicate origin-block bootstrap vs A. **Direction-aware scoring.**

| Design | Window | Calls | Hits | Rate | Final Up | Final Down | Down prec | Flips | Ext. inserts | Prod Down | Δ hits vs A | Event Δ | Reconciled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** prod | T | 1071 | 686 | .6405 | 1067 | 4 | .5000 | 0 | 0 | 4 | 0 | 0 | ✓ |
| | V | 675 | 395 | .5852 | 675 | 0 | — | 0 | 0 | 0 | 0 | 0 | ✓ |
| | C | 799 | **508** | **.6358** | 792 | 7 | .7143 | 0 | 0 | 7 | 0 | 0 | ✓ |
| B/C/D | T/V/C | = A | = A | = A | 1067/675/792 | 4/0/7 | .5000/—/.7143 | 0 | 0 | 4/0/7 | 0 | 0 | ✓ |
| **E** | T | 1071 | 687 | .6415 | 1066 | 5 | .6000 | 0 | 1 | 4 | **+1** | +1 | ✓ |
| | V | 675 | **391** | **.5793** | 641 | 34 | .4118 | 0 | 34 | 0 | **−4** | −4 | ✓ |
| | C | 799 | **503** | **.6295** | 779 | 20 | .5500 | 0 | 13 | 7 | **−5** | −5 | ✓ |
| **F** | T | 1071 | 659 | .6153 | 1046 | 25 | .5200 | 0 | 0 | 25 | −27 | 0 (+up-rank −27) | ✓ |
| | V | 675 | 391 | .5793 | 646 | 29 | .5172 | 0 | 0 | 29 | −4 | 0 (−4) | ✓ |
| | C | 799 | 488 | .6108 | 773 | 26 | .3077 | 0 | 0 | 26 | −20 | 0 (−20) | ✓ |
| **G** | T | 1071 | 662 | .6181 | 1015 | 56 | .6250 | 0 | 31 | 25 | −24 | +3 (−27) | ✓ |
| | V | 675 | 381 | .5644 | 570 | 105 | .4857 | 0 | 76 | 29 | −14 | −10 (−4) | ✓ |
| | C | 799 | 480 | .6008 | 742 | 57 | .3509 | 0 | 31 | 26 | −28 | −8 (−20) | ✓ |

For F and G, "Event Δ" shows `replacement + (up-ranking)`; both sum to the
Δ-hits column with residual 0.

**Corrected bootstrap vs A** (Validation+ origin-level, direction-aware, same
frozen seed 20260921):

| Design | Δ hit rate | p10 | p90 | P(Δ>0) |
|---|---|---|---|---|
| B / C / D | 0.0000 | 0.0000 | 0.0000 | 0.000 |
| **E** | **−0.0544** | **−0.1088** | −0.0068 | 0.064 |
| F | −0.3469 | −0.4898 | −0.2041 | **0.002** |
| G | −0.4490 | −0.6122 | −0.2925 | **0.000** |

The pre-audit bootstrap for E was `−0.0068 / −0.0408 / +0.0272 / 0.363` — the
same wrong-way-Down scoring inflated both the mean and the upper tail. E's
corrected P(Δ>0) is **0.064**, not 0.363.

### 10.1 Replacement delta ledger (E, all 48 events)

| Window | Events | Down hits | Victim hits | Delta |
|---|---|---|---|---|
| Tuning | 1 | 1 | 0 | **+1** |
| Validation | 34 | 14 | 18 | **−4** |
| Confirmation | 13 | 6 | 11 | **−5** |

Σ = −8 = E's total hit difference vs A across all windows. `sum(replacement_delta)
= arch_hit_delta` exactly.

---

## 11. Bugs and defects found

| # | Severity | Location | Defect | Fix |
|---|---|---|---|---|
| 1 | **Critical** | `final_comparison.summarize`, `v3_selection.summarize` | `hits = y_true.sum()` scores a Down call as a hit iff it was wrong | `call_correct`, direction-aware |
| 2 | **Critical** | `final_comparison.design_selections` / `month_select` | `pred_dir` never propagated → A's Down calls hidden, reported 0/0/0; Down rows eligible as victims | set `pred_dir` from `predicted_direction`; exclude Down rows from victim pool |
| 3 | **High** | `month_select` insert constructor | inserted rows carried 11 of 154 columns; `regime_cap`, `config_hash`, `model_version`, dates all NaN | carry the full candidate row |
| 4 | **High** | `month_select` victim scoring | `victim_y` used the Up label even when the victim was itself Down-direction (origin 226: +1 scored, true value 0) | `victim_correct` under the victim's own direction |
| 5 | **Medium** | report §9 narrative | "+2 hits explained by base-rate noise / 16 wrong victims" — removed row cannot add to a hit count | narrative deleted; §3.2 of this audit |
| 6 | **Medium** | report §9 vs §15 | one column name ("down") for two quantities (final Down rows vs new inserts) | explicit `final_down_calls` / `external_down_inserts` |
| 7 | **Low** | `v3_selection.summarize` | `replacement_delta` divided by `len(d)` (all Down rows) rather than by event count | divide by event count |
| 8 | **Low** | `final_comparison.bootstrap` | baseline `a_sel` built before the loop but designs iterate in dict order; A is first, so correct by luck | unchanged, now covered by Test 7 |

**No stale artifacts.** Both scripts were re-run as-is before any edit and
reproduced their committed JSON **byte-identically** (see
`audit/rerun_*_as_is.log`). The numbers in the report were faithful to the code;
the code was wrong about the data. This is why the discrepancy survived review —
re-running could not reveal it.

**No leakage.** All calibrators (Up Platt, Down V2/V3 Platt) are fit on Tuning
rows only and frozen before Validation. Windows are disjoint. Locked origins
268–315 are absent from the artifact's selected set.

---

## 12. Tests added

`research/down_v3/audit/test_reconciliation.py` — 10 tests, all passing
(2.65 s). Also verified the full production suite still passes: **142 passed**.

| # | Test | Asserts |
|---|---|---|
| 1 | `test_architecture_reconciliation` | for B/C/D/E: `arch_hit_delta == replacement_delta` and `other_selection_delta == 0`, every window |
| 1b | `..._decomposition_json` | for A–G incl. F/G: `residual == 0` in the recorded decomposition |
| 2 | `test_replacement_ledger` | per event `delta == challenger_correct − victim_correct`; **no victim is Down-direction** |
| 3 | `test_final_down_calls_semantics` | `final_down_calls == #rows with direction==Down == inserts + prod_down_rows` |
| 3b | `test_hits_are_direction_aware` | `hits == Σ (y_true if Up else 1−y_true)` |
| 4 | `test_call_count_invariance` | every design fills `regime_cap` exactly, per origin |
| 5 | `test_no_duplicate_selections` | no `(origin, indicator)` appears twice |
| 6 | `test_bcd_identical_to_a` | exact `(origin, indicator, direction)` set equality, not just hit rate |
| 7 | `test_design_A_is_production` | A == production accepted set in all 147 origins, incl. its 11 Down rows |
| 8 | `test_e_validation_identity_is_exactly_negative_four` | pins 395 / 391 / 14 / 18 / −4 |

Test 8 is the regression pin for the specific contradiction in the brief: if
wrong-way Down scoring ever returns, E's Validation total becomes 397 and this
test fails.

---

## 13. Corrected V2 vs V3 decision

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
| **Confirmation replacement delta** | **0** | **−5** | **V2** |

**Unchanged conclusion, now on correct arithmetic:** V3 is the better
standalone Down model; V2 is the safer Down policy. V3's higher AUC converts
into worse replacement value out of sample. Standalone model quality and
replacement value are different quantities, and the V3 report's central claim
survives the audit intact — in fact it is strengthened, since the corrected
Confirmation delta (−5) is worse than the reported (−4), and the corrected
bootstrap p10 (−0.1088) is worse than the reported (−0.0408).

---

## 14. Final architecture decision

**A — current production.** Every gate in the promotion rule is evaluated on
corrected numbers:

| Criterion | Requirement | E (corrected) | Verdict |
|---|---|---|---|
| Validation direct hit delta | > 0 | **−4** | FAIL |
| Replacement/event delta consistency | identical | −4 = −4, residual 0 | PASS |
| Bootstrap p10 | ≥ 0 | **−0.1088** | FAIL |
| Confirmation agreement | no reversal | −5, same direction | PASS (negative) |
| Sample size | sufficient | 34 V / 13 C events | PASS |
| Leakage | none | Tuning-only calibration | PASS |
| Unexplained selection changes | none | 0 (48/48 attributed) | PASS |

Two of seven gates fail. **Down V3 does not promote.** Neither does V2 (0
Validation replacement delta, not > 0).

The winning production policy has a **15–20 total-call cap**, with 4 / 0 / 7
Down calls in Tuning / Validation / Confirmation and zero out-of-pool Down
replacements. The tested V3 0–5 policy admitted 0 Down in 10 of 40 Validation
months and never more than 2. `maximum_replacements: 0` stays.

**Correction to the pre-audit report's evidence, not its conclusion:** A's
Confirmation accuracy is 0.6358 (508/799), not 0.6320 (505/799). Production is
slightly better than previously stated. F and G are materially worse than
previously stated (Δ −0.35 / −0.45 rather than −0.27 / −0.35), making the case
against replacing the Up ranking stronger, not weaker.

---

## 15. Whether any production change is justified

**No.** No production code, config, or artifact was modified, and none is
warranted:

1. `configs/regime_adaptive_selector.yaml` — `maximum_replacements: 0` stays.
   It is what kept the system at 0.5852 / 0.6358 while every alternative lost
   hits on corrected arithmetic.
2. `configs/active_model.yaml` — stays `owner_promoted` /
   `research_gate_passed: false`.
3. The pre-registered promotion spec in the report's §19 remains valid as a
   future gate. The new test file is its enforcement: any margin/floor that
   scores a negative Validation replacement delta or a bootstrap p10 < 0 fails
   it, and today's V3 fails on both.
4. Re-run this study unchanged when ~24 more origins are available. The
   pre-registered bar — Validation replacement delta > 0, bootstrap p10 ≥ 0,
   Confirmation agreement, ≥20 Down calls — is unchanged. V3 today is −4 at 34
   Validation calls and −5 at 13 Confirmation events.

---

## 16. Verification status

The property demanded by the brief is now established:

> **Every reported hit difference is exactly reproducible from row-level
> selected calls.**

Evidence: 10/10 reconciliation tests pass; residual 0 for all 7 designs × 3
windows; B/C/D/E reconcile at all 147 origins individually; the A-vs-E
difference is exactly the 48 ledger events and nothing else; the full
production suite (142 tests) passes unchanged.

The prior status **UNVERIFIED** is resolved to **VERIFIED — no promotion**.
Production remains exactly as-is.

---

## Appendix A — reproduction

```bash
# 1. reproduce the audit from the frozen artifacts
python research/down_v3/audit/reconcile_audit.py

# 2. regenerate the corrected selection artifacts and metrics
python research/down_v3/v3_selection.py
python research/down_v3/final_comparison.py   # exits 1 if reconciliation fails

# 3. run the reconciliation tests
python -m pytest research/down_v3/audit/test_reconciliation.py -q

# 4. confirm production is untouched
python -m pytest tests/
```

`final_comparison.py` now ends with a hard gate: if any design/window yields a
non-zero residual, it prints the decomposition and exits 1.

## Appendix B — files changed by this audit

| File | Change |
|---|---|
| `research/down_v3/final_comparison.py` | `call_correct`; `pred_dir` propagation; full-row inserts; `victim_correct`; explicit metric names; decomposition + exit-1 gate |
| `research/down_v3/v3_selection.py` | same `call_correct`/`victim_correct` corrections; direction-aware bootstrap baseline; event-count delta denominator |
| `research/down_v3/audit/` | new: `reconcile_audit.py`, `test_reconciliation.py`, `replacement_ledger.parquet`, `architecture_reconciliation.csv`, `set_classification_by_category.csv`, `corrected_window_table.csv`, `audit_findings.json`, `backup_pre_audit/` |
| `research/down_v3/artifacts/final_sel_*.parquet`, `v3_selections_*.parquet` | regenerated with `final_direction`, `call_correct`, `victim_dir`, `victim_correct` and restored provenance columns |
| `research/down_v3/metrics/v3_selection.json`, `final_comparison.json` | regenerated with corrected metrics + decomposition block |
| `research/down_v3/FINAL_REPORT.md` | corrected tables and narrative (this document supersedes §9, §11, §15, §16) |

`src/`, `configs/`, `artifacts/active/`, `reports/`: **unchanged**.
