# Research outcomes

## Accuracy ceiling (2026-09-23)

This is the finding that frames every other result. At the active model's own call count
and universe, over all 147 non-locked months
([`research/oracle_headroom/`](../research/oracle_headroom/README.md)):

| | Accuracy |
|---|---:|
| Active model | **62.44%** |
| Hindsight oracle that knows every indicator's true long-run Up-rate | 62.71% |
| Hindsight oracle that also knows how drift shifted in each window | 64.36% |
| Active model's picks with perfect market-direction timing | 77.13% |

- **Indicator choice is solved.** The model is within 0.27 points of the long-run drift
  oracle. Better base-rate estimators cannot add more than about 0.3 points. The registry's
  window, shrinkage, ensemble, and seasonal variants all land within noise, which is
  consistent with this.
- **The model stack is real, not a base-rate rule.** Paired at the model's own call count,
  ranking by its prior alone loses 13 hits on Tuning and 13 on Confirmation, with block-bootstrap
  intervals below zero, and gains 1 on Validation
  ([`research/seasonal_prior/`](../research/seasonal_prior/README.md), control arm).
- **The remaining headroom is market timing, and it is not forecastable here.** 54 of 147 months
  are broad-down months. The model's breadth forecast scores AUC 0.52 for them, and realized
  breadth autocorrelation is −0.04. A month-level Down signal would need 61% precision to break
  even against a 36.7% base rate. That is why Down calls are rare (11 in 147 months, 7 correct)
  and why every Down-replacement design below lost hits.
- **There is no calendar seasonality.** The existing `direction_lag_12` feature sits two months
  off the target's season, so this was tested directly in a pre-registered experiment. Split-half
  reliability between even and odd years was r = 0.006, and the overlay lost hits in all three
  windows.

On anonymous price histories alone, about 62–64% is the ceiling. A future claim of a large gain
should first show breadth-forecasting skill above AUC 0.5.

## Down V2/V3

### Delivery decision

Keep the owner-promoted Regime Adaptive selector with its current Up-first
ranking and `maximum_replacements: 0`. Down V2 and V3 remain research outputs;
neither is approved as a production selector. This decision concerns the
*marginal replacement value* of Down calls, not whether Down has any predictive
signal. Production can still make Down direction overrides inside its Up-ranked
base pool: the audited counts are **4 / 0 / 7** in Tuning / Validation /
Confirmation. It makes **zero out-of-pool replacements**.

| Design | Validation | Confirmation | Evidence |
|---|---:|---:|---|
| Production A | 395/675 (58.52%) | 508/799 (63.58%) | Audited direction-aware selected calls |
| V3 limited insertion E | 391/675 (57.93%) | 503/799 (62.95%) | −4 and −5 hits vs A, both reconciled to replacements |
| V3 Up-ranking change F | 391/675 | 488/799 | Up ranking alone loses 4 / 20 hits |
| V3 combined G | 381/675 | 480/799 | Largest loss among tested alternatives |

Source: `research/down_v3/metrics/final_comparison.json` and
`research/down_v3/audit/AUDIT_REPORT.md`. The V3 audit corrected an earlier
direction-scoring error: Confirmation production had **508**, rather than 505,
hits. It also corrected the earlier report's claim that production had no Down
calls. The row-level reconciliation suite covers all 147 origins and reports
zero residual for the production-to-E replacement decomposition.

### What the research delivered

- **Calibration:** Raw Down scores were overconfident (predicted 0.90 in the
  highest bucket, observed Down rate 0.52). V2 Platt calibration reduced
  expected calibration error to about 0.026–0.034 across the three windows.
  This makes Down probabilities more interpretable, but does not make their
  rankings strong enough to displace Up calls.
- **Simpler candidate:** V2 reduced the Down feature set from 28 to 10, removed
  the noisy local model, and improved standalone Validation AUC from 0.5015 to
  0.5256 (isotonic variant). This is a research candidate, not a production
  change. Its Tuning-screened 0.55 threshold fired only once in Validation and
  once in Confirmation, both misses.
- **New feature space:** V3 tested relative weakness and breakdown features.
  Its best standalone Validation AUC was 0.5314, above V2's raw 0.5252, but
  V3 insertion lost four Validation hits. Standalone AUC is therefore not a
  sufficient promotion criterion.
- **Architecture quantified:** V2 found 230 eligible high-Down rows outside
  the Up-ranked base pool across 65 months. V3 tested admitting them under a
  0–5 monthly cap. The mechanism enforced the cap and produced 0, 1, or 2 Down
  calls per Validation month; its selected replacements were worse than the Up
  slots they displaced.
- **Audit value:** V3's direction-aware scorer, replacement ledger, and
  reconciliation tests turn the production comparison into a traceable result
  and prevent the earlier Up-label scoring error from silently returning.

### Failed promotion gates and limits

V3's Validation replacement delta was **−4 at 34 Down calls**, with
Confirmation **−5**. The promotion gate requires a positive Validation delta,
bootstrap p10 at least zero, Confirmation agreement, and at least 20 Down
calls. V2 generated no new Validation replacements under its screened policy.
The tested Down versions therefore fail the decision gate even though some
standalone AUC and calibration metrics improved.

The windows used here are Tuning origins 120–179, Validation 180–219, and
Confirmation 220–266. Small Down-call counts limit claims about conditional
regimes. The separate March–May 2026 terminal holdout described in the
experiment registry was inspected by prior research and is consumed; it cannot
be reused as a fresh promotion gate. No new locked-period result is claimed in
these reports.

### Reproduce and verify

From the repository root, the focused audit can be reproduced from the saved
artifacts with:

```bash
python research/down_v3/audit/reconcile_audit.py
python research/down_v3/v3_selection.py
python research/down_v3/final_comparison.py
python -m pytest research/down_v3/audit/test_reconciliation.py -q
ruff check research/down_v2 research/down_v3
```

The full research rebuild is compute intensive. Run V2's
`build_down_panel.py`, diagnostics P1–P11, `down_v2_build.py`, then
`architecture_comparison.py`; run V3's `build_v3_panel.py`, `v3_ablation.py`,
`v3_models.py`, `up_v3.py`, `v3_selection.py`, then `final_comparison.py`.
Detailed protocols and tables are in `research/down_v2/FINAL_REPORT.md` and
`research/down_v3/FINAL_REPORT.md`. Undefined precision or delta metrics with
zero eligible calls are encoded as JSON `null` in the delivered metric files.
