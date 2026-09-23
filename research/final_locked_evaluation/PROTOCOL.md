# Final locked evaluation — pre-registration

Committed **before** any locked outcome was read. Authorized by the project owner on
2026-09-23 after the research program found no challenger that beats the active model
(see `research/oracle_headroom/README.md`).

## What is evaluated

The active **Regime Adaptive Bidirectional Selector** exactly as committed at the parent of this
commit. No parameter, threshold, weight, cap rule, feature, or configuration value is changed.
The locked window is origins **268–315**: 48 forecast months, with target months June 2022 to
May 2026.

## Method

The production chain has three stages. First, the Uptrend base: structured logistic, frozen
signed graph, and trailing-48 prior. Second, the directional-downside overlay. Third, the
regime-adaptive selection. Each stage is walk-forward, and its configuration ends at origin 266.
The run takes these steps:

1. Export the committed tree (`git archive`) into an isolated copy.
2. **Phase A — procedure proof, no new data.** In the copy, rebuild the directional and regime
   stages on the committed range 120–266. The only configuration difference is that the
   directional stage's base is `artifacts/active/uptrend_predictions.parquet`, the Uptrend
   artifact it was originally built on. Now that the active artifact is the regime model, the
   current configuration points elsewhere. The rebuilt artifacts must equal the frozen ones
   on every row: same accepted set, directions, ranks, and scores.
3. **Phase B — the locked run.** In a second copy, change only the end of the evaluated range
   (`confirmation_origins` 266 → 315) and move the `locked_origins` guard outside the workbook.
   All Tuning- and Development-based parameter selection is unchanged. The rebuilt origins
   120–266 must again be identical to the frozen artifact. If they are not, the pipeline uses
   future information, and that is reported as the finding instead of a score.
4. Score origins 268–315 once. Origin 267 is the buffer and is excluded, as in every other
   evaluation in this repository.

## Pre-registered outputs

Primary: the active model's hits / calls and accuracy on 268–315. Also reported: the monthly
cap distribution, Down calls and Down hits, and a six-month moving-block bootstrap
(5,000 replicates, seed `20260923`) of monthly accuracy.

Secondary comparators, computed on the same months:

- the Uptrend baseline's own top-15 Up selection, taken from the rebuilt base stage;
- the zero-feature rule: top 15 by trailing 48-label Up-rate, all Up;
- the development reference, 62.44% on non-locked origins 120–266.

## Commitments

- The result is reported whatever it is, including if it is below the development windows.
- No model, parameter, or report is revised in response to the locked result. After this
  run the locked window is **consumed**: any later change must be described as developed
  with the locked outcomes known.
- The sealed file `artifacts/audit/locked_evaluation.parquet` is not read. Its hash check in
  `check-project` is unaffected.

## Phase A result (recorded before Phase B)

Completed 2026-09-23. Rebuilt in an isolated `git archive` copy of this commit's parent:

| Artifact (origins 120–266) | Rows | Accepted / direction / rank mismatches | Max score difference |
|---|---:|---:|---:|
| Directional-downside predictions | 6,349 | 0 / 0 / 0 | 0.0 |
| Regime-adaptive predictions | 6,349 | 0 / 0 / 0 | 0.0 |
| Active artifact `artifacts/active/regime_adaptive_predictions.parquet` | 6,349 | 0 / 0 / 0 | 0.0 |

The procedure reproduces the active model exactly. Phase B may proceed.
