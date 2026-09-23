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

## Locked result (recorded 2026-09-23, reported as observed)

Phase B was rebuilt in a fresh `git archive` copy of commit `0972424`, with only the configuration
changes listed above. Before scoring, the integrity check passed: origins 120–266 of the rebuilt
directional, regime, and active artifacts match the frozen artifacts on all 6,349 rows, with 0
mismatches and a maximum score difference of 0.0. The locked window was then scored once with
`score_locked.py`. Output: `metrics/locked_result.json` and `metrics/locked_predictions.parquet`.

| Origins 268–315 (48 months) | Hits / calls | Accuracy | 6-month block bootstrap p5–p95 |
|---|---:|---:|---:|
| **Active model (Regime Adaptive Bidirectional)** | **492 / 807** | **60.97%** | 56.2% – 69.2% |
| Uptrend baseline, top 15 Up | 445 / 720 | 61.81% | 56.7% – 69.9% |
| Zero-feature rule, top 15 by 48-month Up-rate | 436 / 720 | 60.56% | 55.3% – 68.5% |
| Development reference (120–266) | 1,589 / 2,545 | 62.44% | — |

- Cap distribution: {15: 14, 16: 9, 17: 12, 18: 4, 19: 3, 20: 6}. The 15–20 contract held in every month.
- Down calls: 1, with 0 correct.
- The locked accuracy is 1.5 points below development and inside its bootstrap range. The model
  did not overfit its development windows in any material way. It also shows no measurable skill
  over the Uptrend baseline or the zero-feature prior rule: all three lie within 1.3 points of
  each other, and their bootstrap ranges overlap almost completely.

Descriptive breakdown, added after scoring and not used to change anything. By target year,
June–May:

| Year | Active accuracy | Mean universe breadth |
|---|---:|---:|
| 2022–23 | 46.4% | 0.468 |
| 2023–24 | 62.0% | 0.589 |
| 2024–25 | 61.8% | 0.594 |
| 2025–26 | 74.6% | 0.681 |

Seventeen of the 48 locked months were broad-down. With perfect market timing, the same picks
would have scored 83.5%. Monthly accuracy follows market breadth, which confirms the development
finding that the only large lever is market timing, and that it is not forecastable from this
data.

**The locked window is now consumed.** Any later change must be described as developed with these
outcomes known.
