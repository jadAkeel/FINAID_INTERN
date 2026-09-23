# Reference baselines for matched comparison

Shared inputs, so every research candidate is compared against the same numbers.

- `matched_baselines.json` — window definitions, the causal contract, and production's hits and
  calls per window. These are given as delivered (15–20 variable cap) and restricted to ranks
  1–15 on the same artifact rows.
- `production_cap_schedule.csv` — production's call count per origin. This is the cap a
  coverage-matched candidate must use. The file also holds production's hits and Down calls per
  origin.
- `direction_labels_nonlocked.csv` — `d_s = 1[X_{s+1} > X_s]`, indexed by the project's
  **1-based** `position` (as set by `forecast_select.io.load_workbook`), for positions 1–267.
  Row `s` may be used as a label at origin `t` only when `s <= t-2`; row `t` is origin `t`'s
  target. Position 267 is all-NaN because position 268 is never read. No locked position is
  present.

Built on 2026-09-22 and regenerated on 2026-09-23 from
`artifacts/active/regime_adaptive_predictions.parquet` and `data/monthly_indicators.xlsx`. The
first version of the label file used 0-based row numbers and included position 268's level; see
the erratum in `research/prior_only_selector/README.md`. Read-only evidence; not a production
input.
