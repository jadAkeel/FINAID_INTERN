# Reference baselines for matched comparison

Shared inputs for research experiments so every candidate is compared against the same numbers.

- `matched_baselines.json` — window definitions, the causal contract, and production hits/calls per window,
  both as delivered (15–20 variable cap) and restricted to ranks 1–15 on the same artifact rows.
- `production_cap_schedule.csv` — per-origin number of production calls (the cap a candidate must match
  for a coverage-matched comparison), plus production hits and Down calls per origin.
- `direction_labels_nonlocked.csv` — `d_s = 1[X_{s+1} > X_s]` for positions 0..267 only. Row `s` is
  **usable as a label at origin `t` only when `s <= t-2`**. Row `t` is the scoring target for origin `t`.
  No locked position (268–315) is present in this file.

Derived from `artifacts/active/regime_adaptive_predictions.parquet` and `data/monthly_indicators.xlsx`
on 2026-09-22. Read-only evidence; not a model input for production.
