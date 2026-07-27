# Model card: Level-C calibration/reliability v2

Level-C is a development-only selective policy. At each origin it fits Platt calibration and a correctness Logistic model from strictly earlier Level-B rows. A contiguous-month bootstrap estimates a conservative residual correction; the 10th percentile correction is added to the current correctness probability and clipped to `[0, 1]`.

Policy: reliability floor `0.55`, hard monthly cap `20`, soft target `15` with no quota filling. Rows below the floor or beyond the cap are rejected with explicit reasons.

Evidence: `artifacts/oof_predictions/dev_level_c_v2.parquet` and `reports/tables/level_c_dev_metrics.csv`.

This has not been evaluated on `locked_audit_v1`; that artifact remains immutable and was not used for selection. A future final claim requires a separately frozen audit version.

