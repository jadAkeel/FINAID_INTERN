# Reproduction runbook

Run from the repository root with the same Python environment used for the tests.

1. `python -m forecast_select audit` hashes the original Downloads inputs, verifies the workbook fingerprint, writes `reports/data_profile.json` and `reports/data_audit.md`.
2. `python -m pytest` runs unit, integration, leakage, and regression checks.
3. `python -m forecast_select backtest --models baselines` or `--models classical` creates an immutable development Level-A Parquet artifact and metrics table.
4. `python -m forecast_select catboost-full --chunk-size 8 --chunk-index N` runs one resumable CatBoost chunk. Repeat missing chunks, then run `python -m forecast_select catboost-full --chunk-size 8 --assemble`; assembly refuses missing/duplicate/error rows and only then creates `catboost_full_v2.parquet`.
5. `python -m forecast_select ensemble` creates the v2 pre-registered equal-weight Level-B artifact from available development components.
6. `python -m forecast_select level-c` performs per-origin Platt calibration, correctness modeling, block-bootstrap LCB, and cap-20 selection using earlier Level-B rows only.
7. Review the generated metrics and commit the frozen code/configuration before any new audit version.
8. `python -m forecast_select locked-audit` is the historical v1 command and must not be rerun or used for Level-C selection. Any future audit must use a new version.
9. `python -m forecast_select predict-month` creates the v2 June 2026 ledger with calibrated correctness fields and marks it `UNSCORED_JUNE_2026`.
10. `python -m forecast_select report` writes the final report from actual artifacts.

Artifacts are immutable by design: a rerun with changed code/configuration uses a new output name and audit version. Do not overwrite the source files or redesign after inspecting the locked audit.
