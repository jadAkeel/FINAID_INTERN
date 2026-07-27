# Reproduction runbook

Run from the repository root with the same Python environment used for the tests.

1. `python -m forecast_select audit` hashes the original Downloads inputs, verifies the workbook fingerprint, writes `reports/data_profile.json` and `reports/data_audit.md`.
2. `python -m pytest` runs unit, integration, leakage, and regression checks.
3. `python -m forecast_select backtest --models baselines` or `--models classical` creates an immutable development Level-A Parquet artifact and metrics table.
4. `python -m forecast_select ensemble` creates the pre-registered equal-weight Level-B artifact.
5. Review the generated metrics and commit the frozen code/configuration before `freeze`.
6. `python -m forecast_select freeze` writes the manifest and model registry. It must happen before the locked audit.
7. `python -m forecast_select locked-audit` runs the last 48 evaluable origins exactly once into `locked_audit_v1.parquet`.
8. `python -m forecast_select predict-month` creates the June 2026 ledger and marks it `UNSCORED_JUNE_2026`.
9. `python -m forecast_select report` writes the final report from actual artifacts.

Artifacts are immutable by design: a rerun with changed code/configuration uses a new output name and audit version. Do not overwrite the source files or redesign after inspecting the locked audit.

