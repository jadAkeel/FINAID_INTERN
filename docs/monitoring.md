# Monitoring plan

The local `python -m forecast_select monitor` command performs artifact-integrity checks only. It verifies that the locked audit, final model, and unscored production ledger exist; the production ledger is marked unscored, accepted count is at most 20, and `target_date` is absent.

When future monthly data arrive, append a new immutable ledger rather than overwriting June 2026. Recompute full-coverage and selective metrics by date, inspect correctness-LCB calibration, monitor missingness/stale runs and feature drift, and create a new audit version before changing selection policy. This is not a live production-readiness claim.
