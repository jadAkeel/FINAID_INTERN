# Final delivery notes

## Delivery boundary

The repository is organized into three deliberate layers:

1. **Production:** `src/forecast_select/active_model.py`, the compact CLI,
   active configuration, active artifact, and public reports.
2. **Reproducible baseline and research:** model implementations, causal tests,
   registered artifacts, and the experiment registry. The Uptrend baseline writes
   `reports/uptrend_model_performance.json` and `.md`; the public
   `reports/model_performance.*` files describe only the active model. Down V2/V3
   findings and rejected promotion decisions are summarized in
   `docs/RESEARCH_OUTCOMES.md`. Negative results are kept because they prevent
   the same low-value experiments from being repeated.
3. **Local generated state:** `runs/`, Python caches, pytest/ruff caches, and
   replay caches. These are excluded from Git because they can be regenerated
   and do not support a delivery claim.

## Accuracy and evidence policy

- The active model is an explicit owner product decision, not a passed research
  promotion.
- Reported metrics come only from registered non-locked evaluation windows.
- Locked origins 268–315 remain isolated and must not be used for model tuning.
- The full history of accepted, rejected, superseded, and contaminated work is
  recorded in `docs/EXPERIMENT_REGISTRY.md` and
  `docs/SELECTION_GROUP_FAILED_REGISTRY.md`.

## Reproduction checklist

From a clean Python 3.11+ environment:

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src tests research/down_v2 research/down_v3
python -m pytest
python -m pytest research/down_v3/audit/test_reconciliation.py
python -m forecast_select build-uptrend-model
python -m forecast_select build-model
python -m forecast_select check-project
python -m forecast_select show-results
```

Do not treat generated output as verified until these commands pass on the
exact commit being delivered.
