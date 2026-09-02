# Final delivery notes

## Delivery boundary

The repository is organized into three deliberate layers:

1. **Production:** `src/forecast_select/active_model.py`, the compact CLI,
   active configuration, active artifact, and public reports.
2. **Reproducible baseline and research:** model implementations, causal tests,
   registered artifacts, and the experiment registry. Negative results are kept
   because they prevent the same low-value experiments from being repeated.
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
python -m ruff check src tests
python -m pytest
python -m forecast_select check-project
python -m forecast_select show-results
```

Do not treat generated output as verified until these commands pass on the
exact commit being delivered.
