# Task graph and live dashboard

The Codex orchestrator owns shared interfaces, phase gates, freeze, audit, and integration. OpenCode read-only planning was attempted in parallel for validation and architecture; the jobs timed out without changing files, so their status is recorded as a negative execution result.

| Task ID | Owner | Dependency | Status | Artifact |
|---|---|---|---|---|
| W0-SOURCE | orchestrator | none | completed | `reports/source_reconciliation.md`, `reports/source_inventory.json` |
| W0-SKELETON | orchestrator | none | completed | package/config/test structure |
| W0-RESOURCE | orchestrator | none | completed | runtime facts in `DECISIONS.md` |
| W0-PRETRAINED | orchestrator | runtime | blocked/negative | `EXPERIMENTS.csv` |
| W1-AUDIT | orchestrator | W0-SOURCE | completed | `reports/data_audit.md`, `reports/data_profile.json` |
| W1-PIT | orchestrator | W0-SOURCE | completed | `configs/data/availability.yaml` |
| W1-TARGET | orchestrator | W0-SKELETON | completed | `src/forecast_select/targets.py`, alignment tests |
| W1-VALIDATION | orchestrator | W0-SKELETON | completed | `src/forecast_select/validation.py`, leakage tests |
| W1-FEATURES | orchestrator | W1-TARGET | completed | `src/forecast_select/features.py` |
| W2-BASELINES | orchestrator | W1 gate | completed | `artifacts/oof_predictions/dev_oof.parquet` |
| W3-CLASSICAL | orchestrator | W2 gate | completed | global logistic and CatBoost-capable branch |
| W7-ENSEMBLE | orchestrator | W3 gate | completed | `artifacts/oof_predictions/dev_ensemble.parquet` |
| W8-FREEZE | orchestrator | review/tests | completed | `artifacts/model_registry/freeze_manifest_v1.json` |
| W8-AUDIT | orchestrator | W8-FREEZE | completed | `artifacts/oof_predictions/locked_audit_v1.parquet` |
| W8-PRODUCTION | orchestrator | W8-AUDIT | completed | `artifacts/forecast_ledgers/june_2026_unscored.csv` |

Parallelism is used only for read-only planning and independent checks. Shared schemas and the locked audit remain serial.
