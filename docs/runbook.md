# Reproduction runbook

Run all commands from the repository root.

## 1. Install

```powershell
python -m pip install -e ".[dev]"
```

## 2. Audit the workbook

```powershell
python -m forecast_select audit-data
```

This validates the workbook and refreshes:

- `reports/data_profile.json`
- `reports/data_audit.md`

## 3. Build or validate the active model

```powershell
python -m forecast_select build-model
```

The command builds `artifacts/active/uptrend_predictions.parquet` when absent.
When the artifact already exists, it validates its structure and registered
926/1500 result.

## 4. Read the result

```powershell
python -m forecast_select show-results
```

## 5. Verify the project

```powershell
python -m pytest
python -m forecast_select check-project
```

The integrity check confirms the active artifact, registered result, and
preserved locked evaluation. It does not claim live production performance.
