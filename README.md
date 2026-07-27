# Forecast Select

Reproducible, leakage-safe monthly directional forecasting for anonymous indicator series. The repository treats the supplied workbook as immutable source material, keeps full-coverage research evaluation separate from selective production selection, and records a locked final audit.

## Quick start

```powershell
python -m pip install -e .
python -m forecast_select audit
python -m forecast_select features
python -m forecast_select pretrained-backtest
python -m forecast_select train-final
python -m forecast_select predict-month
python -m forecast_select monitor
python -m pytest
python -m forecast_select backtest --models classical
python -m forecast_select catboost-full --chunk-size 8 --chunk-index 0
# repeat chunks, then assemble:
python -m forecast_select catboost-full --chunk-size 8 --assemble
python -m forecast_select ensemble
python -m forecast_select level-c
python -m forecast_select freeze
python -m forecast_select locked-audit
python -m forecast_select train-final
python -m forecast_select predict-month
python -m forecast_select report
```

The default data path is `data/raw/FinalList_Extended.xlsx`. The original files in `Downloads` are never modified. See `docs/runbook.md` for the stage gates and `reports/` for generated evidence.

This is a revised-data pseudo-out-of-sample study: indicator release lags and historical vintages were not supplied. The June 2026 ledger is deliberately unscored because its outcome is absent.
