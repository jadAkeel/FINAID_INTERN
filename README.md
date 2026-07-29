# Forecast Select

Forecast Select is a leakage-safe research pipeline for predicting the next
monthly direction of 50 anonymous indicators.

The project has one active model with a descriptive name:

## Uptrend Selector

The model:

1. builds features using observations available through the previous month;
2. trains one regularized Logistic model across all indicators;
3. adds rolling PCA, peer-correlation, breadth, dispersion, and momentum features;
4. propagates probabilities through a frozen signed correlation graph;
5. selects the 15 indicators with the strongest estimated Up probability.

Registered Selection result:

- 100 months
- 15 indicators per month
- 926 correct calls out of 1,500
- 61.73% accuracy
- 1,500 Up calls and 0 Down calls

The final point is important: this is an Uptrend selector, not a balanced
Up/Down forecasting system. The repository does not claim 65% accuracy,
real-time vintage validity, or production readiness.

## Quick start

```powershell
python -m pip install -e ".[dev]"
python -m forecast_select audit-data
python -m forecast_select build-model
python -m forecast_select show-results
python -m forecast_select check-project
python -m pytest
```

## Project structure

```text
configs/
  config.yaml                 Shared data and validation settings
  uptrend_model.yaml          Active model settings and registered result

data/
  monthly_indicators.xlsx     Immutable monthly input workbook

src/forecast_select/
  features.py                 Causal feature construction
  uptrend_model.py            Structured Logistic model
  indicator_selection.py      Correlation propagation and top-indicator selection
  uptrend_pipeline.py         End-to-end active model pipeline
  project.py                  Data audit and project-integrity checks
  validation.py               Walk-forward timing rules

artifacts/
  active/                     Registered Uptrend Selector predictions
  audit/                      Preserved locked evaluation

research/reference_models/
  artifacts/                  One evidence artifact per comparison model
  metrics/                    Matching metrics

reports/                      Current data, model, and integrity reports
tests/                        Unit, integration, regression, and leakage tests
```

## Validation boundary

At forecast origin `t`:

- feature values use observations through `t-1`;
- model labels and indicator history stop at `t-2`;
- Selection covers origins 120 through 219;
- the preserved locked evaluation is not used to choose or tune the model.

## Documentation

- [Arabic project overview](docs/overview-ar.md)
- [Data contract](docs/data.md)
- [Model methodology](docs/methodology.md)
- [Reproduction runbook](docs/runbook.md)
- [Verification rules](docs/verification.md)
- [Reference model portfolio](research/reference_models/README.md)
- [65% accuracy feasibility study](research/accuracy_feasibility/README.md)
