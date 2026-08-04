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

## 5. Run the experimental downside gate

```powershell
python -m forecast_select build-risk-gate
python -m forecast_select show-risk-gate
```

This writes the experimental risk and gated prediction artifacts under
`research/downside_risk_gate/`. It evaluates Confirmation through origin 266
and does not read the historical locked evidence beginning at 268.

## 6. Run the directional downside experiment

```powershell
python -m forecast_select build-directional-downside
python -m forecast_select show-directional-downside
```

This writes walk-forward Down probabilities and bidirectional top-15
predictions under `research/directional_downside_selector/`. Model blending and
direction thresholds are selected on origins 120-179, then reported separately
on Validation 180-219 and Confirmation 220-266. Locked evidence is not read.

## 7. Run the contextual defensive experiment

```powershell
python -m forecast_select build-context-selector
python -m forecast_select show-context-selector
```

This writes a past-only market-breadth regime experiment under
`research/contextual_defensive_selector/`. Candidate roles are selected inside
Discovery, then evaluated on Confirmation through origin 266. It does not read
historical locked evidence.

## 8. Run the unified controller challenger

```powershell
python -m forecast_select build-unified-controller
python -m forecast_select show-unified-controller
```

This combines the frozen outputs of the three defensive experiments. Overlay
weights are selected on Tuning origins 120-179, with Validation and
Confirmation reported separately. It does not change the active model or read
locked origins 268-315.

## 9. Forecast the next three hidden months

```powershell
python -m forecast_select forecast-next-three
```

This writes `reports/next_three_month_forecast.json`. The first month is the
registered one-step scope. The second and third months use experimental direct
horizon-specific targets so the command never invents intermediate future
indicator values. The configured one-month availability lag remains in force.

## 10. Verify the project

```powershell
python -m pytest
python -m forecast_select check-project
```

The integrity check confirms the active artifact, registered result, and
preserved locked evaluation. It does not claim live production performance.
