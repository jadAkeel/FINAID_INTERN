# Forecast Select

Forecast Select predicts the next-month direction of 50 anonymous indicators
and selects 15–20 calls per forecast month. The code keeps every model decision
inside a causal boundary: features use observations through `t-1`, training
labels stop at `t-2`, and locked origins 268–315 are not used for tuning.

## Current model

The owner-promoted production model is the **Regime Adaptive Bidirectional
Selector** (`forward_breadth_dynamic_cap_v3`). The **Uptrend Selector** remains
the reproducible baseline.

| Evaluation window | Accuracy | Calls | Selection AUC | Directional AUC |
|---|---:|---:|---:|---:|
| Tuning (120–179) | 64.05% | 1,071 | 0.5413 | 0.5795 |
| Validation (180–219) | 58.52% | 675 | 0.4015 | 0.5094 |
| Confirmation (220–266) | 63.58% | 799 | 0.4411 | 0.5380 |

The active model was chosen to support both Up and Down calls. It did **not**
pass the formal research promotion gate, so the repository does not claim 65%
validated accuracy. Rejected and negative experiments are retained in
[`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md).

## Install and run

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[dev]"

# Validate or rebuild the active artifact.
python -m forecast_select build-model

# Write the next three direct-horizon forecasts.
python -m forecast_select forecast-next-three

# Validate delivery artifacts and provenance.
python -m forecast_select check-project

# Display the registered model result.
python -m forecast_select show-results
```

The generated forecast is written to
`reports/regime_adaptive_next_three_forecast.json`. Each selected row includes
the indicator, direction, directional scores, group, and rank.

## Project layout

```text
configs/   Production and research settings
data/      Source workbook
src/       Installable forecast_select package
tests/     Unit, leakage, regression, and artifact-contract tests
artifacts/ Registered active and audit artifacts
research/  Reproducible experiments and retained evidence
reports/   Human- and machine-readable production results
docs/      Methodology, experiment registry, and delivery notes
archive/   Quarantined evidence that must not feed production
```

The production entry point is `forecast_select.cli`. Experimental commands are
implemented in `forecast_select.research_cli` and loaded only when requested.
This keeps normal use readable without erasing research history.

## Verification

```powershell
python -m ruff check src tests
python -m pytest
```

GitHub Actions runs the same lint and test checks. See
[`docs/FINAL_DELIVERY.md`](docs/FINAL_DELIVERY.md) for the delivery boundary and
[`CHANGELOG.md`](CHANGELOG.md) for the consolidated change record.
