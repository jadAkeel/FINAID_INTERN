# Forecast Select

Predict the next-month direction of 50 indicators. The active model selects **15–20 indicators per month**.

**Active Model:** Regime Adaptive Bidirectional Selector (`forward_breadth_graduated_15_to_20`)

## Results

| Period | Accuracy | SelAUC | DirAUC |
|---|---:|---:|---:|
| Tuning (120–179) | 64.05% (686/1071) | 0.5413 | 0.5795 |
| Validation (180–219) | 58.52% (395/675) | 0.4015 | 0.5094 |
| Confirmation (220–266) | 63.58% (508/799) | 0.4411 | 0.5380 |

## How to Run

```powershell
pip install -e ".[dev]"

# Build model (uses data through May 2026)
python -m forecast_select.cli build-regime-adaptive

# Forecast next 3 months: June, July, August
python -m forecast_select.cli forecast-regime-next-three

# See result
type reports\regime_adaptive_next_three_forecast.json
```

**To change number of picks (15–20):**
```powershell
python -m forecast_select.cli build-regime-adaptive --cap 15
python -m forecast_select.cli forecast-regime-next-three
# or --cap 20
```

## Latest Forecast (June–August 2026)

Generated from `2026-05-29` (Origin 316):

```
2026-06 (H1) | mixed | cap 15 | 15 picks: X41, X39, X40, X24, X9, X11...
2026-07 (H2) | mixed | cap 15 | 15 picks: X41, X39, X24, X40...
2026-08 (H3) | mixed | cap 15 | 15 picks: X41, X39, X24, X40...
```
Full details: `reports/regime_adaptive_next_three_forecast.json`

Each pick has: `indicator_id`, `direction` (Up/Down), `p_up`, `p_down`, `selection_score`, `group`, `rank`.

## Project Structure

```
configs/  -> model settings (cap 15-20, group weight 0.25)
data/     -> monthly_indicators.xlsx (316 positions)
src/      -> pipeline code
research/ -> experiments (baseline + two holdout studies)
reports/  -> forecasts
```

## Verification

```powershell
python -m pytest tests/unit/test_regime_adaptive.py -q  # 19 passed
```

Causal: features `<= t-1`, labels `<= t-2`. Locked 268–315 never used for tuning.

## Rejected Selection Experiments

No tested selection-group challenger produced a stable improvement over the active model. Family F (`recent-miss + group-stability`) initially appeared stronger, but its first implementation leaked future labels. After correcting the causal alignment and rerunning it, Validation gained only two hits while Tuning, Development, and Confirmation declined. Family F was removed from production.

See [`docs/SELECTION_GROUP_FAILED_REGISTRY.md`](docs/SELECTION_GROUP_FAILED_REGISTRY.md) for the full evidence and the list of experiments that should not be repeated as-is.
