# Forecast Select

Predict the next-month direction of 50 indicators. The active model selects **15–20 indicators per month**.

**Active Model:** Regime Adaptive Bidirectional Selector (`forward_breadth_graduated_15_to_20`)

## Results

| Period | Accuracy | SelAUC | DirAUC |
|---|---:|---:|---:|
| Tuning (120–179) | 64.05% (686/1071) | 0.5413 | 0.5795 |
| Validation (180–219) | 58.52% (395/675) | 0.4015 | 0.5094 |
| Confirmation (220–266) | 63.58% (508/799) | 0.4411 | 0.5380 |

We tested 164 challengers. Best improved SelAUC to 0.4689 (+0.067) but stayed below 0.50, so **no promotion** — active model unchanged.

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

## Limitations

- Strong Up bias (675 Up / 0 Down in Validation)
- High monthly noise (SelAUC std 0.207)
- Linear group overlay ceiling ~0.47 — needs new causal feature for >0.50
- March 2026 reversal (10% Up) was missed
