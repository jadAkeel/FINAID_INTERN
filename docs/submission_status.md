# Project Submission Status and Data Request

## Executive Summary

The project is now a reproducible and auditable research pipeline with one
registered active model, several defensive experiments, and a unified
non-promoting controller. No unproven experiment has been promoted to the
active model, and the locked evaluation set has not been used for tuning or
promotion.

The project does not currently support a stable 65% accuracy claim. The
registered active result is:

- `926 / 1500` correct selections.
- `61.73%` accuracy on Selection/Discovery.
- 15 selected indicators per month.
- Active model: `Uptrend Selector`; all registered selections are Up.

## Completed Work

### Data contract

- Validated `data/monthly_indicators.xlsx`.
- 316 monthly rows and 50 anonymous indicators, from February 2000 through May 2026.
- Constructed next-month direction targets, treating ties as Down.
- Preserved missing history without interpolation or backfilling.

### Methodological safety

- Built causal features using only information available before each forecast.
- Applied walk-forward training.
- Prevented unavailable targets from entering training or selection.
- Added dedicated leakage and future-invariance tests.
- Preserved the locked evaluation byte-for-byte and kept it outside tuning.

### Active model

- Structured Logistic Regression.
- Momentum, volatility, breadth, PCA, and peer-correlation features.
- Frozen signed correlation graph.
- Top-15 monthly selection.
- Parquet prediction artifacts with configuration/data hashes and provenance.

### Defensive experiments

| Experiment | Current result | Decision |
|---|---:|---|
| Uptrend Selector | 61.73%, `926/1500` | Registered active model |
| Downside Risk Gate | 61.99% on Confirmation; one changed call | Not promoted |
| Directional Downside Selector | 61.84% on Confirmation; Down `6/12` | Not promoted |
| Contextual Defensive Selector | 62.27% Discovery; 61.84% Confirmation | Not promoted |
| Unified Forecast Controller | 61.84% Confirmation; no improvement | Not promoted |

The Unified Forecast Controller uses Directional Downside as its base, then
tests risk-percentile and contextual stress/role overlays without changing the
active model. Tuning selected zero overlay weights, meaning the current
combination did not demonstrate additional predictive value.

## Open Questions and Limitations

- Is the required target Up/Down for all 50 indicators, or selection of a fixed number of indicators?
- Is the 65% metric calculated across all decisions or only accepted selections?
- A complete recent evaluation period beginning in January 2026 has not yet been reported for the current active model.
- The current data does not include indicator names, units, release timestamps, or revision history.
- Monthly observations do not show intra-month movement or daily signals preceding shocks.

## Data Needed for the Next Phase

Please request the following from the data owner:

1. Names, categories, units, currencies, and geographic regions for `X1..X50`.
2. The timestamp when each observation became available to a forecaster, not only the period it describes.
3. Historical vintages or revision history for point-in-time evaluation.
4. Daily or weekly observations, preferably with volume, breadth, and advance/decline measures.
5. Additional leading indicators such as VIX, volatility term structure, credit spreads, liquidity, and financial-stress measures.
6. The official evaluation window, monthly decision count, and exact definition of the 65% target.
7. Actual outcomes through the full evaluation period so Confirmation and persistence can be measured correctly.

## Ready-to-Send Data Request

We need an updated indicator dataset with complete metadata for every series,
including name, category, unit, region, release timestamp, and revision history.
Daily or weekly observations and leading systemic-risk indicators would also be
valuable. Please also confirm the official accuracy definition: whether the
model must predict Up/Down for every indicator or select a fixed number of
indicators. Finally, we need actual outcomes for the complete evaluation window
so that the 65% target and the required persistence period can be evaluated
without using future information.

## Delivery Commands

```powershell
python -m pip install -e ".[dev]"
python -m forecast_select audit-data
python -m forecast_select build-model
python -m forecast_select build-risk-gate
python -m forecast_select build-directional-downside
python -m forecast_select build-context-selector
python -m forecast_select build-unified-controller
python -m forecast_select check-project
python -m pytest
```
