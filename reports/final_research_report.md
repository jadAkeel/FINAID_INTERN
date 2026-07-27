# Final Research Report

## Executive Result

The repository executes a leakage-safe revised-data pseudo-out-of-sample research pipeline with full-coverage and selective tracks. Claims below are limited to generated artifacts.

## Official PDF Milestone Status

The supplied workbook ends in May 2026. A complete six-month evaluation beginning January 2026 and the subsequent five-month persistence period are unavailable. The compensation-related milestone is therefore `NOT_YET_EVALUABLE`. Any selective accuracy above 65% must not be presented as proof of the PDF's overall-accuracy condition.

## Validation

Training rows for origin t are strictly earlier than t; the official target is 1 iff value(t+1) > value(t), with ties recorded as zero_change. The last 48 evaluable origins remain frozen as `locked_audit_v1` and were not read by Level-C.

## Level-C

Level-C v2 fits Platt calibration, a correctness model, a six-month date-block bootstrap bias correction, a 0.55 reliability floor, and a maximum of 20 accepted predictions per month using earlier Level-B outputs only. Full CatBoost walk-forward timed out at the bounded limit and is not promoted.

## Experiments

- Level-C ready rows: `5946`
- Level-C full directional accuracy: `0.554154`
- Level-C accepted rows: `1414`
- Level-C coverage: `0.237807`
- Level-C accepted accuracy: `0.564356`
- Level-C bootstrap LCB p10: `0.546492`

Generated metric artifacts:
- `reports\tables\all_metrics.csv`
- `reports\tables\dev_ensemble_metrics.csv`
- `reports\tables\dev_ensemble_v2_metrics.csv`
- `reports\tables\dev_metrics.csv`
- `reports\tables\level_c_dev_metrics.csv`
- `reports\tables\locked_audit_v1_metrics.csv`

## Limitations

- Anonymous indicators have no supplied units, release lags, revision histories, or vintages.
- Pretrained Chronos-2, TiRex-2, and TimesFM experiments are blocked unless their official package/API and compatible checkpoints are verified locally.
- June 2026 is an unscored forecast ledger; its outcome is not fabricated.

## Reproduction

Run `python -m forecast_select audit`, `python -m pytest`, then the backtest, freeze, locked-audit, prediction, and report commands in `README.md`.
