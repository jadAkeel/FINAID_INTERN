# Level-C calibration and reliability

All calibration, correctness, date-block bootstrap, and selection decisions are fit at each origin from strictly earlier Level-B rows. The locked `locked_audit_v1` artifact is not read or used.

- Reliability floor: 0.55
- Monthly cap: 20
- Bootstrap block: 6 months

## Development summary

- ready_rows: `5946`
- ready_months: `136`
- full_accuracy: `0.5541540531449715`
- calibrated_brier: `0.2485071530438236`
- correctness_brier: `0.24857841985520457`
- accepted: `1414`
- coverage: `0.23780692902791792`
- accepted_accuracy: `0.5643564356435643`
- lcb_p10: `0.5464918598625425`
- lcb_p90: `0.618530223075804`
- artifact: `artifacts\oof_predictions\dev_level_c_v2.parquet`
