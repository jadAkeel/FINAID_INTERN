# Directional Downside Selector

This experiment learns actual Down directions and can include them in the monthly top 15.

## Frozen design

- Global regularized logistic model across all indicators.
- Local regularized logistic model per indicator when history is sufficient.
- Indicator-specific rise-then-stall pattern prior.
- Rolling learned lead-lag peer features; no indicator meanings are assumed.
- Candidate selection uses Tuning origins 120-179 only.
- Validation is 180-219 and Confirmation is 220-266.
- Historical locked evidence 268-315 was not read.

## Selected parameters

- Local weight: `0.5`
- Pattern weight: `0.25`
- Down threshold: `0.65`
- Down margin: `0.1`

## Results

- Tuning base / candidate: `64.3333%` / `64.4444%`
- Validation base / candidate: `57.8333%` / `58.6667%`
- Confirmation base / candidate: `61.8440%` / `61.8440%`
- Confirmation Down calls / hits: `12 / 6`
- Promotion eligible: `False`

This experiment never changes the active model automatically.
