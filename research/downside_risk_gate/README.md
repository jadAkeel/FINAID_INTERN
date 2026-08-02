# Downside Risk Gate

This is an experimental risk filter. It does not replace the active Uptrend Selector.

## Frozen design

- Selected penalty: `0.0`
- Penalty selection window: Discovery origins 120-219.
- Evaluation window: Confirmation origins 220-266.
- Historical locked evidence 268-315 was not read.
- The gate reranks Up candidates; it does not flip predictions to Down.

## Results

- Discovery base accuracy: `61.7333%`
- Discovery gated accuracy: `61.7333%`
- Confirmation base accuracy: `61.8440%`
- Confirmation gated accuracy: `61.9858%`
- Confirmation accuracy delta: `+0.1418%`
- Confirmation changed selections: `1`
- Shock ROC AUC on Confirmation: `0.5640`

The experiment is not promoted automatically. A positive point estimate is insufficient when the paired date-block interval includes zero.
