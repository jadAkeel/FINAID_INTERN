# Contextual Defensive Selector

This experiment forces neutral indicator roles Up only when a past-only market-breadth signal indicates stress.

## Frozen design

- Selected stress threshold: `0.45`
- Selected role indicators: `X44, X49`
- Candidate selection used origins 120-179, with origins 180-219 as internal validation.
- Confirmation evaluation used origins 220-266 once after the rule was selected.
- Historical locked evidence 268-315 was not read.
- Indicator identities remain unknown; role labels describe behavior only.

## Results

- Discovery base accuracy: `61.7333%`
- Discovery contextual accuracy: `62.2667%`
- Confirmation base accuracy: `61.8440%`
- Confirmation contextual accuracy: `61.8440%`
- Confirmation accuracy delta: `+0.0000%`

Promotion eligible: `False`. A positive point estimate alone is not enough; the paired block lower bound must also be positive.
