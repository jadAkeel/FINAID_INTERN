# Uptrend Selector methodology

## Target

For indicator `Xi` at origin `t`, the target is `Up` when
`value(t+1) > value(t)`. A smaller or equal value is recorded as `Down`.
Unavailable future values remain unscored.

## Causal information boundary

The project uses a one-month observation lag:

- features at `t` use observations through `t-1`;
- a target at origin `s` needs the value at `s+1`;
- training labels and rolling indicator history therefore stop at `t-2`.

## Structured Logistic model

One regularized Logistic model is trained across all eligible indicators. Its
inputs include:

- indicator identity;
- lagged changes and directions;
- Momentum over 3, 6, 9, and 12 months;
- rolling level, volatility, dispersion, and robust distance features;
- cross-sectional rank and breadth;
- rolling PCA factors and loadings;
- peer-correlation consensus;
- rolling market-regime summaries.

The model is global: it learns from all indicators while retaining each
indicator's identity.

## Signed correlation propagation

A signed graph is estimated from historical indicator changes through origin
119 and then frozen. Positive and negative correlations propagate the Logistic
probabilities with a fixed blend weight of `0.35`.

## Monthly selection

The graph probability is blended equally with each indicator's trailing
48-label Up rate. Only labels available through `t-2` are used. The 15 highest
scores are selected each month.

## Registered evidence

The model reproduced 926 correct calls out of 1,500 (`61.73%`) on Selection
origins 120 through 219. All selected calls are Up. The preserved locked
evaluation is not used to tune or promote the model.

## Experimental Downside Risk Gate

The risk gate defines a sudden drop as a negative monthly return that is both
below the indicator's causal trailing 5th percentile and more than two robust
MAD-scaled deviations below its trailing median. `X16` is excluded because its
history contains inconsistent scales.

A class-balanced Logistic model estimates this event from past-only
volatility, drawdown, momentum, breadth, dispersion, rolling correlation,
asset-group returns, and the newest shock outcome whose target is available.
The estimated risk can penalize the Uptrend Selector's ranking, but it never
flips a call to Down.

The penalty grid is evaluated on Discovery 120-219. The selected penalty is
then frozen for Confirmation 220-266. The initial experiment selected zero
penalty and is retained only as negative evidence; it is not part of the
active model.

## Experimental Directional Downside Selector

Unlike the risk gate, this model directly targets `Down = 1 - y_true`. It uses
global and per-indicator regularized Logistic models, a causal empirical prior
for rise-then-stall exhaustion, and rolling lead-lag correlations learned from
anonymous peers. No economic meaning or hard-coded indicator group is assumed.

At each origin, the newest observation is `t-1` and the newest training label
is `t-2`. Global, local, and pattern probabilities are blended with parameters
selected only on Tuning 120-179. A Down call must beat both a probability
threshold and the competing Up score before all indicators are ranked and the
strongest 15 are selected.

The initial experiment improved Validation by five hits but produced no net
improvement on Confirmation. Its Confirmation Down precision was `6/12 = 50%`,
so it is retained as unpromoted evidence rather than replacing the active
model.

## Experimental Contextual Defensive Selector

This experiment uses market breadth calculated from observations through
`t-1`, averaged over three months. Within Discovery, a small registered grid
tests whether neutral indicator roles should be forced Up and replace the
lowest-scored selected row during low-breadth regimes.

The selected rule used a breadth threshold of `0.45` and roles `X44` and
`X49`. Discovery improved by eight hits, but Confirmation produced no net
change. The experiment is not promoted. The role labels describe observed
behavior only because the source workbook does not provide indicator names.
