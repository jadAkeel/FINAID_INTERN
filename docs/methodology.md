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
