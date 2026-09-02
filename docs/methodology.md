# Forecast Select methodology

## Active model decision

The Regime Adaptive Bidirectional Selector is the owner-promoted active model.
It was activated to support both Up and Down directions while the Uptrend
Selector remains the reproducible baseline. This product decision is recorded
separately from the research gate: Validation accuracy was 60.00%, the research
promotion gate did not pass, and the locked origins 268 through 315 remain
unread.

## Uptrend Selector baseline

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

## Experimental Regime Adaptive Selector

The adaptive path makes 15 calls normally and expands to 20 only when a
walk-forward market-breadth forecast is at least `0.65`. That forecast uses
structured panel aggregates through `t-1` and targets through `t-2`; the old
descriptive stress-to-cap mapping is retained only as a disabled fallback.
Expansion-only ranks 16-20 remain Up because their admission is conditioned on
a broad-Up forecast. The path excludes `X16`, replaces the frozen graph with a
causal rolling 48-month signed graph, and adds a 12-month asset-group
relative-strength overlay whose labels stop at `t-2`. Its guarded Down fallback
did not pass every research stability gate; its activation is the separate
owner-directed product decision documented above.

A separate fixed-coverage accuracy path searches Development origins 120-219
over causal group weights while enforcing exactly 15 calls per month. Caps
below the project's configured monthly minimum are rejected. The selected
policy uses group weight `0.25`; its point estimates remain below 65%, so this
fixed-coverage Up-only branch is not used by the active bidirectional model.
Confirmation and locked origins are not used for parameter selection.

The 2026-08-07 non-locked follow-up uses a provenance-keyed cache that separates
causal replay inputs from outcomes and rejects locked origin ranges. Threshold
screens use Tuning only; Validation is a gate; Confirmation is descriptive and
cannot make a policy promotion-eligible. Three bounded cap policies, three
single-family Uptrend ablations, and two Down challengers were evaluated. None
improved the 60.00% Validation reference under the temporal and block-bootstrap
gates, so optional model and correctness-meta-model searches were not run.

## Score and correctness semantics

The output fields intentionally separate three concepts:

- `p_up`: estimated Up-direction score; not proven calibrated;
- `selection_score`: ranking utility used to choose calls; not necessarily a
  probability or comparable across regimes;
- `directional_score`: strength of the chosen direction, retained under the
  compatibility alias `directional_confidence`; not a correctness probability.

No individual `correctness_probability` is released. Selected rows carry the
status `unavailable_no_valid_oof_individual_calibrator`, and both
`correctness_probability` and `correctness_lcb` are null. The previous copied
values are retained only in `legacy_correctness_probability` for auditability.

For monitoring, `cohort_correctness_probability` is the Laplace-smoothed
marginal historical hit rate `(hits + 1) / (calls + 2)` over prior selected
calls. Labels stop at `origin - 2`, at least 12 prior selection months are
required, and `cohort_correctness_lcb` is a one-sided 95% Wilson lower bound.
These cohort fields are not individualized and cannot justify threshold-based
abstention.
