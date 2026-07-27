# Methodology summary

The official label at origin `t` is `1` iff the next observed value is strictly greater than the current value. Ties are `0` and are stored as `zero_change`. The panel is split by month, never randomly. For origin `t`, all training rows have origin position `< t`; the forecast row uses only causal features.

The feature builder applies a conservative one-month as-of lag because release lags and vintages were not supplied. It does not backward-fill or interpolate leading missing histories. Late-starting indicators are eligible only after 24 observed months and when current/next target values exist.

Development origins are positions 120–267. The locked audit is positions 268–315, whose targets end at the final supplied row. Position 316 (May 2026) is the production origin for an unscored June 2026 ledger.

The v2 execution completed baselines, the global Logistic anchor, a pre-registered equal-weight Level-B artifact, and a per-origin Level-C calibration/reliability layer. Full CatBoost walk-forward was attempted with a 300-second bound but timed out; only its one-origin smoke test remains available. Pretrained models and broad hyperparameter search remain blocked/not run.
