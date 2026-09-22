# Changelog

## Unreleased

- added the Local / Indicator-Specific Logistic Regression research branch
  (`local_logistic_selector`): per-indicator local logistic models,
  Global/Local fixed and sample-size-aware shrinkage, and an
  `indicator_id x feature` interaction model, each scored through the
  unchanged graph, 48-month prior and top-15 selector; rejected, with the
  Global Logistic baseline retained and no production change;
- promoted the Regime Adaptive Bidirectional Selector as the explicit active
  product model while preserving the Uptrend Selector as the reproducible
  baseline;
- separated the small production CLI from lazily loaded research commands;
- added causal feature families, calibration and signal-ceiling audits,
  bounded regime experiments, and their regression tests;
- added a comprehensive experiment registry so negative, rejected, and
  quarantined results remain discoverable without crowding the main README;
- consolidated delivery instructions around `python -m forecast_select` and
  restored package, test, lint, and Git ignore configuration;
- updated GitHub Actions to its Node 24-compatible releases and made the
  correlation graph compatible with read-only arrays returned by pandas 3;
- excluded generated run folders, Python caches, and reproducible research
  caches from the delivery repository.

## 1.0.0

- introduced the leakage-safe forecasting package and command-line interface;
- registered the Uptrend Selector baseline and its non-locked evidence;
- preserved the locked evaluation boundary and added GitHub continuous
  integration.
