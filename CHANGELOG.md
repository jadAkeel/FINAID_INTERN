# Changelog

## Unreleased

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
- excluded generated run folders, Python caches, and reproducible research
  caches from the delivery repository.

## 1.0.0

- introduced the leakage-safe forecasting package and command-line interface;
- registered the Uptrend Selector baseline and its non-locked evidence;
- preserved the locked evaluation boundary and added GitHub continuous
  integration.
