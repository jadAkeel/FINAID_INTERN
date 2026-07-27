# Model card: global Elastic-Net-style Logistic anchor

## Status

Promoted as the classical anchor for the development pipeline; not a claim of production readiness or PDF milestone success.

## Method

One global long-panel `LogisticRegression` with an indicator one-hot effect, training-only median imputation, training-only scaling, compact causal features, `C=0.25`, and deterministic seed `20260727`. Training rows are restricted to origins earlier than the forecast origin.

## Intended use

Research comparison and a conservative unscored monthly ledger. The supplied data are revised and anonymous; this model must not be treated as a real-time or causal system.

## Evidence

Development OOF artifact: `artifacts/oof_predictions/dev_classical_oof.parquet`.

Development metrics: `reports/tables/dev_metrics.csv`.

## Limitations

The model does not have historical release vintages, indicator semantics, or a fully cross-fitted correctness model in this bounded run. Full CatBoost walk-forward, pretrained models, and Level-C calibration/reliability selection remain incomplete.

