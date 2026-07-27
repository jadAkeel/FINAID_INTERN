# Model card: CatBoost global challenger v2

## Status

Completed full development challenger; rejected for promotion based on standalone out-of-sample evidence.

## Method

One shallow global CatBoost classifier with depth 4, 120 iterations, learning rate 0.04, L2 regularization 8, two training threads, fixed seed `20260727`, compact causal features, categorical `indicator_id`, and monthly grouped walk-forward origins 120–267. The model was executed in 19 resumable chunks of 8 origins.

## Evidence

- Artifact: `artifacts/oof_predictions/catboost_full_v2.parquet`
- Metrics: `reports/tables/catboost_full_v2_metrics.csv`
- Provenance: `reports/experiments/catboost_full_v2_provenance.json`

The artifact has 6,397 eligible rows over 148 development origins, with no error rows or missing probabilities.

## Development metrics

- Accuracy: 52.337%
- Balanced accuracy: 48.342%
- Brier score: 0.258485
- Log loss: 0.712088
- ROC-AUC: 0.465941

These results do not justify promotion, production readiness, a causal claim, or the PDF 65% milestone.

