# Reference models

This folder keeps one read-only evidence artifact for each distinct model family.
These models are not active commands and are retained only for comparison.

| Model | Evaluation | Accuracy | Purpose |
|---|---|---:|---|
| Weighted Ensemble | 15 indicators × 100 months | 61.40% | Combines several historical signals |
| Lead-Lag Logistic | 15 indicators × 100 months | 60.93% | Uses indicators that tend to move earlier than their peers |
| Market Regime Selector | 15 indicators × 100 months | 60.13% | Adds a market-wide Up/Down state |
| Parliament Vote | 15 indicators × 100 months | 56.73% | Uses reliability-weighted majority voting |
| Baseline Models | 4,145 full-coverage rows | 55.66% best baseline | Provides simple comparison rules |
| Structured CatBoost | 4,145 full-coverage rows | 54.28% | Tests nonlinear tree relationships |

The active **Uptrend Selector** is intentionally not duplicated here. It remains
the strongest comparable top-15 result at 926/1500 correct calls (61.73%).

Artifacts live in `artifacts/`; their matching metrics live in `metrics/`.
Probability, target, direction, and selection columns were preserved when the
files were renamed for the first release.
