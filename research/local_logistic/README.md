# Local / Indicator-Specific Logistic Regression

Non-promoting research experiment testing whether indicator-specific logistic
coefficients improve the Up Selector over the shared global logistic
regression.

Full report: [`docs/research/local_logistic_experiment.md`](../../docs/research/local_logistic_experiment.md)

## Candidates

| Id | Model | Frozen configuration |
| --- | --- | --- |
| A | Global Logistic (baseline, unchanged) | `C = 0.25`, `lbfgs`, 44 features + one-hot `indicator_id` |
| B | Pure Local Logistic | `core8` (8 features), `C = 0.01`, minimum 60 local rows |
| C | Global + Local fixed shrinkage | `core8`, `C = 0.25`, minimum 60 rows, `w = 0.40` |
| D | Global + Local sample-aware shrinkage | same as C, `w_max = 0.10`, `n_ref = 180` |
| E | Global + indicator interactions | 6 interaction features × indicator, `C = 0.01` |

All candidates share one causal panel, one training slice per origin, the same
eligible rows, and the unchanged graph → 48-month prior → top-15 selector.

## Result

Top-15 hit delta versus the Global baseline:

| Candidate | Tuning (120–179) | Validation (180–219) | Confirmation (220–266) |
| --- | --- | --- | --- |
| B. Pure Local | −2 | +7 | 0 |
| C. Fixed shrinkage | −1 | +8 | −1 |
| D. Sample-aware shrinkage | −5 | +3 | −1 |
| E. Interactions | −9 | +4 | +2 |

- **0 of 220 tuning configurations beat the baseline**; the best was −1 hit.
- No candidate is positive in two consecutive windows.
- Local fallback rate: 20.54% tuning, 11.20% validation, 8.71% confirmation.

**Decision: `GLOBAL LOGISTIC REMAINS PREFERRED`.** Local modelling is retained
as research evidence only. Nothing in production was changed.

## What was learned anyway

- Indicator-specific structure is real: per-indicator accuracy spread exceeds
  a matched coin-flip null (p ≈ 0.000), and local slopes reproduce at
  r = 0.63–0.87 between origin 179 and origin 266.
- Local models are better calibrated than the global model in all three
  windows (Brier and log loss), and fix its Up overconfidence.
- The gains land on indicators the selector never picks: +2.87 pp on
  never-selected indicators, −0.51 pp on selected ones.
- The top-15 selector barely uses `p_up`. Replacing the logistic signal with a
  constant (selecting on the trailing 48-month prior alone) scores
  570/900, 352/600, 438/705 — matching or beating the production model out of
  sample. Reported as a diagnostic, not a production recommendation.

## Reproduction

```bash
python -m forecast_select.local_logistic_runner tune
python -m forecast_select.local_logistic_runner evaluate
python -m forecast_select.local_logistic_report
python research/local_logistic/diagnostics.py
```

## Locked set

**Locked origins 268–315 were not read or used during this experiment.** The
workbook is loaded with `nrows` capped at position 267 and every origin list
passes `assert_origins_unlocked`.
