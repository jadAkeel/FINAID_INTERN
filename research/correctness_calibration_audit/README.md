# Correctness calibration audit

This audit separates direction scores, ranking utility, and final-decision correctness.

- Active legacy correctness AUC: `0.5225`.
- Adaptive legacy correctness AUC: `0.5002`.
- Best causal candidate AUC: `0.4970`.
- No evaluated causal calibrator produced stable useful individual-call discrimination.
- `correctness_probability` and `correctness_lcb` are therefore unavailable rather than copied from directional score.
- A causal marginal cohort rate and Wilson lower bound are retained for monitoring only.
- The locked evaluation artifact was not read.

## Field contract

- `p_up`: estimated Up-direction score; not proven calibrated.
- `selection_score`: ranking utility; not a correctness probability.
- `directional_score`: uncalibrated strength of the chosen direction.
- `correctness_probability`: null because no individual calibrator passed the release gate.
- `cohort_correctness_probability`: causal Laplace-smoothed historical selected-call rate, for monitoring only.
- `cohort_correctness_lcb`: one-sided 95% Wilson lower bound for the same cohort rate.

## Reproduction

```powershell
python -m forecast_select build-correctness-audit
python -m forecast_select show-correctness-audit
```

The `metrics` directory contains candidate comparison, reliability, slice, accuracy-coverage, and temporal block-bootstrap tables.
