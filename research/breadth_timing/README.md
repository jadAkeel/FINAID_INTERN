# Breadth timing study — pre-registration

Committed **before** the script was run. Nothing below is revised after seeing results.
Research positions are 1-based. The script reads the workbook with `maximum_position=267`.
The locked window (origins 268–315) and `artifacts/audit/locked_evaluation.parquet` are not read.

## Why this study

The oracle-headroom analysis (`research/oracle_headroom/`) shows that production picks indicators
almost as well as hindsight allows: 62.44% against a 62.71% long-run drift oracle. If the same
picks were called Down in every month when most of them fell, accuracy would be 77.13%. All of
the remaining headroom is **market timing**. Production's own breadth forecast has AUC 0.52.
This study asks one question: does any causal, market-level signal in this data predict the months
when production's picks mostly fall?

## Target

At origin `t` (120–266), production accepts 15–20 calls. The **flip target** is
`F_t = 1` if fewer than half of those accepted indicators rose from `t` to `t+1`. That is exactly
the condition under which calling all of them Down gains hits.
Secondary target: universe breadth below 0.5, over production's selection universe.

## Signals (fixed list, levels through `t-1` only)

The availability lag is one month, as in `features.py`. X16 is excluded. `r` is the monthly log
change, and `z` is `r` divided by the standard deviation of that indicator's previous 36 changes,
with at least 24 required.

| id | signal | definition at origin t |
|---|---|---|
| S1 | `ma10_breadth` | share of indicators with `X[t-1]` above its 10-month mean `X[t-10..t-1]` |
| S2 | `mom12_breadth` | share with `X[t-1] > X[t-13]` |
| S3 | `shock_magnitude` | cross-sectional mean of `abs(z[t-1])` |
| S4 | `shock_dispersion` | cross-sectional std of `z[t-1]` |
| S5 | `risk_on_spread` | mean `z` over the last 3 changes of equity groups (thematic, US sector, global) minus that of fixed income |
| S6 | `breadth_12` | mean direction label over `s = t-13 .. t-2` (all indicators) |
| S7 | `vol_regime` | cross-sectional mean of (3-month mean `abs(z)`) minus (36-month mean `abs(z)`) |
| C1 | `combined_logit` | walk-forward L2 logistic (C = 1.0) on standardized S1–S7 |

C1 is trained at each origin `t` on every earlier origin `s ≤ t-2`, starting at position 40. It
uses target breadth over all available indicators below 0.5. Production's picks do not exist
before origin 120, so the flip target cannot be used for training. Features are standardized with
training-row statistics only.

## Selection (Tuning 120–179 only)

1. **S\***: the single signal with the largest `|AUC − 0.5|` for the flip target on Tuning. Its
   sign is fixed on Tuning.
2. **C1** needs no selection.
3. **Flip rule** for each finalist: flip all of production's accepted calls at `t` to Down when the
   score is at or above its own causal expanding quantile `1 − q`. The quantile uses the score's
   values at positions 40..t-1. Here `q ∈ {0.10, 0.20, 0.30}`, chosen by the Tuning hit gain.
   Call counts are unchanged, so the 15–20 contract holds.

## Pre-registered gate

A finalist passes only if **all** of the following hold, out of sample on Validation (180–219)
and Confirmation (220–266):

- **G1 — skill:** flip-target AUC ≥ 0.60 on pooled V+C (87 months), and ≥ 0.55 in each of V and C.
  In addition, the circular-shift permutation p-value is below 0.05. The null uses 2,000 random
   shifts of at least 6 months of the score against the target.
- **G2 — value:** the flip rule gains hits over production in **both** V and C. The pooled V+C
  six-month moving-block bootstrap 5th percentile of the monthly hit gain is above 0. The
  bootstrap uses 5,000 replicates and seed `20260923`.

If neither finalist passes, the study is recorded as a rejection. That completes the evidence
that next-month market direction is not forecastable from this data. The final locked evaluation
then proceeds on production unchanged. If one passes, it becomes the single pre-registered
candidate for that locked evaluation, alongside production.

All signals' AUCs are reported for every window and both targets, including those not selected.

---

## Results (added after the single pre-registered run)

**Rejected. Neither finalist passes any gate condition.**

Reproduce: `python research/breadth_timing/breadth_timing.py`. Outputs are in `metrics/`:
`summary.json`, `signal_auc.csv`, `single_signal_flip_diagnostic.csv`, and `monthly_scores.csv`.
The flip target's base rate is 30% on Tuning, 35% on Validation, and 32% on Confirmation.

| Finalist | Tuning AUC | Val AUC | Conf AUC | V+C AUC | perm. p | Hit gain T / V / C | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| S\* = `mom12_breadth` (chosen on Tuning, q = 0.20) | 0.624 | 0.464 | 0.597 | 0.529 | 0.50 | +23 / 0 / **−26** | fail |
| C1 = `combined_logit` (q = 0.10) | 0.463 | 0.635 | 0.544 | 0.589 | 0.07 | 0 / 0 / **−25** | fail |

The flip rule fired in only 3 out-of-sample months for S\*, and 2 for C1. None of those months
was a flip-target month (precision 0/3 and 0/2), so each firing cost about 8–13 hits.

Flip-target AUC for every signal, with each signal's raw sign:

| Signal | Tuning | Validation | Confirmation | V+C |
|---|---:|---:|---:|---:|
| ma10_breadth | 0.538 | 0.468 | 0.547 | 0.505 |
| mom12_breadth | 0.624 | 0.464 | 0.597 | 0.529 |
| shock_magnitude | 0.504 | 0.723 | 0.604 | 0.663 |
| shock_dispersion | 0.534 | 0.734 | 0.608 | 0.661 |
| risk_on_spread | 0.493 | 0.497 | 0.469 | 0.489 |
| breadth_12 | 0.550 | 0.431 | 0.598 | 0.502 |
| vol_regime | 0.431 | 0.703 | 0.435 | 0.560 |
| combined_logit | 0.463 | 0.635 | 0.544 | 0.589 |

The Tuning winner, 12-month momentum breadth, does not hold up out of sample. Trend-following
and cross-asset (risk-on/risk-off) signals carry no information about next month's direction.

### Post-hoc observation: shock signals

**This section is exploratory and cannot support a claim.** The two shock signals (the size and
spread of last month's standardized moves) reach AUC 0.66 on Validation+Confirmation, but only
0.50–0.53 on Tuning, so the pre-registered selection did not pick them. Seven signals were
examined across three windows, so one of them looking good in two windows is not surprising. More
important, the signal has no *tail* value. `post_hoc_shock_tail.csv` is produced by
`post_hoc_tail_check.py`, with quantiles computed in hindsight within each window. It shows that
in the months with the **largest** shocks, production's picks are among its *best*: 74–86%
accuracy in the top 10%, because these are rebound months. Flipping them to Down loses hits in
every window and at every quantile tested. The only exception is +6 on Validation at the top
20% of `shock_dispersion`. The AUC comes from middle-ranked months, where no rule can act
profitably. The correlation between shock magnitude and production's monthly accuracy is
+0.05, −0.18, and +0.06 by window.

### Conclusion

Three independent attempts have found no forecastable market-direction timing in this data:
production's own breadth forecast (AUC 0.52), the calendar-seasonal prior, and this study. The
study tested trend, momentum, shock, volatility-regime, cross-asset, and combined signals. The
15-point timing headroom seen by the hindsight oracle cannot be reached with causal information
from these 50 series. Production is kept unchanged as the single candidate for the final locked
evaluation.
