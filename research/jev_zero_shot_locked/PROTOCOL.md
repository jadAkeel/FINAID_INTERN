# Jev zero-shot on the consumed locked window — pre-registration

Written 2026-09-25, before any Jev request was sent. Requested by the project owner, who also
authorised sending the anonymous workbook values to TypeSafe AI's API.

## Status of the evidence this can produce

The locked window, origins 268–315, was **consumed** on 2026-09-23 by the final locked
evaluation (`research/final_locked_evaluation/`). Its outcomes, and the active model's score of
492/807 = 60.97%, are known. This experiment is therefore **descriptive, developed with the
locked outcomes known**. It cannot promote a model or replace the locked result, and it is not
a blind test. To keep it as close to one as possible:

- Everything below — state format, questions, model id, metrics, thresholds — is fixed in this
  file before the first request.
- Jev is used zero-shot. There is no fine-tuning, prompt search, threshold search, or rerun
  with a changed prompt. The run happens once. Every raw response is cached so the result can
  be re-scored without sending anything again.
- The sealed `artifacts/audit/locked_evaluation.parquet` is not read.

## Model

TypeSafe AI **Jev**, requested as `jev-latest` at `https://api.typesafe.ai/v1/systemone`
(alternative route: Vercel AI Gateway, model `typesafe-ai/jev`). The versioned model name
returned by the API is recorded with every response. Only `noul` (probability) questions are
used.

## What Jev sees

Causal boundary as in the rest of the project: at origin `t` the newest observation is `t-1`,
and the target is the move from `t` to `t+1`. Month `t` itself is unobserved, so each question
is a two-step-ahead question.

**Indicator question.** One request per (origin, indicator) in the active model's locked
universe (`level_c_ready` and not adaptively excluded): 2,256 requests over 48 months.
State: the indicator's last 36 month-over-month changes through `t-1`, divided by their own
standard deviation, rounded to 2 decimals; the matching U/D direction string; and the share
of those months that rose. The indicator id is not sent. Question `rises`: *"Month M+1 has not
been observed yet. Will this indicator's value in month M+2 be strictly higher than its value
in month M+1?"* The answer is `p_up`.

**Market question.** One request per origin, 48 requests. State: for the last 36 months
through `t-1`, the share of the 50 indicators whose value rose. Question `broad_down`:
*"Month M+1 has not been observed yet. Between month M+1 and month M+2, will fewer than half of
these indicators rise?"* The answer is `p_broad_down`.

## Pre-registered measurements

Targets are recomputed from the workbook and must equal the locked artifact's `y_true` on every
row, otherwise the run stops. A broad-down month is one in which fewer than half of the locked
universe rose (17 of the 48 months).

| Id | Measurement | Comparator |
|---|---|---|
| **P1** | Jev Up-only selection: each month, the top `k` indicators by `p_up`, all called Up, where `k` is the active model's own call count that month | Active model, 492/807 |
| **P2** | Market timing: AUC of `p_broad_down` for the 17 broad-down months; and the hit change from flipping the active model's picks to Down in months with `p_broad_down >= 0.5` | AUC 0.5; active model's hits |
| **P3** | Per-call reliability: AUC of Jev's probability for the active model's chosen direction, against whether each of its 807 calls was correct | AUC 0.5 |
| S1 | Directional AUC of `p_up` over all 2,256 universe rows | AUC 0.5 |
| S2 | Direction-free selection: top `k` by `abs(p_up - 0.5)`, direction from `p_up >= 0.5` | Active model |
| S3 | Distribution of `p_up`: mean, quartiles, share within 0.45–0.55 | — |

Hit deltas are paired by month and reported with a six-month moving-block bootstrap over the
48 months (5,000 replicates, seed `20260925`), p5 / p50 / p95.

## Interpretation, fixed in advance

Jev is reported as **worth a forward test on genuinely new months** if at least one holds:

- P1 hit delta above 0 with bootstrap p5 above 0;
- P2 AUC at least 0.60 and a positive flip-rule hit change;
- P3 AUC at least 0.55.

Otherwise the result is **no evidence of added signal**, and it is recorded in
`docs/EXPERIMENT_REGISTRY.md` as rejected. Either way, no production model, configuration, or
report changes because of this run.

## Reproduction

```bash
# Dry run: builds every request and checks targets. Sends nothing.
python research/jev_zero_shot_locked/jev_locked.py

# Live run, once. Needs TYPESAFE_API_KEY in the environment.
python research/jev_zero_shot_locked/jev_locked.py --send

# Re-score from the cached responses without sending anything.
python research/jev_zero_shot_locked/jev_locked.py --score-only
```
