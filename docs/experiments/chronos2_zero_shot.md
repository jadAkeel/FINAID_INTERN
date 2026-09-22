# Experiment: Chronos-2 Zero-Shot Directional Forecasting

## 1. Experiment Overview

- **Experiment ID**: `chronos_zero_shot`
- **Model ID**: `amazon/chronos-2`
- **Revision**: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- **Package Pin**: `chronos-forecasting==2.3.2`
- **Status**: Research Only / Non-Promoting
- **Promotion Barrier**: `promotion_eligible = False`; requires locked evaluation protocol.

This experiment evaluates the zero-shot directional forecasting capability of Chronos-2 on macroeconomic indicators. Chronos-2 is a pre-trained probabilistic time-series foundation model based on transformer architecture.

## 2. Formulation & Horizon Alignment

### Two-Step Differenced Formulation
- Standard macro indicators exhibit unit-root non-stationarity and availability lags.
- Given an observation lag of 1 month, at origin $t$, data is causally available only through $t - 1$.
- Causal context: raw historical observations $[X_1, X_2, \dots, X_{t-1}]$ are differenced purely within this cutoff:
  $$\Delta X_\tau = X_\tau - X_{\tau-1}, \quad \tau \le t - 1$$
- Chronos-2 predicts a horizon of 2 steps forward on the differenced sequence:
  - Step 1: $\Delta X_t$ (unobserved historical month)
  - Step 2: $\Delta X_{t+1}$ (target forecast month)
- Target label: $y = \mathbb{I}(X_{t+1} - X_t > 0) = \mathbb{I}(\Delta X_{t+1} > 0)$.
- Step-2 quantiles $[q_{0.10}, \dots, q_{0.90}]$ are analyzed via non-parametric piecewise-linear interpolation to evaluate $F(0) = P(\Delta X_{t+1} \le 0)$.
- Directional probability:
  $$P(\text{Up}) = 1 - F(0)$$
  clipped to $[0.001, 0.999]$.

## 3. Evaluation Windows & Hard Contract

| Window | Origin Range | Months | Role |
|---|---|---|---|
| **Tuning** | `[120, 179]` | 60 | Calibration fitting and blend weight selection |
| **Validation** | `[180, 219]` | 40 | Non-locked research gate evaluation |
| **Confirmation**| `[220, 266]` | 47 | Descriptive out-of-sample confirmation |
| **Excluded** | `267` | 1 | Buffer origin excluded from evaluation |
| **Locked** | `[268, 315]` | 48 | **Strictly locked and untouched** |

Any attempt to read or evaluate origins $\ge 267$ or set `locked_evaluation_read=True` raises a hard error.

## 4. Calibration & Blend Optimization

### Causal Platt Calibration
1. **Tuning Window (120-179)**:
   - Evaluated prequentially. For each origin $t$, Platt logistic regression is fitted exclusively on historical tuning origins $\le t - 2$.
   - Requires $\ge 12$ distinct origins and both binary classes present in the history; otherwise, falls back to raw probability.
   - Fit through origin is strictly $\le t - 2$.
2. **Validation and Confirmation Windows (180-266)**:
   - A single frozen Platt model is fitted on valid Tuning rows through origin `178`, the
     latest label available at Validation origin `180` under the `t-2` rule.
   - Applied out-of-sample to Validation and Confirmation.
   - Validation and Confirmation labels are never accessed during calibration.

### Active Weight Selection
- Candidate blends active probability $P_{\text{active}}$ and calibrated Chronos probability $P_{\text{chronos}}$:
  $$P_{\text{blend}}(w) = w \cdot P_{\text{active}} + (1 - w) \cdot P_{\text{chronos}}$$
- $w \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ is chosen to minimize the Brier score strictly on Tuning out-of-fold rows through origin `178`.
- Tie-breaking rule: ties strictly prefer $w = 1.0$ (favoring the active model).
- Confirmation labels are strictly descriptive and never affect $w$.

## 5. Matched Monthly Coverage & Gate Evaluation

- To eliminate coverage bias, candidate calls match the exact active accepted call count $k_t$ for every origin $t$.
- Candidate ranks common eligible indicators by confidence $|P_{\text{blend}} - 0.5|$ descending, with stable `indicator_id` ascending tie-breaker.
- An origin-blocked paired bootstrap (contiguous blocks of 6 months, 500 replicates, seed `20260727`) evaluates per-origin hit deltas $\Delta(t) = \text{hits}_{\text{cand}}(t) - \text{hits}_{\text{act}}(t)$.
- Non-locked gate passing requires:
  1. $\sum_{t \in \text{val}} \Delta(t) > 0$
  2. Validation bootstrap $p_{10} \ge 0$
  3. Validation blend Brier score $<$ Active Brier score
  4. $\sum_{t \in \text{conf}} \Delta(t) \ge 0$
  5. Zero locked reads

## 6. Pre-Training Contamination Caveat

Chronos-2 is pre-trained on expansive public time-series corpora, including publicly available macroeconomic datasets. Backtest performance across historical windows may reflect memorization or pre-training contamination rather than genuine generalizable signal. Consequently:
- Strong historical performance is insufficient evidence for deployment.
- True out-of-sample validation requires live forward tracking and post-release paper trading.
- Automatic production promotion is disabled (`promotion_eligible = False`).
