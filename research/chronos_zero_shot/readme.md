# Chronos-2 Zero-Shot Directional Forecasting Research

This directory houses the experimental evaluation and research-pipeline artifacts for evaluating Amazon's Chronos-2 (`amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`) in a zero-shot setting against the active production model.

## Core Methodology

### 1. Two-Step Differenced Horizon
- Inputs are differenced series constructed purely inside truncated causal context strictly through $t - 1$ (availability lag = 1 month).
- Horizon = 2 steps: step 1 corresponds to month $t$, step 2 corresponds to target month $t + 1$.
- Directional probability $P(\text{Up}) = P(\Delta X_{t+1} > 0) = 1 - F(0)$ is derived via piecewise linear interpolation over repaired quantiles (quantiles 0.1 through 0.9) with zero Gaussian assumptions.

### 2. Evaluation Windows
Strict temporal windows are enforced across all evaluations:
- **Tuning**: Origins `120` to `179` (60 months)
- **Validation**: Origins `180` to `219` (40 months)
- **Confirmation**: Origins `220` to `266` (47 months)
- **Locked Evaluation**: Origins `268` to `315` are **strictly locked and untouched**.
- **Excluded**: Origin `267` is excluded by protocol. Any origin $\ge 267$ is rejected with a hard error.

### 3. Tuning-Only Selection and Calibration
- **Causal Platt Calibration**:
  - In Tuning: fitted prequentially per origin $t$ using only earlier tuning rows with $\text{origin} \le t - 2$ with minimum history and both classes present. Explicit `fit_through_origin <= origin - 2`. If conditions are unmet, falls back to raw probability.
  - Out of Tuning: a frozen Platt model is fit on Tuning labels through origin `178`, the latest available at Validation origin `180`, and applied to Validation and Confirmation.
- **Active Blend Weight Selection**:
  - Selected from a fixed grid: `[0.0, 0.25, 0.5, 0.75, 1.0]` by minimizing the Brier score on common eligible Tuning OOF predictions through origin `178`.
  - Tie-breaker: ties strictly prefer `1.0` (favoring the active model).
  - Validation and Confirmation labels **never** affect calibration parameters or weight selection. Confirmation is purely descriptive.

### 4. Matched Coverage Comparison
- For each month / origin $t$, the candidate blend matches the exact count of active accepted calls $k_t$ (15 to 20 indicators).
- Candidate ranks common eligible indicators by confidence $|P(\text{blend}) - 0.5|$ descending, with stable `indicator_id` ascending tie-breaking.
- Equal calls per origin are strictly asserted: $\text{calls}_{\text{candidate}}(t) \equiv \text{calls}_{\text{active}}(t) \equiv k_t$.
- Active comparator uses the original production accepted indicators and directions.

### 5. Origin-Blocked Paired Bootstrap & Gate Criteria
- Per-origin hit delta $\Delta(t) = \text{hits}_{\text{candidate}}(t) - \text{hits}_{\text{active}}(t)$ is evaluated using contiguous blocks of size 6 over 500 deterministic replicates (seed `20260727`).
- `nonlocked_gate_passed` requires:
  1. Validation total hit delta $> 0$
  2. Validation bootstrap $p_{10} \ge 0$
  3. Validation blend Brier score $<$ active Brier score
  4. Confirmation total hit delta $\ge 0$
  5. Zero locked evaluation reads (`locked_evaluation_read=False`, max origin $\le 266$)

### 6. Contamination Limit and Production Promotion Barrier
- **Contamination Caveat**: Pre-trained foundation models may have encountered public macroeconomic series during pre-training. Historical backtest outperformance may reflect training data memorization. Out-of-sample validity requires forward live or post-release paper trading.
- **Production Barrier**: `promotion_eligible` is **strictly False**. Any production promotion requires unblinding the locked evaluation window (origins 268-315), which cannot be done automatically by this research pipeline. Active production files remain untouched.
