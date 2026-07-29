# Original financial directional forecasting and selective ensemble plan

**Project type:** Monthly financial and macroeconomic directional forecasting  
**Research snapshot:** 2026-07-23  
**Dataset:** Approximately 300 monthly observations x 40 indicators  
**Primary objective:** Maximize the out-of-sample accuracy of accepted predictions, not the number of predictions  
**Final production output:** A calibrated ensemble that predicts Up/Down, estimates correctness, rejects weak predictions, and accepts at most 20 indicators per month; 15 is a soft coverage target, never a mandatory minimum

---

## 0. Executive Decision

This project will not be built around one model. It will be developed as a staged research system in which simple, classical, pretrained, and hybrid approaches compete under the same leakage-safe walk-forward evaluation.

The expected final system is:

```text
Point-in-time monthly data
        |
        v
Leakage-safe feature generation
        |
        +---------------- Classical branch ----------------+
        |                                                   |
        |   Global Elastic-Net Logistic Regression          |
        |   Shallow Global CatBoost                          |
        |   Dynamic Factor / State-Space expert              |
        |   Momentum and persistence baselines               |
        |                                                   |
        +--------------- Pretrained branch -----------------+
        |                                                   |
        |   Chronos-2 zero-shot probabilistic forecasts      |
        |   Chronos-2 zero-shot probabilistic forecasts      |
        |   TiRex-2 zero-shot multivariate forecasts          |
        |   Optional Chronos-2/MOMENT frozen embeddings       |
        |                                                   |
        +---------------------------------------------------+
                                |
                                v
                  Cross-fitted probability outputs
                                |
                                v
                   Constrained convex ensemble
                                |
                                v
                      Probability calibration
                                |
                                v
                  Correctness / reliability model
                                |
                                v
              Bootstrap lower confidence bound (LCB)
                                |
                                v
             Reject weak predictions and rank survivors
                                |
                                v
              Accept at most 20 reliable predictions
```

### Primary starting model

**Global Elastic-Net Logistic Regression** in long panel format.

### Primary nonlinear challenger

**Shallow Global CatBoost**.

### Primary pretrained forecasting experiment

**Chronos-2 zero-shot probabilistic forecasting**.

### Primary pretrained representation experiment

**Chronos-2 frozen encoder embeddings plus an Elastic-Net Logistic head**, tested only after the zero-shot forecasting experiment. MOMENT or TSPulse may be used as later representation challengers.

### Final decision rule

Use a calibrated estimate of expected correctness and select predictions by a conservative lower confidence bound. Never force the system to output 15 predictions when fewer than 15 pass the reliability threshold.

---

# 1. Problem Definition

For indicator `i` and month `t`:

```text
Target(i, t) = 1 if Value(i, t+1) > Value(i, t)
Target(i, t) = 0 otherwise
```

The system observes all information genuinely available at the end of month `t` and predicts the movement from `t` to `t+1`.

The task is simultaneously:

1. A global panel classification problem.
2. A partially pooled multi-task problem.
3. A probabilistic forecasting problem.
4. A ranking problem.
5. A selective classification and abstention problem.
6. A confidence calibration problem.

It is not merely 40 independent binary classification tasks. Although long format may contain approximately 12,000 indicator-month rows, the effective temporal sample size remains approximately 300 dates.


## 1.1 Mathematical and Selective Objective

For indicator `i` at month `t`:

```text
Y(i,t) = 1 if x(i,t+1) > x(i,t), otherwise 0
p(i,t) = P(Y(i,t)=1 | information genuinely available at t)
```

The directional prediction is `Up` when `p(i,t) >= 0.5`; otherwise it is `Down`. The system then makes a second decision, `accept` or `reject`, for every forecast.

The production objective is not ordinary accuracy over all 40 indicators. It is to maximize the expected correctness of accepted forecasts subject to:

```text
accepted_count(t) <= 20
correctness_LCB(i,t) >= selected_threshold
```

The threshold is selected using earlier walk-forward results only. When fewer than 15 forecasts pass, the system returns fewer than 15 rather than weakening the threshold.

## 1.2 Production outputs

For each indicator and month, store:

- Indicator ID.
- Predicted direction: Up or Down.
- Raw probability of Up.
- Calibrated probability of Up.
- Directional confidence.
- Expected probability that the prediction is correct.
- Conservative correctness lower bound.
- Accepted or rejected status.
- Selection rank.
- Rejection reason, when rejected.
- Historical out-of-sample accuracy for comparable predictions.
- Number of comparable historical predictions.
- Model agreement and disagreement.
- Forecast uncertainty.
- Data-quality status.
- Model and data version.

---

# 2. Non-Negotiable Research Rules

1. Never use random train-test splitting.
2. Split and bootstrap by date, keeping all 40 indicators from the same month together.
3. Fit every transformation inside each training fold.
4. Do not fit scaling, imputation, PCA, clustering, feature selection, calibration, ensemble weights, or thresholds on the full dataset.
5. Training labels available at forecast origin `t` end at feature row `t-1`.
6. Use point-in-time availability and historical vintages whenever possible.
7. Do not use revised macro data as if it had been known historically.
8. Do not select a model from one short favorable period.
9. Do not trust raw model probabilities.
10. Do not promote a model based only on accuracy.
11. Do not force the monthly output to contain 15-20 predictions.
12. Preserve an untouched final audit period.
13. Record every attempted experiment, including failed experiments.
14. Prefer the simpler model when performance is statistically indistinguishable.
15. Treat deep learning and foundation models as hypotheses, not guaranteed improvements.


## 2.1 Critical Risk Register

| Risk | Why it matters | Mandatory control |
|---|---|---|
| Effective sample size | The panel has about 300 independent dates, not 12,000 independent rows. | Split, tune, bootstrap, and compare models by date; give each month equal aggregate weight. |
| Point-in-time availability | A value labelled month `t` may have been released after the forecast cutoff. | Maintain cell-level timestamps or indicator release lags; otherwise apply a conservative lag. |
| Revised macro history | Latest revised values may contain information unavailable in real time. | Use historical vintages where possible; otherwise label the exercise revised-data pseudo-out-of-sample. |
| Anonymous indicators | Units, transformations, and revision behavior may be unknown. | Maintain a restricted data dictionary with unit, positivity, lag, revision, and carry-forward flags. |
| Structural breaks | Relationships may reverse across regimes. | Compare expanding, rolling, and recency-weighted experts; monitor change points and drift. |
| Cross-indicator dependence | Many forecasts can fail together in the same month. | Compute intervals and paired comparisons using contiguous date blocks, not individual rows. |
| Multiple testing | Many models, features, pairs, and lags can produce false discoveries. | Pre-register families, cap trials, apply FDR to exploratory relations, and preserve all results. |
| Selection bias | Top-ranked probabilities can appear accurate simply because the selector was tuned on them. | Evaluate the full ranking and acceptance policy through Level-C cross-fitting. |
| Miscalibration | A raw `0.70` score may not mean 70% correctness. | Use rolling beta or Platt calibration and a separate correctness model. |
| Forced coverage | Filling a quota can admit weak forecasts. | Treat 15 as a soft target, keep 20 as the hard cap, and never lower the reliability floor to fill the list. |
| Foundation-model contamination | Public financial histories may overlap pretraining data. | Include genuinely fresh or post-release audit months and live paper trading when possible. |
| Experiment over-search | A lucky configuration may win a short backtest. | Limit trials, keep a locked audit, and apply Reality Check or Model Confidence Set procedures after broad comparison. |

The largest non-model risk is **data observability**. No model or validation technique can repair a backtest that used values not actually available at the historical forecast cutoff.

---

# 3. Data Audit and Point-in-Time Reconstruction

This stage must be completed before modeling.

## 3.1 Required checks

- Verify dates are sorted and monthly.
- Detect missing months and duplicate dates.
- Quantify missing values by indicator and period.
- Detect carried-forward or stale values.
- Count exact zero changes.
- Detect extreme outliers and discontinuities.
- Identify indicators that rarely change.
- Determine whether levels can be negative or near zero.
- Determine whether percentage changes and logarithms are mathematically valid.
- Detect changes in scale, rebasing, or methodology.
- Determine publication lag for every indicator.
- Determine whether observations are preliminary, revised, or final.
- Detect survivorship bias and series replacement.

## 3.2 Availability table

Create an availability configuration:

```yaml
IND_01:
  publication_lag_months: 0
  revision_policy: unknown
  allow_log_return: true
  allow_percentage_change: true
  carry_forward_valid: false

IND_02:
  publication_lag_months: 1
  revision_policy: revised
  allow_log_return: false
  allow_percentage_change: false
  carry_forward_valid: true
```

If metadata are unavailable, apply conservative assumptions and clearly label the final study as a revised-data pseudo-out-of-sample backtest rather than a true vintage backtest.

## 3.3 Missing values

Permitted strategies:

- Training-only median or robust local imputation.
- Forward fill only when operationally valid for the indicator.
- Missing-value flags.
- Time-since-last-observation feature.
- Model-native missing handling for CatBoost.

Forbidden:

- Backward filling from future observations.
- Interpolation using future points at prediction time.
- Full-dataset imputation before validation.

---

# 4. Target Strategies to Test

The official target remains binary direction, but several compatible formulations will be compared.

## 4.1 Direct binary classification

```text
Input features -> P(Up)
```

This is the primary objective because it directly matches the decision.

## 4.2 Numerical forecast converted to direction

```text
Forecast Value(t+1) or Change(t+1)
Direction = sign(Forecast - Current Value)
```

Use for ARIMA, ETS, state-space models, and pretrained forecasting models.

## 4.3 Distribution or quantile forecast

Estimate the distribution of the next value or next change, then compute:

```text
P(Up) = P(Value(t+1) > Value(t))
```

This is the preferred pretrained-model formulation because it exposes uncertainty.

## 4.4 Joint classification and regression

Potential later experiment:

```text
Loss = BinaryClassificationLoss + lambda * QuantileOrRegressionLoss
```

This is not an initial experiment because it increases training variance.

## 4.5 Dead-zone auxiliary target

Create an auxiliary label that separates tiny or unchanged moves from meaningful movement:

```text
Up        if change > +delta_i
Down      if change < -delta_i
Uncertain otherwise
```

The official evaluation remains binary. The dead-zone target is used only as a reliability feature or secondary model.

---

# 5. Leakage-Safe Feature Plan

Keep the first feature set intentionally compact.

## 5.1 Own-indicator features

Initial features:

- Current robustly normalized level.
- First difference.
- Percentage change, only when valid.
- Log return, only when strictly valid.
- Direction lags: 1, 2, 3, 6, and 12 months.
- Value/change lags: 1, 2, 3, 6, and 12 months.
- Momentum: 3, 6, and 12 months.
- Rolling median: 3, 6, and 12 months.
- Rolling mean: 3, 6, and 12 months.
- Rolling standard deviation and MAD.
- Robust rolling z-score.
- Distance from 6- and 12-month moving average or median.
- Robust trend slope.
- Acceleration.
- Drawdown.
- Direction streak.
- Sign-change frequency.
- Missing-value indicator.
- Time since last valid observation.

Postpone until needed:

- Rolling skewness.
- Rolling kurtosis.
- Many autocorrelation lags.
- Large technical-indicator libraries.
- Thousands of automatically generated features.

## 5.2 Cross-sectional and shared features

Initial shared features:

- Percentage of indicators rising.
- Cross-sectional median change.
- Cross-sectional dispersion.
- Cross-sectional rank of each indicator's recent return.
- Relative strength versus the cross-sectional median.
- First 2-5 PCA factors.
- Factor momentum and volatility.
- Indicator cluster ID, if clusters are stable.
- Cluster-level mean and dispersion.

## 5.3 Transformation eligibility

Inside each training fold:

1. Always allow raw differences and ranks.
2. Allow log return only for strictly positive, stable series.
3. Allow percentage change only when near-zero denominators are not a problem.
4. Use robust scaling based on median and MAD.
5. Use fold-local validation to choose among a small set of transformations.
6. Do not use full-series stationarity tests to choose the transformation.

## 5.4 Cross-indicator lead-lag features

Only add a cross-indicator feature when it:

- Survives false-discovery control.
- Has the same sign in several training subperiods.
- Adds out-of-fold value after PCA or dynamic factors are included.
- Remains useful under more than one training window.
- Is limited to a maximum of 1-3 selected external features per target indicator.

---

# 6. Model Groups and How They Combine

This section distinguishes features, competing models, model alternatives, and ensemble components.

## 6.1 Group A: Baselines

These are required reference models and may also be retained as low-complexity ensemble experts.

### Required baselines

1. Historical majority class.
2. Last-direction continuation.
3. Last-direction reversal.
4. 3-, 6-, and 12-month momentum rules.
5. Mean-reversion / moving-average rule.
6. AR(1) and AR(2).
7. Exponential smoothing or local-level state-space model.

### Combination rule

Baselines may enter the final ensemble if their out-of-sample predictions add diversity and receive positive constrained weights.

Do not assume that a baseline is too simple to be useful. In noisy financial data, a simple expert can improve robustness.

---

## 6.2 Group B: Classical panel models

### B1. Global Elastic-Net Logistic Regression - mandatory first model

Long-format structure:

```text
Date | Indicator_ID | Own Features | Shared Factors | Target
```

Model structure:

```text
Shared feature coefficients
+ indicator-specific intercept
+ a very small number of indicator-specific interactions
```

Recommended indicator-specific interactions:

- Lag-1 direction.
- 3-month momentum.
- 6-month volatility.
- Distance from moving average.

Why it is first:

- Directly predicts probability.
- Pools information across indicators.
- Strong regularization controls overfitting.
- Easy to calibrate and interpret.
- Provides a strong benchmark for every complex model.

### B2. Shallow Global CatBoost - mandatory nonlinear challenger

Use:

- One global model.
- Indicator ID as an input.
- Depth 2-5.
- Strong L2 regularization.
- Low learning rate.
- Early stopping by date-based validation.
- Compact feature set.

Do not start with 40 separate CatBoost models.

### B3. XGBoost and LightGBM - alternatives, not simultaneous defaults

Run CatBoost first. Test XGBoost and LightGBM as challengers. Retain only the best stable tree family or at most two genuinely diverse tree experts.

### B4. Per-indicator Logistic Regression - optional partial-pooling expert

Use only:

- Strong regularization.
- Compact own-history features.
- Shared factors.
- Shrinkage toward the global model.

Per-indicator models enter the ensemble only for indicators where they demonstrate stable out-of-sample improvement.

### B5. Hierarchical Bayesian Logistic Regression - strong alternative architecture

Use if the global panel model shows stable indicator heterogeneity that fixed effects cannot capture.

### B6. Explainable Boosting Machine - optional interpretable nonlinear benchmark

Use as a research benchmark, not a mandatory production component.

### B7. Random Forest and Extra Trees - benchmark only

These are not primary production candidates because their probability estimates can be unstable with few independent dates.

---

## 6.3 Group C: Statistical and latent-factor experts

### C1. PCA factors - first shared representation

Fit PCA inside every fold and keep 2-5 factors.

PCA is a feature generator, not automatically a separate forecast model.

### C2. Dynamic Factor Model - next statistical expert

Two uses:

1. Generate filtered latent factors for Logistic Regression and CatBoost.
2. Produce a standalone statistical forecast that may enter the ensemble.

### C3. State-space / Kalman expert

Use a local-level, local-trend, or factor state-space model to provide:

- Point forecast.
- Forecast variance.
- Trend estimate.
- Filtered state uncertainty.

### C4. HMM or Markov-switching model

Use primarily to generate filtered regime probabilities. Do not use smoothed probabilities because they use future information.

### C5. Bayesian VAR or factor-augmented VAR

Test only after dimensionality reduction. Do not fit an unrestricted 40-variable VAR.

---

## 6.4 Group D: Pretrained forecasting models

### D1. Chronos-2 - primary pretrained model

Initial usage:

- Zero-shot probabilistic or quantile forecasting.
- Univariate and multivariate experiments.
- Convert the predictive distribution into probability of Up.
- Retain interval width, asymmetry, and dispersion as uncertainty features.

Second usage:

- Extract frozen encoder embeddings.
- Train a small shared Elastic-Net Logistic head.

Third usage, only after success:

- LoRA with rank 2-4, conservative learning rate, and nested walk-forward validation.

Do not begin with full fine-tuning.

### D2. TiRex-2 - primary multivariate zero-shot challenger

Use TiRex-2 as a probabilistic joint-multivariate expert with past and future-known covariates where applicable. Request its supported quantiles and convert them into `P(Up)`. Retain interval width, asymmetry, and failure flags as reliability inputs.

The current open release should be treated as an inference-first model: do not assume a stable embedding interface, ordinary fine-tuning workflow, or classification-head API. Pin the exact release and validate its package behavior before implementation.

### D3. Frozen representation experiments - secondary

Test these only after the zero-shot experts are established:

1. Chronos-2 encoder embeddings plus an Elastic-Net Logistic classifier.
2. MOMENT embeddings plus an Elastic-Net Logistic classifier.
3. TSPulse embeddings as a later lightweight challenger.

All representation models must be fully cross-fitted. High-dimensional embeddings may be compressed with fold-local PCA or PLS before the classifier.

### D4. TimesFM - LoRA and forecasting challenger

Initial use:

- Zero-shot or quantile forecast benchmark.

Later use:

- Conservative LoRA experiment if Chronos-2 and TiRex-2 experiments show that pretrained adaptation is useful.

### D5. MOMENT - representation challenger

Use frozen embeddings plus a linear classifier. Compare against Chronos-2 embeddings and, later, TSPulse embeddings.

### D6. Secondary pretrained models

Research-only or lower-priority experiments:

- Chronos-Bolt.
- Moirai / Uni2TS.
- IBM Granite FlowState.
- IBM TSPulse / PatchTST foundation variants.
- IBM Tiny Time Mixers after frequency compatibility verification.
- TabPFN-TS.
- FinCast.
- Toto.
- Sundial.
- Lag-Llama.
- Time-MoE.
- Timer / Timer-XL.
- TempoPFN.
- Reverso.
- TimeGPT, only if data governance permits external API use.

### D7. Models not recommended initially

- Kronos for this dataset format, because it is specialized for candlestick/OHLCV-style financial data.
- Full fine-tuning of any foundation model.
- Multiple large pretrained backbones in the first production version.

---

## 6.5 Group E: From-scratch deep learning

These are not initial priorities.

Potential later benchmarks:

- Small TCN.
- MiniRocket or ROCKET plus linear classifier.
- Echo State Network / reservoir model.
- Small global DeepAR-style probabilistic model.
- Small N-BEATS or N-HiTS benchmark.

Postpone:

- LSTM or GRU trained from scratch.
- Bidirectional LSTM.
- Temporal Fusion Transformer.
- PatchTST trained from scratch.
- iTransformer.
- TimesNet.
- TimeMixer.
- Informer, Autoformer, FEDformer.
- Mamba-based model trained from scratch.
- Large neural multi-task model with 40 deep heads.

Deep learning is promoted only if it beats the classical and pretrained hybrid system under the same locked evaluation.

---

## 6.6 Group F: Graph approaches

Do not begin with an end-to-end GNN.

Permitted graph-related research:

- Shrinkage correlation network.
- Graphical Lasso network.
- Stable lead-lag network.
- Cluster graph used for feature generation.

Postpone:

- Graph Attention Network.
- Temporal GNN.
- Dynamic graph neural network.
- End-to-end graph structure learning.

Reason: both the graph and the predictor would be learned from only approximately 300 dates.

---

# 7. Valid and Invalid Combinations

## 7.1 Valid combinations

### Combination A: Classical linear model

```text
Own features + PCA factors + Indicator ID
-> Global Elastic-Net Logistic Regression
```

### Combination B: Classical nonlinear model

```text
Own features + PCA factors + regime features + Indicator ID
-> Shallow Global CatBoost
```

### Combination C: Pretrained zero-shot expert

```text
Historical sequence
-> Chronos-2
-> quantiles / samples
-> P(Up) + uncertainty
```

### Combination D: Pretrained representation classifier

```text
Historical sequence
-> frozen Chronos-2, MOMENT, or TSPulse embeddings
-> optional fold-local compression
-> shared Elastic-Net Logistic head
-> P(Up)
```

### Combination E: Forecast features inside a local model

```text
Chronos forecast median
+ Chronos P(Up)
+ Chronos interval width
+ local engineered features
-> Elastic-Net Logistic or CatBoost
```

### Combination F: Final model ensemble

```text
Global Logistic probability
+ CatBoost probability
+ statistical expert probability
+ Chronos probability
+ TiRex-2 probability
+ baseline probability
-> constrained convex ensemble
```

### Combination G: Reliability model

```text
Calibrated ensemble probability
+ model disagreement
+ bootstrap variability
+ pretrained interval width
+ regime distance
+ data-quality features
+ recent out-of-sample performance
-> probability that the prediction is correct
```

## 7.2 Combinations to avoid initially

1. CatBoost, XGBoost, LightGBM, Random Forest, and Extra Trees all in the first production ensemble.
2. Chronos zero-shot, Chronos forecast features, Chronos embeddings, and Chronos LoRA all simultaneously before individual ablation tests.
3. Full fine-tuning of Chronos-2, TimesFM, and TiRex-2 in parallel.
4. Forty separate boosted-tree models.
5. A large MLP on top of high-dimensional pretrained embeddings.
6. PCA fitted once on the entire dataset.
7. A GNN built on correlations computed from the complete history.
8. A top-20 rule based only on raw model probability.
9. A dynamic ensemble or regime router before a fixed ensemble is validated.

## 7.3 Alternative groups

Choose one starting member, then compare challengers:

| Group | Starting choice | Challengers |
|---|---|---|
| Tree model | CatBoost | XGBoost, LightGBM, EBM |
| Shared factors | PCA | Dynamic Factor Model, Sparse PCA, PLS |
| Pretrained forecast | Chronos-2 | TiRex-2, TimesFM, Moirai |
| Pretrained representation | Chronos-2 embeddings | MOMENT, TSPulse |
| Head | Elastic-Net Logistic | Small MLP, compressed-embedding CatBoost |
| Adaptation | Frozen backbone | LoRA, adapter, partial unfreezing |
| Calibration | Beta calibration | Platt, Venn-Abers, isotonic only with adequate data |
| Statistical forecast | Local-level/state-space | ARIMA, ETS, factor-augmented VAR |

---

# 8. Exact Experiment Phases

Each phase has a promotion gate. Do not skip a gate.

## Phase 0: Data validity

### Deliverables

- Data audit report.
- Availability and revision table.
- Leakage tests.
- Missing-value policy.
- Target statistics.

### Gate

No modeling begins until the pipeline can prove that changing observations after date `t` does not change any prediction through `t`.

---

## Phase 1: Required baselines

### Models

- Historical majority.
- Persistence.
- Reversal.
- Momentum 3/6/12.
- Mean reversion.
- AR(1)/AR(2).
- Simple state-space or ETS.

### Outputs

- Walk-forward predictions.
- Probability estimates.
- Baseline accuracy, Brier score, and calibration.

### Gate

The learned model must beat or complement these baselines at matched coverage.

---

## Phase 2: Classical global system

### Models

1. Global Elastic-Net Logistic Regression.
2. Shallow Global CatBoost.
3. PCA factor expert.
4. Dynamic Factor / state-space challenger.

### Feature set

- Compact own-series features.
- PCA factors.
- Breadth and dispersion.
- Indicator ID.
- Missingness and staleness features.

### Gate

Promote a model only if it improves at least one proper scoring rule and does not create severe instability in rolling selected accuracy.

---

## Phase 3: Indicator heterogeneity

### Experiments

- Strongly regularized per-indicator Logistic Regression.
- Mixed-effects Logistic Regression.
- Hierarchical Bayesian Logistic Regression.
- Low-rank multi-task Logistic Regression.

### Gate

Retain indicator-specific components only where improvement is stable and not concentrated in a short period.

---

## Phase 4: Pretrained zero-shot forecasting

### Models

1. Chronos-2.
2. TiRex-2.
3. TimesFM as a challenger.

### Outputs

- Median forecast.
- Quantiles or samples.
- Probability of Up.
- Interval width.
- Distribution asymmetry.
- Failure and out-of-range flags.

### Gate

A pretrained model is useful if either:

- Its probability improves ensemble Brier/log loss and selective risk, or
- Its uncertainty improves the reliability model, even if its standalone accuracy is neutral.

---

## Phase 5: Pretrained representation classification

### Experiments

1. Chronos-2 frozen embedding plus Elastic-Net Logistic head.
2. MOMENT frozen embedding plus Elastic-Net Logistic head.
3. TSPulse frozen embedding plus Elastic-Net Logistic head.

### Optional second stage

- PCA or PLS compression inside each fold.
- Small 16-32 unit MLP.
- Shallow CatBoost on compressed embeddings.

### Gate

The representation model must beat the raw-feature Logistic model or add stable ensemble diversity.

---

## Phase 6: Pretrained adaptation

Run only after Phase 4 or 5 succeeds.

### Experiments

- Chronos-2 LoRA rank 2-4.
- TimesFM LoRA.
- Small feature-space adapter.
- Last-layer or final-block training, only after LoRA.

### Prohibited at this stage

- Full fine-tuning.
- Multiple simultaneous adapters without ablation.

### Gate

The adapted model must beat its own frozen or zero-shot version under nested walk-forward validation and the locked audit.

---

## Phase 7: Fixed ensemble

### Candidate components

- Global Elastic-Net Logistic.
- Shallow CatBoost.
- Best statistical expert.
- Best naive baseline.
- Chronos-2 zero-shot probability.
- TiRex-2 zero-shot probability.
- Optional Chronos-2 or MOMENT embedding-head probability.

### Ensemble method

Use constrained convex stacking:

```text
weight_m >= 0
sum(weight_m) = 1
```

Regularize ensemble weights toward the global Logistic model.

### Gate

The ensemble must outperform:

- The best single model.
- Equal-weight averaging.
- The global Logistic anchor.

Evaluation must use cross-fitted ensemble predictions, not in-sample stacker predictions.

---

## Phase 8: Calibration

### Primary methods

1. Beta calibration.
2. Platt scaling as a stability alternative.

### Avoid initially

- Per-indicator isotonic calibration.
- Complex neural calibration.

### Evaluation

- Calibration intercept.
- Calibration slope.
- Brier score.
- Log loss.
- Reliability diagrams.
- Expected calibration error.

---

## Phase 9: Correctness and reliability model

Create a historical out-of-fold correctness label:

```text
Correct = 1 if historical predicted direction was correct
Correct = 0 otherwise
```

Inputs:

- Calibrated directional confidence.
- Expert probability standard deviation.
- Number of agreeing experts.
- Ensemble entropy.
- Chronos-2/TiRex-2 interval width.
- Bootstrap prediction variability.
- Recent out-of-sample accuracy.
- Indicator or indicator-cluster effect.
- Regime distance.
- Nearest historical analog distance.
- Change-point probability.
- Missingness and staleness.
- Predicted direction.

Output:

```text
P(Current directional prediction is correct)
```

Use a strongly regularized Logistic or hierarchical Logistic model first.

---

## Phase 10: Selective prediction

For each candidate, compute a date-block-bootstrap distribution of expected correctness.

Use the 10th percentile as the conservative score:

```text
Correctness_LCB = 10th percentile of bootstrapped expected correctness
```

Selection gates:

1. Data are available and valid.
2. Current feature state lies within acceptable historical support.
3. Optional conformal output is a singleton direction.
4. Correctness lower bound exceeds the selected threshold.
5. Rank passing predictions by correctness lower bound.
6. Accept at most 20.

If fewer than 15 pass, accept fewer than 15.

---

# 9. Exact Walk-Forward Validation Design

Assuming 300 monthly rows:

## 9.1 Proposed ranges

| Stage | Approximate rows/origins | Purpose |
|---|---:|---|
| Feature warm-up | Months 1-12 | Build 12-month features |
| Initial estimation | Through month 119 | Minimum training history |
| Development walk-forward | Origins 120-251 | Feature, model, calibration, ensemble, and threshold development |
| Locked final audit | Origins 252-299 | Predict months 253-300 without redesign |
| Production fit | Through month 300 | Forecast month 301 |

## 9.2 Information available at origin `t`

```text
Training feature rows: <= t-1
Training targets:      <= t-1
Inference feature row:  t
Forecast target:        movement from t to t+1
```

## 9.3 Window strategies

Compare:

- Expanding history.
- Last 180 months.
- Last 120 months.
- Last 60 months for simple models only.
- Exponentially weighted history with half-lives 36, 60, and 120 months.

A multi-window ensemble may be tested after the fixed models are stable.

## 9.4 Nested evaluation layers

### Layer A: Base-model OOF

Train each base model only on earlier dates and predict the next date.

### Layer B: Ensemble OOF

At each meta-origin, fit ensemble weights only on earlier Layer-A predictions and predict the current Layer-A row.

### Layer C: Calibration and reliability OOF

At each later origin:

- Fit calibration using earlier Layer-B predictions.
- Fit the correctness model using earlier calibrated predictions.
- Apply the selection policy to the current origin.

Only Layer-C predictions evaluate the complete final policy.

## 9.5 Hyperparameter schedule

- Use conservative random search or Optuna.
- Limit each family to approximately 20-30 carefully selected trials.
- Re-optimize no more than once per simulated year.
- Carry the chosen configuration forward between re-optimization dates.
- Do not tune on the locked 48-month audit.

---

# 10. Safe Hyperparameter Search Spaces

## 10.1 Elastic-Net Logistic

```yaml
C: log-uniform [1e-3, 10]
l1_ratio: [0.0, 0.25, 0.5, 0.75, 1.0]
window: [120, 180, expanding]
recency_half_life: [36, 60, 120, null]
n_factors: [1, 2, 3, 4, 5]
```

## 10.2 CatBoost

```yaml
depth: [2, 3, 4, 5]
learning_rate: [0.01, 0.02, 0.04, 0.08]
iterations: [100, 200, 400, 600]
l2_leaf_reg: [3, 5, 10, 20, 30]
random_strength: conservative
one_hot_indicator_id: true
```

## 10.3 Frozen embedding head

```yaml
head: [elastic_net_logistic]
embedding_pooling: [last, mean, attention_if_officially_supported]
dropout: [0.0, 0.2, 0.4]
pca_dimensions: [none, 16, 32, 64]
C: log-uniform [1e-3, 10]
```

## 10.4 Small MLP head - later only

```yaml
hidden_size: [16, 32]
dropout: [0.2, 0.4]
weight_decay: [1e-4, 1e-3, 1e-2]
early_stopping: true
```

## 10.5 LoRA - later only

```yaml
rank: [2, 4, 8]
alpha: [4, 8, 16]
learning_rate: very_small
weight_decay: strong
early_stopping: true
```

---

# 11. Ensemble Construction

## 11.1 Base probability matrix

For each indicator-month:

```text
p_logistic
p_catboost
p_statistical
p_momentum
p_chronos2
p_tirex_head
p_tirex2_optional
```

## 11.2 Constrained convex ensemble

```text
p_ensemble = sum(weight_m * p_m)
```

Constraints:

- All weights are nonnegative.
- Weights sum to one.
- Weights are fitted from previous OOF predictions only.
- A regularization term shrinks weights toward the global Logistic anchor.
- A model that does not add incremental value may receive weight zero.

## 11.3 Ensemble ablations

Evaluate:

1. Logistic only.
2. Logistic + CatBoost.
3. Logistic + CatBoost + statistical expert.
4. Add Chronos-2.
5. Add TiRex-2 zero-shot.
6. Add the best validated frozen embedding head if useful.
7. Equal-weight versus optimized weights.

Do not keep a component merely because it is modern.

---

# 12. Calibration and Confidence

## 12.1 Directional probability

After calibration:

```text
P_up_calibrated
Predicted direction = Up if P_up_calibrated >= 0.5 else Down
Directional confidence = max(P_up_calibrated, 1 - P_up_calibrated)
```

## 12.2 Expected correctness

The reliability model estimates:

```text
P(predicted direction is correct | confidence, disagreement, uncertainty, regime, quality)
```

This is the score used for selection, not raw `P(Up)`.

## 12.3 Historical accuracy for similar predictions

Define similar historical OOF predictions by:

- Same indicator or stable indicator cluster.
- Same predicted direction.
- Similar expected-correctness band.
- Similar expert agreement.
- Similar volatility and factor regime.
- Similar data-quality state.

Report:

- Match count.
- Recency-weighted accuracy.
- Beta-binomial shrunk accuracy.
- Date-block-bootstrap interval.

Do not display an unqualified percentage for a tiny match set.

---

# 13. Final Monthly Decision Rule

For month `t`:

1. Validate the point-in-time data snapshot.
2. Generate fold-consistent features.
3. Produce all retained base-model probabilities.
4. Apply the cross-fitted ensemble specification.
5. Calibrate the ensemble probability.
6. Generate reliability meta-features.
7. Predict expected correctness.
8. Bootstrap by historical date blocks.
9. Calculate the correctness lower confidence bound.
10. Apply hard gates:
   - valid data;
   - acceptable historical support;
   - no critical model error;
   - minimum correctness lower bound;
   - optional singleton conformal direction.
11. Rank passing predictions by correctness lower bound.
12. Select at most 20.
13. If fewer than 15 pass, return fewer than 15.
14. Optionally show the next highest rejected predictions as a watchlist, clearly labeled rejected.

---

# 14. Evaluation Metrics

## 14.1 Direction classification

- Accuracy.
- Balanced accuracy.
- Precision for Up.
- Precision for Down.
- F1.
- Matthews correlation coefficient.
- ROC-AUC when class variation permits.

## 14.2 Probability quality

- Brier score.
- Log loss.
- Calibration intercept.
- Calibration slope.
- Expected calibration error.
- Reliability diagrams.

## 14.3 Selective prediction

Primary metrics:

- Accuracy of accepted predictions.
- Number accepted per month.
- Coverage.
- Selective risk.
- Risk-coverage curve.
- Area under the risk-coverage curve.
- Top-5, Top-10, Top-15, and Top-20 accuracy.
- Accuracy by confidence bucket.
- Accuracy at matched coverage versus baselines.

## 14.4 Stability

- Per-indicator performance.
- Per-month performance.
- Rolling 12- and 24-month performance.
- Performance by regime.
- Performance by Up versus Down prediction.
- Performance by missingness and staleness.
- Performance near detected change points.
- Performance by model agreement level.

## 14.5 Statistical confidence

- Bootstrap contiguous blocks of dates, not individual rows.
- Use block lengths approximately 6-12 months.
- Compare paired monthly losses at matched coverage.
- Report 90% and 95% uncertainty intervals.
- Apply data-snooping corrections after many model comparisons.

A nominal 65% accuracy is not considered credible unless it is stable over enough independent months, exceeds same-coverage baselines, and has a defensible lower confidence bound.

---

# 15. Experiment Priority Table

| Priority | Experiment | Hypothesis | Success criterion | Complexity |
|---:|---|---|---|---|
| 0 | Data availability and leakage audit | Historical simulation is valid | Future-perturbation and point-in-time tests pass | Medium |
| 1 | Naive and statistical baselines | Simple predictability exists | Honest benchmark established | Low |
| 2 | Global Elastic-Net Logistic | Partial pooling improves generalization | Better Brier/selective risk than baselines | Low |
| 3 | PCA/Dynamic factors | Shared market state adds signal | Stable OOF improvement | Low-Medium |
| 4 | Shallow Global CatBoost | Limited nonlinearities are useful | Improves proper score or matched-coverage accuracy | Medium |
| 5 | Per-indicator shrinkage | Some indicators need custom coefficients | Stable improvement for supported indicators | Medium |
| 6 | Chronos-2 zero-shot | Pretraining adds forecasting or uncertainty value | Incremental OOF ensemble or reliability gain | Medium-High |
| 7 | TiRex-2 zero-shot | Joint multivariate pretraining is useful | Improves probability, uncertainty, or ensemble metrics | Medium-High |
| 8 | Chronos-2/MOMENT frozen linear probe | Pretrained representations encode reusable patterns | Beats raw-feature head or adds stable diversity | Medium-High |
| 9 | MOMENT representation | Alternative pretrained representation helps | Stable linear-probe improvement | Medium-High |
| 10 | Fixed constrained ensemble | Experts are complementary | Beats best component and equal average | Medium |
| 11 | Beta/Platt calibration | Raw probabilities are miscalibrated | Lower Brier/log loss and improved slope | Low |
| 12 | Correctness meta-model | Reliability can be predicted | Better risk-coverage curve | Medium |
| 13 | Bootstrap LCB selection | Conservative ranking is safer | Better locked selected accuracy with stable coverage | Medium-High |
| 14 | Multi-window experts | Recent regimes require different history | Better worst-regime stability | Medium |
| 15 | Chronos/TimesFM LoRA | Light adaptation adds domain value | Beats frozen/zero-shot under nested validation | High |
| 16 | MiniRocket/Reservoir | Fixed sequence features add diversity | Reproducible incremental ensemble gain | Medium |
| 17 | Hierarchical dynamic Logistic | Coefficients need temporal adaptation | Stable improvement over fixed Logistic | High |
| 18 | From-scratch deep learning | Local patterns require learned sequence model | Strong locked-audit gain | Very High |
| 19 | GNN | Learned graph adds value beyond factors | Strong stable gain after all simpler models | Very High |

---

# 16. Promotion and Rejection Criteria

## 16.1 Promote a model when

- It improves Brier score or log loss.
- It improves selected accuracy at matched coverage.
- It reduces area under the risk-coverage curve.
- Improvement appears across several rolling windows.
- The gain is not concentrated in one indicator or one short period.
- Calibration remains acceptable.
- The model adds ensemble diversity.
- The result survives the locked audit.

## 16.2 Reject or postpone a model when

- It improves training accuracy but not OOF performance.
- It requires many hyperparameter trials to find one favorable result.
- It is unstable across seeds or folds.
- Its probability is badly calibrated.
- It duplicates an existing component without adding value.
- Its performance is dependent on revised or unavailable data.
- Its gain disappears at matched coverage.
- It increases complexity without improving the lower confidence bound.

---

# 17. Error Analysis Plan

For every retained and rejected model, break errors down by:

- Indicator.
- Date.
- Predicted direction.
- Confidence bucket.
- Market/factor regime.
- Window type.
- Missingness and staleness.
- Change-point proximity.
- Pretrained versus classical disagreement.
- Zero-change outcome.
- Large versus small realized movement.
- Model agreement count.

Investigate whether errors arise from:

- Calibration failure.
- Directional-model failure.
- Regime shift.
- Incorrect availability assumptions.
- Weak historical support.
- A specific indicator family.
- Selection-model bias.

---

# 18. Explainability

## 18.1 Global Logistic

Report:

- Standardized coefficients.
- Sign stability across folds.
- Indicator intercepts.
- Factor contributions.

## 18.2 CatBoost

Report:

- SHAP values.
- Feature-importance stability across folds.
- Local explanation for each accepted prediction.

## 18.3 Pretrained models

Do not fabricate causal explanations. Report observable evidence:

- Forecast quantiles.
- Probability of Up.
- Interval width.
- Agreement with other experts.
- Embedding-head contribution.

## 18.4 Selection reason templates

Examples:

- "Accepted: four of five validated experts agree, bootstrap variability is low, and similar OOF cases were historically reliable."
- "Accepted: the global Logistic model and Chronos-2 independently support the same direction, with a narrow predictive interval."
- "Rejected: high model disagreement and current factor state lies outside historical support."
- "Rejected: expected-correctness lower bound is below the production threshold."

---

# 19. Repository Structure

```text
financial-direction-selection/
├── README.md
├── PLAN.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── dvc.yaml
├── configs/
│   ├── data/
│   │   ├── sources.yaml
│   │   ├── availability.yaml
│   │   └── indicators.yaml
│   ├── features/
│   │   ├── compact.yaml
│   │   └── transformations.yaml
│   ├── models/
│   │   ├── baselines.yaml
│   │   ├── global_logit.yaml
│   │   ├── catboost.yaml
│   │   ├── factor_model.yaml
│   │   ├── indicator_models.yaml
│   │   └── pretrained.yaml
│   ├── validation/
│   │   ├── development.yaml
│   │   └── locked_audit.yaml
│   ├── calibration/
│   ├── ensemble/
│   ├── selection/
│   └── production/
├── data/
│   ├── raw/
│   ├── vintages/
│   ├── interim/
│   ├── processed/
│   └── schemas/
├── src/
│   └── forecast_select/
│       ├── data/
│       │   ├── load.py
│       │   ├── schema.py
│       │   ├── audit.py
│       │   ├── availability.py
│       │   └── vintage.py
│       ├── targets/
│       │   ├── direction.py
│       │   └── dead_zone.py
│       ├── features/
│       │   ├── transforms.py
│       │   ├── univariate.py
│       │   ├── cross_sectional.py
│       │   ├── factors.py
│       │   ├── regimes.py
│       │   └── lead_lag.py
│       ├── models/
│       │   ├── baselines.py
│       │   ├── global_logit.py
│       │   ├── catboost_panel.py
│       │   ├── indicator_logit.py
│       │   ├── dynamic_factor.py
│       │   ├── state_space.py
│       │   ├── hierarchical_logit.py
│       │   └── pretrained/
│       │       ├── chronos2.py
│       │       ├── tirex.py
│       │       ├── tirex2.py
│       │       ├── timesfm.py
│       │       ├── moment.py
│       │       └── distribution.py
│       ├── validation/
│       │   ├── splitters.py
│       │   ├── walk_forward.py
│       │   ├── crossfit.py
│       │   └── oof_store.py
│       ├── ensemble/
│       │   ├── convex_stack.py
│       │   └── online_weights.py
│       ├── calibration/
│       │   ├── beta.py
│       │   ├── platt.py
│       │   └── conformal.py
│       ├── selection/
│       │   ├── reliability.py
│       │   ├── bootstrap.py
│       │   ├── policy.py
│       │   └── reasons.py
│       ├── evaluation/
│       │   ├── classification.py
│       │   ├── calibration.py
│       │   ├── selective.py
│       │   ├── stability.py
│       │   └── statistics.py
│       ├── inference/
│       │   ├── pipeline.py
│       │   ├── contracts.py
│       │   └── ledger.py
│       └── monitoring/
│           ├── drift.py
│           ├── performance.py
│           └── alerts.py
├── scripts/
│   ├── audit_data.py
│   ├── build_features.py
│   ├── run_baselines.py
│   ├── run_backtest.py
│   ├── run_pretrained_cache.py
│   ├── run_locked_audit.py
│   ├── train_final.py
│   ├── predict_month.py
│   └── monitor.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   │   ├── test_future_perturbation.py
│   │   ├── test_fold_local_fit.py
│   │   ├── test_label_availability.py
│   │   └── test_point_in_time.py
│   └── regression/
├── artifacts/
│   ├── oof_predictions/
│   ├── pretrained_cache/
│   ├── model_registry/
│   └── forecast_ledgers/
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── model_cards/
│   └── experiments/
└── docs/
    ├── methodology.md
    ├── data_dictionary.md
    ├── experiment_protocol.md
    └── production_runbook.md
```

---

# 20. Recommended Libraries

## Core data and validation

- `numpy`
- `pandas` or `polars`
- `pyarrow`
- `pandera`
- `pydantic`

## Classical and statistical models

- `scikit-learn`
- `statsmodels`
- `statsforecast`
- `arch`
- `pymc`
- `arviz`

## Tree models

- `catboost`
- `xgboost`
- `lightgbm`
- `interpret`

## Pretrained and deep learning

- `torch`
- `transformers`
- `huggingface_hub`
- `safetensors`
- `chronos-forecasting`
- official TiRex-2 package, pinned to the validated release
- `timesfm`
- `momentfm`
- `uni2ts` and `gluonts` for Moirai experiments
- `granite-tsfm` for IBM experiments
- `peft` and `accelerate` for later LoRA experiments

## Calibration, conformal, and online learning

- `betacal`
- `mapie`, with time-series adaptation rather than naive IID usage
- `venn-abers`
- `river`

## Optimization and reproducibility

- `optuna`
- `mlflow`
- `dvc`
- `hydra-core`
- `omegaconf`
- `joblib`

## Explainability and testing

- `shap`
- `pytest`
- `hypothesis`
- `ruff`
- `mypy`
- `pre-commit`
- `uv` or `conda-lock`

---

# 21. Pseudocode

## 21.1 Base walk-forward loop

```python
for origin in forecast_origins:
    train_feature_end = origin - 1
    train_target_end = origin - 1

    train_values = values.loc[:train_feature_end]
    current_values = values.loc[:origin]

    transformer = fit_transform_rules(train_values)
    imputer = fit_imputer(train_values)
    scaler = fit_robust_scaler(train_values)
    pca = fit_pca(train_values, n_factors=selected_n_factors)

    X_train = build_long_features(
        values=train_values,
        transformer=transformer,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
    )

    X_current = build_features_for_date(
        values=current_values,
        date=origin,
        transformer=transformer,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
    )

    y_train = targets.loc[X_train.date <= train_target_end]

    for model in base_models:
        fitted = model.fit(X_train, y_train, groups=X_train.date)
        probability = fitted.predict_proba(X_current)
        save_oof_prediction(origin, model.name, probability)
```

## 21.2 Pretrained probability conversion

```python
def probability_of_up(current_value, forecast_distribution):
    if forecast_distribution.samples is not None:
        return mean(sample > current_value for sample in forecast_distribution.samples)

    if forecast_distribution.quantiles is not None:
        cdf_at_current = interpolate_monotone_cdf(
            forecast_distribution.quantile_values,
            forecast_distribution.quantile_levels,
            current_value,
        )
        return 1.0 - cdf_at_current

    raise ValueError("A point-only forecast is insufficient for direct confidence.")
```

## 21.3 Cross-fitted ensemble

```python
for meta_origin in meta_origins:
    historical_base_oof = base_oof[base_oof.origin < meta_origin]
    current_base_oof = base_oof[base_oof.origin == meta_origin]

    stacker = fit_constrained_convex_weights(
        historical_base_oof,
        nonnegative=True,
        sum_to_one=True,
        shrink_toward="global_logistic",
    )

    raw_ensemble_probability = stacker.predict(current_base_oof)
    save_meta_oof(meta_origin, raw_ensemble_probability)
```

## 21.4 Calibration and correctness model

```python
for reliability_origin in reliability_origins:
    historical_meta = meta_oof[meta_oof.origin < reliability_origin]
    current_meta = meta_oof[meta_oof.origin == reliability_origin]

    calibrator = fit_beta_or_platt_calibration(historical_meta)
    calibrated_probability = calibrator.predict(current_meta.raw_probability)

    historical_correctness = build_correctness_labels(historical_meta)
    reliability_model = fit_regularized_correctness_model(historical_correctness)

    expected_correctness = reliability_model.predict_proba(
        build_reliability_features(current_meta, calibrated_probability)
    )

    save_reliability_oof(reliability_origin, expected_correctness)
```

## 21.5 Selection

```python
def select_month(candidates, threshold, max_predictions=20):
    eligible = candidates[
        candidates.data_quality_ok
        & candidates.in_historical_support
        & candidates.model_execution_ok
        & (candidates.correctness_lcb >= threshold)
    ]

    eligible = eligible.sort_values(
        by=["correctness_lcb", "expected_correctness"],
        ascending=False,
    )

    accepted = eligible.head(max_predictions)
    rejected = candidates.drop(index=accepted.index)

    return accepted, rejected
```

---

# 22. Monthly Inference Pipeline

1. Ingest the latest Bloomberg snapshot.
2. Validate schema and dates.
3. Apply publication-lag and vintage rules.
4. Detect missing, stale, and out-of-range inputs.
5. Refit or update permitted base models using history only.
6. Run retained pretrained models.
7. Retrieve cached pretrained outputs where reproducible and valid.
8. Build base-model probability vector.
9. Apply the locked ensemble specification.
10. Calibrate probability.
11. Build reliability features.
12. Predict expected correctness.
13. Calculate bootstrap lower confidence bound.
14. Select accepted indicators.
15. Produce reasons and uncertainty fields.
16. Write an immutable forecast ledger before outcomes are known.
17. After the next month arrives, score predictions and update monitoring.

---

# 23. Monitoring and Retraining

Monitor monthly:

- Accepted prediction count.
- Accepted accuracy.
- Rolling 12- and 24-month selected accuracy.
- Brier score and log loss.
- Calibration slope and intercept.
- Model disagreement.
- Feature drift.
- Factor/regime distance.
- Data revision rate.
- Missingness and staleness.
- Performance by indicator and direction.
- Rejection-rate changes.

Retraining policy:

- Update simple base models monthly.
- Recalculate permitted fold-local features monthly.
- Reconsider hyperparameters on a fixed annual schedule.
- Do not change architecture after one bad month.
- Trigger a research review if calibration, selected accuracy, or coverage deteriorates for a sustained period.

---

# 24. Final Target Architecture

The target production architecture, subject to experiment results, is:

```text
Branch 1: Global Elastic-Net Logistic
    Inputs: compact own features, PCA/dynamic factors, breadth, Indicator ID

Branch 2: Shallow Global CatBoost
    Inputs: same compact local/shared features, regime and missingness features

Branch 3: Statistical expert
    Dynamic factor or state-space probability and uncertainty

Branch 4: Chronos-2 zero-shot
    P(Up), median forecast, quantile width, asymmetry

Branch 5: TiRex-2 zero-shot multivariate expert
    Joint quantiles -> P(Up), interval width, and asymmetry

Branch 6: Optional frozen representation expert
    Chronos-2 or MOMENT embedding -> Elastic-Net Logistic head -> P(Up)

Branch 7: Simple momentum/persistence baseline
    Retained only when it adds stable diversity

All branch probabilities
    -> constrained convex ensemble
    -> beta or Platt calibration
    -> correctness/reliability model
    -> date-block bootstrap correctness LCB
    -> minimum reliability threshold
    -> accept up to 20 indicators
```

Potential later upgrades:

- Chronos-2 or MOMENT embedding head.
- TSPulse lightweight representation experiment.
- Chronos-2 LoRA.
- TimesFM LoRA challenger.
- Multi-window dynamic weights.
- Hierarchical dynamic Logistic model.

---

# 25. What Must Not Be Attempted Initially

- Full fine-tuning of a foundation model.
- Large neural network trained from scratch.
- Forty independent boosted-tree models.
- Unrestricted 40-variable VAR.
- End-to-end GNN.
- All-pairs Granger or transfer-entropy feature generation.
- Thousands of automated technical indicators.
- Per-indicator isotonic calibration.
- Random train-test split.
- Full-dataset PCA or scaling.
- Raw top-20 probability selection.
- Mandatory minimum of 15 accepted predictions.
- Repeated redesign after seeing the locked audit.
- Claims of 65% accuracy without date-block uncertainty and matched-coverage baselines.

---

# 26. Final Success Definition

The project is successful only if the complete pipeline demonstrates, on the locked audit:

1. Better selected accuracy than naive and classical baselines at matched coverage.
2. Acceptable probability calibration.
3. Stable performance across time and indicators.
4. A useful risk-coverage tradeoff.
5. Honest abstention in weak-signal months.
6. No detected leakage.
7. Reproducibility from fixed configurations and data versions.
8. A defensible lower confidence bound, not only a favorable average.

The final answer is not predetermined to be Logistic Regression, CatBoost, or a pretrained model. The research process is designed so that each candidate must demonstrate incremental out-of-sample value. The most likely robust endpoint is a calibrated selective ensemble combining a regularized global panel model, a shallow nonlinear model, a low-dimensional statistical expert, and one or two validated pretrained components.

---

# 27. Research Basis and Current Model Verification

This plan incorporates the original project requirements and the completed model research. At the 2026-07-23 research snapshot:

- The official Amazon Chronos repository describes Chronos-2 as the current model with zero-shot univariate, multivariate, and covariate-informed forecasting; current releases also expose encoder embeddings.
- The TiRex-2 release is treated here as a newer multivariate zero-shot forecasting model. Its current open workflow is inference-first; embedding and fine-tuning interfaces must not be assumed without release-specific verification.
- The official TimesFM repository provides pretrained checkpoints and current forecasting support, with adapter/LoRA-related code paths available in the project history.
- The official MOMENT project provides open time-series foundation models and code for general-purpose forecasting, classification, and representation experiments.

Before production deployment, re-check the exact checkpoint license, version, package API, and maintenance state for every pretrained component. Pin all dependencies and model revisions.

---


# 27.1 Stage Exit Checklist

A phase is complete only when its artifact and gate are recorded:

| Stage | Required artifact | Exit condition |
|---|---|---|
| Data validity | Audit report, availability map, future-perturbation test | No unresolved leakage or as-of ambiguity for production inputs |
| Baselines | Immutable OOF prediction files | Baseline metrics and block-bootstrap intervals are reproducible |
| Global models | OOF probabilities for Elastic-Net and CatBoost | At least one model improves a proper score without unstable rolling behavior |
| Statistical expert | Fold-local factor/state-space outputs | Adds stable OOF value or useful uncertainty |
| Pretrained zero-shot | Cached Chronos-2 and TiRex-2 forecasts at every origin | Incremental forecast or reliability value is demonstrated |
| Representation probe | Cross-fitted Chronos-2/MOMENT/TSPulse embeddings | Run only after zero-shot stage; retain only stable incremental gain |
| Ensemble | Level-B OOF predictions and weight history | Beats best component and equal-weight average |
| Calibration/reliability | Level-C OOF decisions | Improves Brier/calibration and risk-coverage performance |
| Locked audit | Signed configuration and immutable ledger | One-time audit completed without redesign |
| Production | Model card, runbook, monitoring dashboard | Monthly inference is reproducible and point-in-time safe |

# 28. Immediate Next Actions

Execute these in order:

1. Create the repository and configuration skeleton.
2. Load the real 300 x 40 dataset.
3. Produce the data audit and availability report.
4. Implement target generation and leakage tests.
5. Implement compact fold-local features.
6. Run naive/statistical baselines.
7. Train the Global Elastic-Net Logistic model.
8. Train the Shallow Global CatBoost model.
9. Compare PCA and Dynamic Factor features.
10. Generate Chronos-2 zero-shot walk-forward forecasts.
11. Generate TiRex-2 zero-shot walk-forward forecasts.
12. Test Chronos-2/MOMENT frozen representation heads only if the zero-shot stage justifies it.
13. Build cross-fitted ensemble predictions.
14. Calibrate probabilities.
15. Train the correctness/reliability model.
16. Tune the selection threshold in development only.
17. Freeze the full methodology.
18. Run the final 48-month locked audit once.
19. Select the production ensemble based on the complete evidence.


---

# 29. Master Candidate and Combination Matrix

This appendix is the single consolidated table for all methods discussed during the research. It separates each option by its role in the pipeline, what it can combine with, what it replaces, and whether it belongs in the initial system.

**Priority key:**

- **P0:** mandatory foundation.
- **P1:** primary experiment.
- **P2:** secondary experiment after the foundation is stable.
- **P3:** later or conditional experiment.
- **P4:** do not attempt initially.

| Group | Method / model | Role in the project | Can combine with | Alternative to / do not duplicate initially | Main benefit | Main risk | Priority |
|---|---|---|---|---|---|---|---|
| Baseline | Historical majority class | Simple directional probability baseline | Final benchmark and optional ensemble component | None | Establishes whether the problem beats class frequency | Ignores current market state | P0 |
| Baseline | Last-direction continuation / persistence | Predict next direction from the previous direction | Benchmark and optional ensemble component | None | Detects simple momentum | Fails during reversals | P0 |
| Baseline | Last-direction reversal | Predict the opposite of the previous direction | Benchmark and optional ensemble component | None | Detects simple mean reversion | Too simple for changing regimes | P0 |
| Baseline | Momentum rules | Direction from 1-, 3-, 6-, or 12-month momentum | Logistic, CatBoost, and baseline ensemble | Different windows are alternatives to be validated | Transparent and cheap | Window-selection data snooping | P0 |
| Baseline | Moving-average / mean-reversion rules | Direction from distance to rolling level | Logistic, CatBoost, and baseline ensemble | Different windows are alternatives | Useful for cyclical or reverting series | Poor during persistent trends | P0 |
| Statistical | AR / autoregression | Forecast next change or value, then convert to direction | Ensemble and reliability model | ARIMA / ETS are challengers | Strong low-complexity time-series reference | Mostly univariate and linear | P1 |
| Statistical | ARIMA / SARIMA | Forecast level or change | Ensemble | AR / ETS / state-space | Standard forecasting benchmark | Parameter instability with short data | P1/P2 |
| Statistical | ETS / exponential smoothing / Theta | Forecast level and trend | Ensemble | ARIMA / local trend | Robust and low variance | Limited cross-indicator information | P1 |
| Statistical | Local-level / local-trend state-space | Forecast latent level and trend with uncertainty | Ensemble and reliability features | ETS / Kalman variants | Natural uncertainty estimates | Limited nonlinear behavior | P1 |
| Statistical | Kalman / general state-space model | Estimate evolving latent state and forecast distribution | Ensemble and regime features | Dynamic factor model for shared structure | Handles noise and gradual drift | Specification sensitivity | P1/P2 |
| Statistical | Dynamic Factor Model | Shared latent market factors or standalone forecast expert | Logistic, CatBoost, ensemble | PCA is the simpler first representation | Models cross-indicator dependence compactly | Mostly linear | P1 |
| Statistical | Full VAR | Joint forecast of all 40 indicators | Theoretically ensembleable | Bayesian or factor VAR | Direct cross-series dynamics | Extreme parameter count | P4 |
| Statistical | Bayesian VAR / Minnesota VAR | Shrunk joint forecast | Ensemble | Full VAR / factor-augmented VAR | Controls some VAR overfitting | Still heavy for 40 series and 300 dates | P2 |
| Statistical | Factor-augmented VAR | VAR over a few factors | Ensemble | Dynamic factor model / Bayesian VAR | Cross-series dynamics with compression | Window and lag instability | P2 |
| Statistical | HMM | Filtered regime probabilities as features | Logistic, CatBoost, reliability model | Markov-switching regression | Adds regime context | Unstable state identification | P2 |
| Statistical | Markov-switching regression | Regime-specific forecast model | Ensemble | HMM feature approach | Allows relationships to change by regime | High variance if too many states | P2/P3 |
| Statistical | Gaussian Process | Probabilistic nonlinear univariate model | Ensemble | State-space / kernel alternatives | Clear uncertainty framework | Kernel instability and cost | P3 |
| Statistical | Historical analog / nearest-neighbor states | Empirical forecast and similarity-based reliability | Reliability model and ensemble | None | Interpretable historical support | Few close analogs in some months | P2 |
| Classical ML | Global Elastic-Net Logistic Regression | Primary shared panel classifier | Factors, engineered features, pretrained forecast features, ensemble | Ridge, Lasso, mixed-model variants | Best bias-variance tradeoff and direct probability | Mostly linear | P0 |
| Classical ML | Ridge Logistic Regression | Stable linear benchmark | Same features as Elastic Net | Elastic Net / Lasso | Very stable with correlated features | Keeps irrelevant features | P1 |
| Classical ML | Lasso Logistic Regression | Sparse linear benchmark | Same features as Elastic Net | Elastic Net / Ridge | Automatic feature selection | Selection instability | P2 |
| Classical ML | Per-indicator Logistic Regression | Optional indicator-specific expert | Shared factors and global-model shrinkage | Forty separate tree models | Captures genuine heterogeneity | Fewer than 300 labels per model | P1/P2 |
| Classical ML | Hierarchical Bayesian Logistic Regression | Partial-pooling alternative architecture | Shared factors and indicator effects | Global Elastic Net / GLMM | Principled pooling and uncertainty | Slower and prior-sensitive | P2 |
| Classical ML | Generalized Linear Mixed Model | Shared slopes with random effects | Same features as global Logistic | Hierarchical Bayesian model | Interpretable heterogeneity | Less flexible than Bayesian version | P2 |
| Classical ML | Low-rank multi-task Logistic | Shared plus low-rank indicator coefficients | Factors and embeddings | Forty independent classifiers | Flexible partial pooling | Requires careful rank regularization | P2 |
| Classical ML | Linear SVM plus calibration | Linear classification challenger | Engineered or frozen embedding features | Logistic head | Strong high-dimensional margin model | No native probabilities | P2 |
| Tree ML | Shallow Global CatBoost | Primary nonlinear panel model | Same features as Logistic and ensemble | XGBoost / LightGBM as tree-family challengers | Handles nonlinearities and missingness | Overfitting if deep | P1 |
| Tree ML | Global XGBoost | Nonlinear challenger | Logistic, factors, pretrained forecast features | CatBoost / LightGBM | Powerful regularized boosting | More tuning sensitivity | P2 |
| Tree ML | Global LightGBM | Fast nonlinear challenger | Logistic, factors, pretrained forecast features | CatBoost / XGBoost | Efficient experimentation | Leaf-wise overfitting | P2 |
| Tree ML | HistGradientBoosting | Lightweight boosting benchmark | Engineered features | Other boosting models | Simple scikit-learn integration | Less flexible | P2 |
| Tree ML | Explainable Boosting Machine | Interpretable nonlinear benchmark | Engineered features and factors | Tree boosting for nonlinear explanation | Strong interpretability | Limited high-order interactions | P2 |
| Tree ML | Random Forest | Benchmark only | Optional ensemble if genuinely diverse | Extra Trees and boosted trees | Robust simple tree baseline | Poor probability stability with small data | P3 |
| Tree ML | Extra Trees | Randomized tree benchmark | Optional ensemble if diverse | Random Forest | Strong diversity | Can fit noise | P3 |
| Tree ML | Forty separate CatBoost/XGBoost/RF models | Fully indicator-specific models | Technically ensembleable | Per-indicator Logistic | Maximum local flexibility | Severe overfitting | P4 |
| Representation | PCA | First fold-local shared representation | Logistic and CatBoost | Dynamic factors as next challenger | Simple robust dimension reduction | Unsupervised factors | P0/P1 |
| Representation | Sparse PCA | Sparse latent factors | Logistic and CatBoost | PCA | More interpretable factor loadings | Less stable | P2 |
| Representation | ICA | Independent latent components | Logistic and CatBoost | PCA / factor models | Alternative non-Gaussian representation | Component instability | P2/P3 |
| Representation | PLS | Supervised low-dimensional representation | Logistic or linear head | PCA | Uses target-related directions | Leakage risk if not nested | P2 |
| Representation | Kernel PCA | Nonlinear latent representation | Logistic / CatBoost | Autoencoder / PCA | Nonlinear compression | Kernel and sample-size instability | P3 |
| Representation | Autoencoder | Learned nonlinear latent factors | Classifier or ensemble | PCA / pretrained embeddings | Flexible representation | Too few time points | P3 |
| Representation | Variational Autoencoder | Probabilistic nonlinear latent representation | Classifier | Autoencoder | Uncertainty-aware latent space | Even higher overfitting risk | P4 |
| Representation | NMF | Nonnegative factors for eligible data | Logistic / CatBoost | PCA | Interpretable parts | Requires nonnegative-compatible data | P3 |
| Cross-indicator | Pearson / Spearman correlation | Exploratory clustering and feature screening | Factors, clusters, reliability features | None | Simple dependence map | Multiple-testing false discoveries | P1/P2 |
| Cross-indicator | Rolling correlation | Regime-varying dependence features | Logistic, CatBoost, reliability model | Static correlation | Captures changing relationships | Noisy in short windows | P2 |
| Cross-indicator | Partial correlation / Graphical Lasso | Sparse conditional-dependence network | Feature discovery and clustering | Full graph learning | Reduces redundant edges | Regularization sensitivity | P2 |
| Cross-indicator | Lagged correlation / lead-lag network | Small stable leader features | Logistic and CatBoost | All-pairs unrestricted lags | Potential predictive cross-series signals | Very large search space | P2 |
| Cross-indicator | Granger causality | Exploratory directed relationships | Stable selected features only | Lead-lag correlations | Structured temporal test | Low power and false discoveries | P3 |
| Cross-indicator | Cointegration | Long-run spread features within coherent groups | Statistical experts and CatBoost | Generic pairwise spreads | Potential mean-reverting residuals | Anonymous indicators and structural breaks | P3 |
| Cross-indicator | Mutual information | Nonlinear dependency screening | Feature discovery | Correlation methods | Captures nonlinear dependence | Unstable with 300 dates | P3/P4 |
| Cross-indicator | Transfer entropy | Nonlinear directed dependency | Research only | Granger / lag correlation | Flexible directional relation | Extremely data hungry | P4 |
| Cross-indicator | Generic causal discovery | Hypothesis generation only | None in initial production | Graphical Lasso / lead-lag screening | May suggest structural hypotheses | Hidden confounding and nonstationarity | P4 |
| Regime / drift | Change-point detection | Reliability and recency feature | Logistic, CatBoost, reliability model | HMM regimes | Detects structural breaks | False alarms | P1/P2 |
| Regime / drift | Multi-window experts | Separate expanding, 60-, 120-, and 180-month models | Ensemble | Single-window training | Adapts to changing regimes | More experiments and correlation | P1 |
| Regime / drift | Exponential recency weighting | Weight recent training dates more heavily | Logistic and tree models | Hard rolling window | Smooth adaptation | Half-life tuning risk | P1 |
| Fixed representation | ROCKET / MiniRocket | Fixed convolutional features plus linear classifier | Ensemble | From-scratch TCN | Low trainable parameter count | Large feature space needs compression | P2 |
| Fixed representation | Echo State Network / reservoir computing | Fixed recurrent representation plus linear output | Ensemble | LSTM from scratch | Cheap nonlinear sequence features | Reservoir hyperparameter sensitivity | P2 |
| Deep from scratch | LSTM / GRU / BiLSTM | Direct sequence classifier or forecaster | Multi-task heads and factors | Pretrained sequence representations | Learns temporal dependencies | Insufficient effective sample size | P4 |
| Deep from scratch | Temporal Convolutional Network | Causal sequence model | Multi-task output | ROCKET / pretrained model | Efficient local temporal patterns | Overfitting risk | P3 |
| Deep from scratch | N-BEATS / N-HiTS | Neural value forecasting | Ensemble | Statistical forecasting and TSFM | Strong generic forecasting design | Too little local data | P3 |
| Deep from scratch | DeepAR | Global probabilistic autoregression | Ensemble | TSFM probabilistic expert | Shared probabilistic model | Data requirement too high | P3 |
| Deep from scratch | Temporal Fusion Transformer | Multi-horizon covariate model | Multi-output setup | Simpler global models | Flexible covariates and gating | Excessive capacity | P4 |
| Deep from scratch | PatchTST / iTransformer / TimesNet / TimeMixer | Modern multivariate sequence model | Multi-task setup | Pretrained TSFM | Powerful temporal representation | 300 dates are inadequate | P4 |
| Deep from scratch | Informer / Autoformer / FEDformer | Long-horizon forecasting | None initially | Other transformers | Efficient long-sequence forecasting | Wrong horizon and sample regime | P4 |
| Deep from scratch | Mamba / neural state-space sequence model | Efficient deep sequence model | Multi-output setup | Pretrained state-space model | Modern efficient architecture | Too little data from scratch | P4 |
| Deep from scratch | Shared backbone plus 40 heads | Multi-task classifier | Factors and embeddings | Shared head with indicator bias | Indicator specialization | Forty heads can memorize noise | P3/P4 |
| Graph | GNN / GCN / GAT | Cross-indicator graph forecasting | Temporal encoder | Factor and sparse-graph features | Explicit relational modeling | Graph and weights both estimated from 300 dates | P4 |
| Graph | Temporal GNN / dynamic graph | Time-varying graph and sequence model | Multi-output prediction | Rolling correlations plus panel model | Rich dynamic dependence | Excess complexity and leakage risk | P4 |
| Pretrained TSFM | Chronos-2 | Primary zero-shot probabilistic expert; optional embeddings and later LoRA | Ensemble, local models through forecast features | TimesFM / TiRex-2 / Moirai as challengers | Best overall fit: probabilistic, multivariate, covariates, embeddings | Domain mismatch and calibration | P1 |
| Pretrained TSFM | Chronos-Bolt | Fast zero-shot quantile benchmark | Ensemble | Chronos-2 | Efficient inference | Less complete than Chronos-2 | P2 |
| Pretrained TSFM | Legacy TiRex representation path | Deferred representation experiment only after release-specific verification | Optional later ensemble | Chronos-2 / MOMENT / TSPulse embeddings | Possible recurrent representation diversity | Not supported by the final research recommendation as an initial path | P3 |
| Pretrained TSFM | TiRex-2 | Multivariate zero-shot probabilistic expert | Ensemble and reliability features | Chronos-2 / TimesFM multivariate forecast | Joint variate modeling and quantiles | Newer ecosystem | P1/P2 |
| Pretrained TSFM | TimesFM 2.5 | Zero-shot and later LoRA challenger | Ensemble | Chronos-2 / Moirai | Strong official model and PEFT path | Less direct joint-panel fit | P2 |
| Pretrained representation | MOMENT | Frozen embeddings plus linear classifier | Ensemble | Chronos-2 / TSPulse embeddings | Designed for representation and classification tasks | May not encode financial direction | P2 |
| Pretrained TSFM | Moirai 2 / Uni2TS | Probabilistic multivariate challenger | Ensemble | Chronos-2 / TiRex-2 | Flexible universal forecasting | Check weight license for production | P2/P3 |
| Pretrained TSFM | IBM Granite FlowState | Lightweight probabilistic expert | Ensemble | Other small TSFMs | Low local compute | Weaker joint cross-series support | P2/P3 |
| Pretrained TSFM | IBM Tiny Time Mixers | Small adaptation benchmark | Ensemble or fine-tuned head | FlowState / other small models | Lightweight and adaptable | Monthly-frequency checkpoint fit must be verified | P3 |
| Pretrained representation | IBM TSPulse / PatchTST-FM | Frozen feature extractor | Linear head and ensemble | MOMENT / Chronos-2 embeddings | Small representation model | Not the leading direct forecaster | P2/P3 |
| Pretrained tabular | TabPFN-TS | Zero-shot tabularized forecasting benchmark | Ensemble | Classical lag model | Strong small-data prior | License and multivariate limitations | P2 |
| Pretrained financial | FinCast | Financial-domain probabilistic research expert | Ensemble | General TSFMs | Potential domain advantage | Immature tooling and uncertain domain match | P3 |
| Pretrained TSFM | Toto 2.0 | Multivariate quantile expert | Ensemble | TiRex-2 / Chronos-2 | Variate-aware architecture | Training domain closer to telemetry | P3 |
| Pretrained TSFM | Sundial | Sample-based probabilistic expert | Ensemble and uncertainty features | Other generative TSFMs | Full forecast samples | Inference cost and mainly univariate use | P3 |
| Pretrained TSFM | Lag-Llama | Probabilistic autoregressive benchmark | Ensemble | Newer TSFMs | Open probabilistic output | Older and less suitable than leading options | P3 |
| Pretrained TSFM | Time-MoE | Small mixture-of-experts benchmark | Ensemble | Timer / large TSFMs | Flexible capacity | Larger models unnecessary | P3 |
| Pretrained TSFM | Timer / Timer-XL | Long-context benchmark | Ensemble | Time-MoE / other decoders | Strong long-context modeling | Context is not the bottleneck | P3 |
| Pretrained TSFM | TempoPFN | Synthetic-pretrained zero-shot benchmark | Ensemble | TabPFN-TS | Lower real-data contamination risk | Limited finance evidence | P3 |
| Pretrained TSFM | Reverso | Small modern state-space/convolution model | Research benchmark | Granite / other small models | Lightweight architecture | Very new and immature | P3/P4 |
| Commercial TSFM | TimeGPT | External commercial benchmark | Ensemble only if governance permits | Local open-source TSFMs | Easy managed probabilistic forecasts | Closed weights, privacy, and cost | P3 |
| Financial model | Kronos | OHLCV/candlestick model | Only if the dataset is converted to a compatible market format | FinCast / generic TSFM | Financial market specialization | Structural mismatch to anonymous monthly scalar indicators | P4 |
| Pretrained usage | Zero-shot point forecast | Model produces next value; convert to direction | Ensemble | Probabilistic zero-shot | No local training | Weak confidence information | P2 |
| Pretrained usage | Zero-shot probabilistic / quantile forecast | Convert quantiles or samples into probability of Up | Ensemble and reliability model | Point-only forecast | Best initial pretrained strategy | Requires local calibration | P1 |
| Pretrained usage | Forecast outputs as local features | Add forecast median, probability, widths, asymmetry | Logistic or CatBoost | Same model as independent expert must be ablated | Combines global prior and local supervision | Duplicate signal if used carelessly | P1 |
| Pretrained usage | Pretrained uncertainty as a feature | Add interval width, sample dispersion, or entropy | Reliability model | Raw uncertainty alone | Useful even when point accuracy is neutral | Uncertainty may be miscalibrated | P1 |
| Pretrained usage | Frozen embeddings plus Elastic-Net head | Freeze backbone and train a small shared classifier | Ensemble | Small MLP / CatBoost head | Lowest-risk representation adaptation | High-dimensional embeddings | P1 |
| Pretrained usage | Frozen embeddings plus small MLP | Train a 16-32-unit nonlinear head | Ensemble | Linear head | Captures mild nonlinearity | Higher variance | P2 |
| Pretrained usage | Frozen embeddings plus CatBoost | Fold-local compression then shallow CatBoost | Ensemble | Linear / MLP head | Nonlinear embedding interactions | Easy to overfit | P2 |
| Pretrained usage | Train final classification head only | Freeze backbone and replace task head | Ensemble | External linear probe | Direct task adaptation | Depends on model interface | P1/P2 |
| Pretrained usage | LoRA | Low-rank updates to selected backbone layers | Ensemble | Full fine-tuning / adapters | Stronger adaptation with fewer trainable weights | Still high variance at 300 dates | P3 |
| Pretrained usage | Adapter / feature-space adapter | Small trainable projection before or after backbone | Ensemble | LoRA / full tuning | Adaptation without changing backbone | More validation complexity | P2/P3 |
| Pretrained usage | Partial unfreezing | Train the last block or two | Ensemble | LoRA | More domain specialization | High overfitting risk | P3/P4 |
| Pretrained usage | Full fine-tuning | Update the whole backbone | Ensemble | LoRA / frozen head | Maximum flexibility | Not justified with current data | P4 |
| Pretrained usage | Distillation | Train a smaller student from a proven pretrained ensemble | Production optimization | Direct deployment of teacher | Faster inference | Premature before teacher proves value | P3 |
| Ensemble | Soft voting / simple mean | Average calibrated model probabilities | All accepted base models | Weighted stacking | Stable simple ensemble baseline | Includes weak models equally | P1 |
| Ensemble | Weighted averaging | Learn or set model weights | Diverse base models | Soft voting / stacking | Better than equal average if stable | Weight overfitting | P1/P2 |
| Ensemble | Bayesian model averaging | Posterior-style model weighting | Statistical and classical experts | Convex stack | Accounts for model uncertainty | Assumption and implementation complexity | P2 |
| Ensemble | Constrained convex stacking | Nonnegative weights summing to one from OOF predictions | Final base-model set | Unconstrained stacking | Primary final ensemble method | Requires multi-level cross-fitting | P1 |
| Ensemble | Dynamic / regime-dependent selection | Change weights by regime | Fixed ensemble | Online weighting | Adapts to structural change | Regime overfitting | P2 |
| Ensemble | Online Hedge / dynamic model averaging | Update weights prequentially | Multi-window and model experts | Static stacking | Adaptive without large retuning | Learning-rate sensitivity | P2 |
| Ensemble | Per-indicator model selection | Use different experts by indicator | Global ensemble | Forty full local models | Captures stable heterogeneous winners | Selection bias | P2/P3 |
| Ensemble | Agreement filter | Use model agreement as a reliability feature | Reliability model | Hard agreement rule | Simple diversity signal | Agreement does not guarantee correctness | P1 as feature |
| Calibration | Platt scaling | Logistic probability calibration | Ensemble output | Beta / isotonic | Stable and simple | Limited shape flexibility | P1 |
| Calibration | Beta calibration | Flexible binary calibration | Ensemble output | Platt / isotonic | Preferred initial calibration | Needs enough OOF history | P1 |
| Calibration | Isotonic regression | Nonparametric calibration | Ensemble output | Beta / Platt | Highly flexible | Overfits small samples | P3 |
| Calibration | Venn-Abers | Probability interval calibration benchmark | Ensemble output | Standard calibration | Adds interval-style information | More complex operations | P2 |
| Selective prediction | Correctness / reliability meta-model | Predict whether the directional forecast will be correct | Ensemble probability, disagreement, uncertainty, regime, quality | Raw confidence threshold | Core selection mechanism | Must be fully cross-fitted | P0/P1 |
| Selective prediction | Adaptive conformal classification | Require singleton Up or Down sets | Reliability gate | Ordinary iid conformal | Formalized abstention guardrail | Time-series guarantees are weaker | P2 |
| Selective prediction | Date-block bootstrap LCB | Conservative lower bound on expected correctness | Reliability ranking | Raw expected correctness | Accounts for temporal uncertainty | Computationally heavier | P1 |
| Selective prediction | Fixed confidence threshold | Accept predictions above one threshold | Calibrated probabilities | Dynamic / per-indicator threshold | Simple | May ignore heterogeneity | P1 benchmark |
| Selective prediction | Per-indicator threshold | Different threshold per indicator | Calibrated reliability | Global threshold | Captures stable indicator differences | Too little history per indicator | P2/P3 |
| Selective prediction | Dynamic monthly threshold | Threshold responds to the opportunity set | Reliability scores | Fixed threshold | Adapts monthly coverage | Selection overfitting | P2 |
| Selective prediction | Top-K only | Always take highest K | None | Thresholded Top-K | Stable coverage | Forces weak predictions | P4 as final rule |
| Selective prediction | Top-K plus reliability floor | Accept at most 20 that exceed the threshold | Reliability model and LCB | Top-K only | Recommended final policy | May return fewer than 15 | P0/P1 |
| Selective prediction | Abstention / reject option | Reject weak, unsupported, stale, or shifted cases | Entire pipeline | Forced coverage | Protects selected accuracy | Lower coverage | P0 |
| Target | Direct Up/Down classification | Primary supervised target | Global and local classifiers | Numerical-only forecasting | Aligns directly with decision | Ignores move size | P0 |
| Target | Numerical forecast then direction | Secondary route for forecasting experts | Ensemble | Direct classification | Uses classical and TSFM forecasting tools | Small value error can flip direction | P1 |
| Target | Return distribution / quantile forecast | Estimate probability of positive change | Pretrained and statistical experts | Point forecast | Natural uncertainty | Needs robust probability conversion | P1 |
| Target | Joint classification plus regression/quantile loss | Multi-task objective | Neural or adapted pretrained model | Single-task heads | Shared information from size and sign | Higher variance | P2/P3 |
| Target | Dead-zone auxiliary target | Mark tiny moves as uncertain for reliability modeling | Reliability model | Official binary target | Separates noise from meaningful changes | Delta selection | P2 |
| Validation | Point-in-time / vintage-correct data | Ensure inputs existed at each origin | All models | Revised-data pseudo-backtest | Prevents the largest leakage source | Metadata may be unavailable | P0 |
| Validation | Expanding walk-forward | Main real-time simulation | All models | Rolling windows | Maximum history | Old regimes may dominate | P0 |
| Validation | Rolling 60/120/180-month windows | Recency challengers | Multi-window ensemble | Expanding only | Regime adaptation | Smaller training sets | P1 |
| Validation | Nested time-series CV | Tune every stage without future information | All trainable stages | Flat tuning | Honest model selection | High compute cost | P0 |
| Validation | Date grouping | Keep all 40 indicators from a month together | All folds and bootstraps | Row-wise splitting | Respects dependence | Reduces effective sample size | P0 |
| Validation | Purging / embargo | Prevent release-lag and label overlap leakage | Relevant folds | No embargo | Safer temporal separation | Less training data | P1 when needed |
| Validation | Locked final 48-month audit | Untouched final evaluation | Entire fixed pipeline | Repeated test tuning | Credible final estimate | Less development data | P0 |
| Final architecture | Hybrid selective panel ensemble | Classical panel + statistical factors + selected pretrained experts + calibration + reliability + abstention | All promoted components | Any single-model solution | Highest expected robustness and diversity | Engineering complexity | Final target |

---

# 30. Complete Combination Map

## 30.1 Components that are inputs, not competing models

The following are feature or representation layers and can be used inside Logistic Regression or CatBoost:

- Lags, momentum, volatility, moving-average distance, robust z-scores, ranks, breadth, and missingness features.
- PCA or Dynamic Factor features.
- Filtered HMM or change-point probabilities.
- Stable lead-lag features.
- Chronos-2 forecast median, probability of Up, interval width, and asymmetry.
- Frozen Chronos-2, MOMENT, or TSPulse embeddings after fold-local compression where needed. A legacy TiRex representation path is deferred pending separate verification.

## 30.2 Models that can produce independent probabilities for the ensemble

- Global Elastic-Net Logistic Regression.
- Shallow Global CatBoost.
- Momentum / persistence baseline.
- Dynamic Factor or state-space forecast expert.
- Optional shrunk per-indicator Logistic expert.
- Chronos-2 zero-shot probabilistic expert.
- TiRex-2 zero-shot multivariate expert.
- Optional Chronos-2 or MOMENT frozen embedding head.
- Optional MOMENT or Chronos-2 embedding head if it adds unique value.

## 30.3 Alternative families where one primary option is chosen first

| Family | First option | Challengers tested separately |
|---|---|---|
| Linear panel classifier | Global Elastic-Net Logistic | Ridge, Lasso, GLMM, hierarchical Bayesian Logistic |
| Tree model | Shallow Global CatBoost | XGBoost, LightGBM, EBM |
| Randomized tree benchmark | Extra Trees or Random Forest | Keep at most one unless both add unique OOF diversity |
| Shared factor representation | PCA | Dynamic Factor Model, Sparse PCA, PLS |
| Statistical univariate expert | Local trend / ETS or AR | ARIMA, Kalman, Gaussian Process |
| Pretrained probabilistic forecast | Chronos-2 | TiRex-2, TimesFM, Moirai, FlowState |
| Pretrained representation | Chronos-2 embeddings | MOMENT, TSPulse |
| Embedding head | Elastic-Net Logistic head | Small MLP, compressed-embedding CatBoost |
| Pretrained adaptation | Frozen backbone | Adapter, LoRA, partial unfreezing, full fine-tuning |
| Calibration | Beta calibration | Platt, Venn-Abers, isotonic |
| Ensemble | Constrained convex stacking | Simple mean, weighted average, Bayesian averaging |

## 30.4 Combinations allowed in the first serious system

```text
Engineered features + fold-local PCA factors
    -> Global Elastic-Net Logistic

Engineered features + fold-local PCA factors
    -> Shallow Global CatBoost

Raw histories
    -> Chronos-2 zero-shot quantiles
    -> P(Up) and uncertainty

Raw histories
    -> Frozen Chronos-2 or MOMENT backbone
    -> Optional fold-local compression
    -> Shared Elastic-Net Logistic head

Promoted independent probabilities
    -> Constrained convex stacking
    -> Beta or Platt calibration
    -> Correctness / reliability model
    -> Date-block bootstrap LCB
    -> Top-20 maximum with a minimum reliability floor
```

## 30.5 Combinations that require ablation before coexistence

The following may coexist only if each contributes unique rolling OOF value:

- Chronos-2 as an independent expert plus Chronos-2 forecast features inside Logistic or CatBoost.
- Chronos-2 zero-shot plus Chronos-2 embedding head.
- Chronos-2 embedding head plus Chronos-2 zero-shot forecast.
- MOMENT embedding head plus TiRex-2 zero-shot forecast.
- PCA factors plus Dynamic Factor Model outputs.
- CatBoost plus XGBoost or LightGBM.
- Expanding-window and rolling-window versions of the same model.

For every pair, compare:

1. Model A alone.
2. Model B alone.
3. A plus B.
4. Incremental OOF Brier score, log loss, selective risk, and weight stability.

## 30.6 Combinations prohibited initially

- Full fine-tuning of multiple foundation models in parallel.
- Forty separate boosted-tree models.
- GNN plus a learned dynamic graph.
- A large MLP or transformer on top of a high-dimensional pretrained embedding.
- CatBoost, XGBoost, LightGBM, Random Forest, and Extra Trees all together without ablation.
- Chronos-2 used simultaneously as zero-shot expert, forecast-feature generator, embedding head, and LoRA model before each path is independently validated.
- Any feature transformation, PCA, scaling, clustering, calibration, or threshold fitted on all 300 months before walk-forward testing.
- Selection of exactly 15-20 predictions without a minimum reliability threshold.

---

# 31. Coverage Audit Against the Research

The final plan now explicitly preserves all research families and decisions:

- Naive and statistical baselines.
- Global and per-indicator classical classifiers.
- Logistic, Elastic Net, SVM, CatBoost, XGBoost, LightGBM, Random Forest, Extra Trees, HistGradientBoosting, and EBM.
- PCA, sparse factors, dynamic factors, PLS, autoencoders, and other representation options.
- Correlation, partial correlation, lead-lag, Granger, cointegration, mutual information, transfer entropy, and graph-discovery options.
- LSTM, GRU, TCN, N-BEATS, N-HiTS, DeepAR, TFT, PatchTST, iTransformer, TimesNet, TimeMixer, Mamba, reservoir computing, and ROCKET/MiniRocket.
- GNN and temporal graph approaches, with their postponement rationale.
- Chronos-2, Chronos-Bolt, legacy TiRex (deferred), TiRex-2, TimesFM, MOMENT, Moirai/Uni2TS, IBM Granite models, TabPFN-TS, FinCast, Toto, Sundial, Lag-Llama, Time-MoE, Timer, TempoPFN, Reverso, TimeGPT, and Kronos.
- Zero-shot, probabilistic zero-shot, frozen forecasts as features, frozen embeddings, linear heads, small MLP heads, CatBoost heads, adapters, LoRA, partial unfreezing, full fine-tuning, and distillation.
- Soft voting, weighted averaging, Bayesian averaging, stacking, online weighting, regime-dependent selection, per-indicator selection, and agreement features.
- Platt, beta, isotonic, Venn-Abers, conformal, bootstrap uncertainty, correctness modeling, abstention, and top-K plus a reliability floor.
- Point-in-time reconstruction, nested walk-forward validation, date grouping, rolling and expanding windows, purging/embargo, calibration, threshold tuning, and a locked final audit.

This matrix is a preservation appendix. The execution order remains the staged plan in Sections 8, 15, 16, and 28 rather than an instruction to run every method simultaneously.
