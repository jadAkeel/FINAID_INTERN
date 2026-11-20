# Project questions

## 2026-09-27 — Why was the Uptrend Selector correlation graph frozen?

**Question:** Why was the indicator correlation graph used in the original Uptrend Selector's second stage estimated from the first 119 chronological rows and then frozen, rather than updated with a rolling window or adapted to changing market conditions?

**Status:** The inspected files and Git history did not establish an explicit rationale for this original decision. A fixed setting does not prove that a frozen graph is better, or that reducing noise or preventing leakage motivated it; those remain undocumented possible explanations.

### Established evidence

- `configs/uptrend_model.yaml`: graph estimation ends at `119`, with weight `alpha = 0.35`.
- `src/forecast_select/uptrend_pipeline.py`: the graph is calculated from changes in the first 119 rows, outside the monthly forecast loop, then reused in later months. Monthly Logistic refitting does not re-estimate this graph.
- `docs/methodology.md` and `docs/EXPERIMENT_REGISTRY.md` describe the frozen graph without an explicit reason for preferring it to a rolling window.
- The `uptrend_pipeline.py` version in commit `1da8a6f` already contains this design.
- Later development in `research/regime_adaptive_selector/README.md` uses a rolling 48-month percentage-return graph through `t-1`, with pair-reliability shrinkage and weight `0.35`. A rolling graph therefore appears later in the project.
- This score-adjustment graph differs from the model's peer/PCA features, which use rolling windows.

### Open questions

1. What was the original historical rationale for freezing the Uptrend graph? Is there a documented comparison isolating frozen versus rolling graph effects?
2. Comparing the complete original Uptrend and Regime pipelines does not isolate the graph, because several components changed together.
3. Any later comparison must hold the other components fixed and respect monthly information availability. No new training or testing was performed in this investigation.

No source code or model settings were changed while recording this question.

## 2026-09-28 — Reconsider the Uptrend indicator prior and its measurement

**Owner's observation:** The concern is broader than whether the prior should receive 50% of the final selection score. The current prior itself may be the wrong quantity, or may be estimated in the wrong way. We should review both questions together before proposing a change.

**Current implementation:** For each eligible indicator, the Uptrend Selector takes the mean of up to 48 most recent *available* Up/Down labels, using only labels known by the forecast origin (`s <= t-2`). It requires at least 24 indicator labels. The final selection score is `0.5 × graph-adjusted Up score + 0.5 × indicator prior`. This prior measures the indicator's historical Up frequency; it does **not** measure the past correctness of our forecasts or the expected reliability of selecting that indicator. For example, at origin 120, X40's prior was `34/48 = 0.7083`.

**Questions to revisit:**

1. Why was a fixed 50% prior weight chosen? Is there documented evidence that it is preferable to a smaller, larger, or context-dependent weight? The configuration establishes the value, but the reviewed records do not establish an isolated optimization of this weight.
2. Is an equally weighted 48-label Up frequency the right estimate for the decision we care about? Would another window, recency weighting, shrinkage for limited/noisy histories, or conditioning on market regime give a more useful causal estimate?
3. Should the quantity being estimated be the chance of an Up move, the chance that an issued call is **correct**, or the expected value of including that call among the monthly 15? These are different targets and must not be treated as interchangeable.
4. Does the prior duplicate information already present in the Logistic features or graph? If so, how much does the added blend actually improve out-of-sample selection, at identical monthly coverage?

**Owner's explicit proposal:** Treat the prior blend weight `w_prior` as unknown, not as an accepted 50% constant. The score would remain `(1 - w_prior) × graph-adjusted Up score + w_prior × indicator prior`, but `w_prior` must earn its value from evidence. One route is a predefined comparison of weights such as 0%, 10%, 20%, 30%, 50%, 70%, and 100%; another is to learn the weight from past, causally available selection outcomes. The owner does not accept an unsupported 50% assumption as a reason to keep the current value. Include the current 50% rule as a comparator, not as the presumed winner.

For a fixed-weight comparison, change only `w_prior`; hold the model scores, prior definition, candidate pool, monthly call count, and other settings constant. Choose any candidate using the designated Tuning data, then report Validation and Confirmation separately without choosing again on their outcomes. For a learned or month-varying weight, use only earlier out-of-sample predictions and outcomes whose labels were already available at that origin; avoid fitting the weight on the same predictions it will be judged on. Because the historical windows have already been examined repeatedly, a persuasive final adoption claim would need genuinely new, untouched months. Do not confuse learning this blend weight with the Logistic model's separate monthly coefficient fitting.

**Evidence and limits:** `configs/uptrend_model.yaml` and `src/forecast_select/indicator_selection.py` define the current 48-label, 50/50 rule. Existing prior-only and local-logistic diagnostics show that the prior is influential, but they do not identify an optimal blend weight or establish a better prior definition. Window-only comparisons vary across Tuning, Validation, and Confirmation. These observations motivate a controlled study; they are not a decision to change the active model.

**For our later review:** First agree on the exact target and metric. Then compare prior definitions and blend weights as separate changes, preserving causal information cutoffs, the same eligible candidates, and the same number of calls per month. Report per-window hits/calls and paired uncertainty, and keep genuinely new data for a final claim. No new training or evaluation was run for this note.
