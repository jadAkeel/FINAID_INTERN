# Regime Adaptive Bidirectional Selector

This non-promoting experiment combines the frozen Uptrend Selector, the ordinary Downside probability research, causal market stress, known shock continuation, and the current non-selected peer set.

## Behavior

- Effective monthly cap: `dynamic`.
- Dynamic cap is graduated 15..20 linearly from breadth `0.52` to `0.68` (`15` at low, `20` at high); `--cap` overrides it with a fixed value.
- Observed monthly cap distribution: `{'15': 33, '16': 23, '17': 20, '18': 27, '19': 24, '20': 20}`; average `17.3129`.
- Regime stress activates guarded Down calls in the configured fallback when no policy clears the development gate.
- The full-panel replacement search can replace up to the selected maximum with non-selected indicators when their Down evidence clears the replacement margin.
- Indicators excluded by the downside data-quality gate are removed before selection: `X16`.
- Selection ranking adds a causal `12`-month asset-group relative-strength prior through `t-2`, with weight `0.25`.
- The frozen correlation graph is generalized with a rolling `48`-month signed graph over percentage returns through `t-1`, pair-reliability shrinkage, and alpha `0.35`.
- The non-selected indicators are summarized at each origin and are not treated as future information.
- Each non-selected indicator also gets a causal `nonselected_warning_score` and explainable warning reasons in the prediction artifact.
- Candidate selection requires adequate Down evidence, non-negative hit delta in both internal tuning windows 120-149 and 150-179, and no loss on the declared Validation development window.
- If no candidate passes that stability gate, the configured fallback keeps conservative Down calls enabled without forcing a Down quota or non-selected replacements.
- The group and generalized-correlation overlays are development-stage changes selected after reviewing Tuning and Validation; Confirmation remains descriptive and only locked origins can provide a clean future test.
- Locked origins 268-315 were not read.

## Result

- Selected parameters: `{"allow_down_predictions": true, "down_margin": 0.0, "down_threshold": 0.65, "maximum_down_share": 0.5, "maximum_replacements": 0, "regime_down_bonus": 0.1, "replacement_margin": 0.05, "shock_down_bonus": 0.1, "stress_trigger": 0.5}`
- Selection mode: `guarded_bidirectional_fallback_no_stable_candidate`
- Tuning candidate accuracy: `686/1071` (`64.0523%`).
- Validation candidate accuracy: `395/675` (`58.5185%`).
- Confirmation base accuracy: `63.2040%`
- Confirmation candidate accuracy: `63.5795%`
- Confirmation Down calls / precision: `7 / 71.4286%`
- Promotion eligible: `False`.

## Fixed-coverage accuracy alternative

This separate policy searches causal group weights while enforcing the configured minimum of 15 Up-ranked indicators every month.

- Selected cap / group weight: `15 / 0.25`.
- Development accuracy: `946/1500` (`63.0667%`).
- Confirmation accuracy: `443/705` (`62.8369%`).
- Development temporal block-bootstrap P10: `59.8000%`.
- Coverage versus fixed Top-15: `100.0000%`.
- Confirmation and locked origins were not used to select its cap or group weight; the active model is unchanged.
