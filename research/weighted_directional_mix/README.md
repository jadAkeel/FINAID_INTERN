# Weighted Up/Down confidence trial

Run `python -m research.weighted_directional_mix` from the repository root.
The script writes `summary.json`, `weight_screen.csv`, and selected rows in
`predictions.parquet`. This is a research-only comparison; the active model is
unchanged.

The trial calibrates four scores using labels through origin 148: the current
Up selection score, raw V3 Down score, Up indicator-history prior, and Down
indicator-history prior. It tests weights 0, 0.25, 0.5, 0.75, and 1 for each
direction, with either opposite-direction evidence or indicator-history prior
as the other arm. Three fixed Down floors/margins make 150 bounded candidates.
Weights and the Down rule are selected by hits on origins 150–179. The chosen
rule is then frozen for Validation 180–219 and descriptive Confirmation
220–266. Every candidate uses the same monthly 15–20 total-call schedule and
allows at most five Down calls without a Down quota.

The highest-screen-hit setting assigned weight 1.0 to each direction's own
score. **None of the 150 candidates selected any Down calls on the screen.**
The frozen setting selected two Down calls on Validation, one correct. Total
Validation hits were 395/675, equal to the active model; Confirmation was
505/799 versus the active model's 508/799. The selected score's correctness
AUC fell from 0.5392 on the screen to 0.4123 on Validation. In Validation's
lowest score quintile, 68.15% of calls were correct; in the highest quintile,
48.15% were correct. A higher score therefore did not mean higher observed
reliability outside the screen.

No tested weight produced a trustworthy high-confidence Up/Down mix. Raw score
magnitudes or a weighted average must not be presented as validated individual
correctness probabilities. The V3 feature family and active Up overlay were
developed using windows previously viewed in earlier research, so these
comparisons are exploratory and not a fresh blind acceptance test. Locked
origins were excluded.
