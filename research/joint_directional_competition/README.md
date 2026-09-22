# Joint directional competition check

Run `python -m research.joint_directional_competition` from the repository root.
This research command does not change the active model or its forecast.

For each eligible indicator, an Up score and a Down score are calibrated using
only labels through origin 148. Their equal-weight complementary combination
produces one coherent directional pair. The stronger direction for each
indicator competes for the active model's original monthly cap of 15–20 calls.
There is no Down quota or maximum Down count, and an indicator outside the old
Up-selected pool can enter. The rule and weights are fixed before evaluating
Validation 180–219 and Confirmation 220–266.

| Window | Joint rule | Active reference | Down calls | Correctness AUC |
|---|---:|---:|---:|---:|
| Validation | 396/675 | 395/675 | 0 | 0.4542 |
| Confirmation | 501/799 | 508/799 | 0 | 0.4587 |

The Validation gain is one hit, followed by seven fewer hits on Confirmation.
The score fails as a correctness ranking: higher-score quintiles are not more
accurate, and correctness AUC is below 0.50 in both windows. The research gate
fails, so this rule is not promoted and its scores must not be presented as
individual confidence probabilities. The full summary and selected rows are
`summary.json` and `predictions.parquet` in this directory.

These windows and the upstream model components have been viewed during earlier
research. This replay is exploratory evidence, not a fresh blind validation.
An untouched future period with observed outcomes is needed for a final
confidence claim.
