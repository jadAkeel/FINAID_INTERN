# Correctness-trained Up/Down mix trial

Run `python -m research.trained_directional_mix` from the repository root. The
script writes `summary.json` and row-level `predictions.parquet` here. It keeps
the existing 15–20 total calls per month and permits 0–5 Down calls without a
Down quota. Each indicator has one selected direction, and a Down call can
replace an Up call or flip a selected Up indicator. This is research only; the
active model and its forecast are not changed.

The meta-model is trained to predict whether an Up or Down call will be right,
using saved causal Up and Down model scores and regime stress. Labels through
origin 148 train the initial model; origins 150–179 choose regularization,
Down floor, and Down margin. The final model uses labels through origin 178,
then evaluates Validation 180–219 and descriptive Confirmation 220–266. At
forecast origin 180, origin 179's outcome is not yet available, so the two-
origin gap is required.

The screen picked `C=10`, a 0.55 Down score floor, and a 0.08 margin. On the
screen it gained 1 hit with just 1 Down call. After refitting, it selected
133 Down calls in Validation, of which 61 were correct (45.9%). Total
Validation performance fell to 373/675 (55.26%) from the active model's
395/675 (58.52%). Confirmation was 485/799 (60.70%) versus 508/799
(63.58%). The correctness model's Validation AUC across both directional
options was 0.5047, close to random ordering. The policy did not generalize
and is not a production candidate.

The actual correctness of next month's calls cannot be known when selecting.
The model can estimate it only from past outcomes. The score produced in this
trial failed the needed out-of-sample discrimination test, so it must not be
shown as a validated per-call correctness probability. Locked origins were
excluded. Validation and Confirmation have also been inspected in earlier
research and are not fresh blind holdouts. The raw V3 Down scores are
walk-forward by origin, but their feature family was chosen in earlier research
using Tuning origins 120–179. The policy screen is therefore exploratory rather
than an untouched holdout.
