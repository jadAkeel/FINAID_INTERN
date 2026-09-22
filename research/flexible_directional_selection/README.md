# Flexible directional selection trial

Run `python research/flexible_directional_selection.py` from the repository root.
The script reads the saved active prediction panel and Down V3 predictions,
then writes `summary.json` and row-level `predictions.parquet` here. It does not
change the active model or forecast.

Each indicator contributes its stronger eligible Up or Down call on separate
Tuning-fitted Platt scales. The selector ranks those calls together, allows at
most five Down calls, takes at least 15 calls, and adds positions 16–20 only
above the Tuning-selected score floor. An Up call in the original pool may flip
to Down; an outside Down call may enter; a weak indicator may be left out. The
original `regime_cap` remains the monthly upper bound. All policy thresholds
are selected using origins 120–179. Origins 180–219 are Validation, 220–266
are descriptive Confirmation, and locked origins are excluded.

The selected policy used `down_floor=0.60`, `down_margin=0.08`, and
`extra_floor=0.60`. Validation selected 357/600 correct calls (59.50%),
including 2/5 correct Down calls; the active model selected 395/675 (58.52%).
At matched monthly coverage, the active model's first 15 calls scored 351/600
(58.50%), a six-hit difference. Confirmation selected 445/709 (62.76%) versus
447/709 (63.05%) for the matched active calls. At the original monthly caps,
the joint ranking scored 394/675 versus 395/675 on Validation and 505/799
versus 508/799 on Confirmation. The policy therefore fails the non-locked
screen and is not a production promotion candidate.

The chosen extra floor led to exactly 15 calls in every Validation month, so
the raw accuracy difference from the active model also reflects lower
coverage. Platt fitting and policy choice both use Tuning, making its reported
Tuning result in-sample. The scores are ranking estimates, not validated
individual correctness probabilities. No new blind holdout result is claimed.
