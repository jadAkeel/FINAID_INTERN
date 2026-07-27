import pandas as pd

from forecast_select.targets import build_targets
from forecast_select.validation import assert_target_alignment


def test_alignment_uses_t_to_t_plus_one():
    frame = pd.DataFrame({"Dates": pd.date_range("2020-01-31", periods=4, freq="ME"), "position": [1, 2, 3, 4], "X1": [10.0, 9.0, 9.0, 11.0]})
    targets = build_targets(frame)
    assert_target_alignment(targets, frame)
    assert targets.loc[0, "y_true"] == 0
    assert targets.loc[1, "y_true"] == 0
    assert targets.loc[2, "y_true"] == 1

