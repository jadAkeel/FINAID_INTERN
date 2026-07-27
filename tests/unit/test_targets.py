import pandas as pd

from forecast_select.targets import build_targets


def test_ties_are_down_and_zero_change_is_recorded():
    frame = pd.DataFrame({"Dates": pd.date_range("2020-01-31", periods=3, freq="ME"), "position": [1, 2, 3], "X1": [1.0, 1.0, 2.0]})
    targets = build_targets(frame)
    assert targets.loc[0, "y_true"] == 0
    assert bool(targets.loc[0, "zero_change"])
    assert targets.loc[1, "y_true"] == 1

