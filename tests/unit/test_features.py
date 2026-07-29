import numpy as np
import pandas as pd

from forecast_select.features import build_feature_panel


def test_momentum_9_is_causal_and_available_after_warmup():
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=15, freq="ME"),
        "position": range(1, 16),
        "X1": np.arange(1, 16, dtype=float),
    })
    panel = build_feature_panel(frame, availability_lag=1)
    x1 = panel[panel["indicator_id"].eq("X1")].sort_values("origin_position")
    assert x1["momentum_9"].iloc[:10].isna().all()
    assert np.isclose(x1["momentum_9"].iloc[10], 10 / 1 - 1)


def test_missing_changes_are_not_encoded_as_down_in_direction_or_breadth():
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=5, freq="ME"),
        "position": range(1, 6),
        "X1": [np.nan, np.nan, 1.0, 2.0, 3.0],
        "X2": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    panel = build_feature_panel(frame, availability_lag=1)
    at_origin = panel[panel["origin_position"].eq(3)].set_index("indicator_id")
    assert pd.isna(at_origin.loc["X1", "direction_1"])
    assert at_origin.loc["X2", "direction_1"] == 1.0
    assert np.isclose(at_origin["cross_section_breadth"].iloc[0], 1.0)


def test_cross_section_rank_distinguishes_indicators_within_origin():
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=4, freq="ME"),
        "position": range(1, 5),
        "X1": [1.0, 2.0, 4.0, 7.0],
        "X2": [1.0, 4.0, 5.0, 6.0],
        "X3": [1.0, 1.0, 1.0, 1.0],
    })
    panel = build_feature_panel(frame, availability_lag=1)
    ranks = panel[panel["origin_position"].eq(4)].set_index("indicator_id")["cross_section_rank"]
    assert np.isclose(ranks.loc["X1"], 1.0)
    assert np.isclose(ranks.loc["X2"], 2.0 / 3.0)
    assert np.isclose(ranks.loc["X3"], 1.0 / 3.0)
