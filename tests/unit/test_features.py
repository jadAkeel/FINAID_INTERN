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


def test_signed_run_length_resets_after_missing_values():
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=9, freq="ME"),
        "position": range(1, 10),
        "X1": [1.0, 2.0, 3.0, np.nan, 4.0, 5.0, 4.0, 3.0, 2.0],
    })
    panel = build_feature_panel(
        frame,
        availability_lag=1,
        feature_families=("trend_persistence",),
    )
    run = panel.sort_values("origin_position")["signed_run_length"]
    assert run.iloc[4:6].isna().all()
    assert run.iloc[6] == 1.0
    assert run.iloc[7] == -1.0
    assert run.iloc[8] == -2.0


def test_risk_normalized_momentum_is_scale_invariant():
    values = np.array([1, 2, 3, 5, 4, 6, 8, 7, 9, 12, 11, 14, 16], dtype=float)
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=len(values), freq="ME"),
        "position": range(1, len(values) + 1),
        "X1": values,
        "X2": values * 10.0,
    })
    panel = build_feature_panel(
        frame,
        availability_lag=1,
        feature_families=("risk_normalized",),
    )
    pivot = panel.pivot(
        index="origin_position",
        columns="indicator_id",
        values="risk_normalized_momentum_3",
    ).dropna()
    np.testing.assert_allclose(pivot["X1"], pivot["X2"])
    pivot_6 = panel.pivot(
        index="origin_position",
        columns="indicator_id",
        values="risk_normalized_momentum_6",
    ).dropna()
    np.testing.assert_allclose(pivot_6["X1"], pivot_6["X2"])


def test_cross_sectional_rank_changes_do_not_cross_indicator_boundaries():
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=7, freq="ME"),
        "position": range(1, 8),
        "X1": [1, 2, 4, 7, 11, 16, 22],
        "X2": [1, 5, 6, 7, 8, 9, 10],
    })
    panel = build_feature_panel(
        frame,
        availability_lag=1,
        feature_families=("cross_sectional_dynamics",),
    )
    first = panel.groupby("indicator_id").head(1)
    assert first["cross_section_rank_change_1"].isna().all()
