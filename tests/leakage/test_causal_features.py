import numpy as np
import pandas as pd
import pytest

from forecast_select.features import build_feature_panel


def test_future_perturbation_does_not_change_earlier_features():
    frame = pd.DataFrame({"Dates": pd.date_range("2020-01-31", periods=30, freq="ME"), "position": range(1, 31), "X1": np.arange(30, dtype=float), "X2": np.arange(30, dtype=float) * 2})
    before = build_feature_panel(frame, availability_lag=1)
    changed = frame.copy()
    changed.loc[25:, "X1"] += 1000000
    after = build_feature_panel(changed, availability_lag=1)
    cols = [c for c in before.columns if c not in {"origin_date", "indicator_id"}]
    left = before[before["origin_position"] <= 25][cols].reset_index(drop=True)
    right = after[after["origin_position"] <= 25][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


@pytest.mark.parametrize(
    "family",
    [
        "trend_persistence",
        "cross_sectional_dynamics",
        "risk_normalized",
    ],
)
def test_future_perturbation_does_not_change_family_features(family):
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=35, freq="ME"),
        "position": range(1, 36),
        "X1": np.arange(1, 36, dtype=float),
        "X2": np.arange(1, 36, dtype=float) ** 1.1,
    })
    before = build_feature_panel(
        frame,
        availability_lag=1,
        feature_families=(family,),
    )
    changed = frame.copy()
    changed.loc[30:, "X1"] += 1_000_000
    after = build_feature_panel(
        changed,
        availability_lag=1,
        feature_families=(family,),
    )
    columns = [
        column
        for column in before.columns
        if column not in {"origin_date", "indicator_id"}
    ]
    pd.testing.assert_frame_equal(
        before.loc[before["origin_position"].le(30), columns].reset_index(
            drop=True
        ),
        after.loc[after["origin_position"].le(30), columns].reset_index(
            drop=True
        ),
    )
