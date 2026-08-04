import pandas as pd
import pytest

from forecast_select.future_forecast import build_direct_horizon_targets
from forecast_select.targets import build_targets


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Dates": pd.date_range("2025-01-31", periods=5, freq="ME"),
        "position": range(1, 6),
        "X1": [10.0, 12.0, 11.0, 14.0, 13.0],
    })


def test_direct_horizon_targets_predict_the_named_future_month_direction():
    targets = build_direct_horizon_targets(_frame(), horizon_months=2)

    first = targets.loc[targets["origin_position"].eq(1)].iloc[0]
    assert first["direction_start_date"] == pd.Timestamp("2025-02-28")
    assert first["target_date"] == pd.Timestamp("2025-03-31")
    assert first["y_true"] == 0.0

    third = targets.loc[targets["origin_position"].eq(3)].iloc[0]
    assert third["direction_start_date"] == pd.Timestamp("2025-04-30")
    assert third["target_date"] == pd.Timestamp("2025-05-31")
    assert third["y_true"] == 0.0

    assert not targets.loc[
        targets["origin_position"].ge(4),
        "target_available",
    ].any()


def test_direct_horizon_targets_reject_nonpositive_horizons():
    with pytest.raises(ValueError, match="must be positive"):
        build_direct_horizon_targets(_frame(), horizon_months=0)


def test_horizon_one_matches_the_registered_target_definition():
    frame = _frame()
    direct = build_direct_horizon_targets(frame, horizon_months=1)
    registered = build_targets(frame)

    assert direct["y_true"].equals(registered["y_true"])
    assert direct["target_available"].equals(registered["target_available"])
