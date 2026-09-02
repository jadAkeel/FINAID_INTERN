import pandas as pd
import pytest

from forecast_select.group_score_challenger import (
    apply_reliability_gated_group_score,
    build_causal_group_reliability,
)


def _frame() -> pd.DataFrame:
    rows = []
    for origin in range(10, 18):
        rows.extend(
            [
                {
                    "origin_position": origin,
                    "indicator_id": "X1",
                    "y_true": 1.0,
                    "p_up": 0.55,
                    "p_up_generalized_calibrated": 0.55,
                    "asset_group_relative_logit": 0.4,
                    "regime_stress": 0.3,
                    "p_down": 0.3,
                },
                {
                    "origin_position": origin,
                    "indicator_id": "X2",
                    "y_true": 0.0,
                    "p_up": 0.45,
                    "p_up_generalized_calibrated": 0.45,
                    "asset_group_relative_logit": -0.4,
                    "regime_stress": 0.3,
                    "p_down": 0.7,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_reliability_uses_only_causally_available_months() -> None:
    reliability = build_causal_group_reliability(
        _frame(), window_months=4, label_lag_months=2, ridge=0.0
    )
    row = reliability.loc[reliability["origin_position"].eq(17)].iloc[0]
    assert row["group_reliability_fit_through_origin"] == 15
    assert row["group_reliability_history_months"] == 4


def test_positive_only_weight_is_bounded() -> None:
    reliability = build_causal_group_reliability(
        _frame(),
        window_months=4,
        ridge=0.0,
        maximum_absolute_weight=0.2,
        positive_only=True,
    )
    assert reliability["group_reliability_weight"].between(0.0, 0.2).all()


def test_apply_score_preserves_rows_and_adds_probabilities() -> None:
    frame = _frame()
    reliability = build_causal_group_reliability(frame, window_months=4)
    result = apply_reliability_gated_group_score(frame, reliability)
    assert len(result) == len(frame)
    assert result["p_up_group_reliability"].between(0.0, 1.0).all()
    assert result["selection_score_group_reliability"].between(0.0, 1.0).all()


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Missing group-reliability columns"):
        build_causal_group_reliability(pd.DataFrame({"origin_position": [1]}), window_months=4)
