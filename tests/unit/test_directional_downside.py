import numpy as np
import pandas as pd

from forecast_select.directional_downside import (
    apply_bidirectional_selector,
    build_directional_downside_features,
)


def _frame(rows: int = 90) -> pd.DataFrame:
    return pd.DataFrame({
        "Dates": pd.date_range("2010-01-31", periods=rows, freq="ME"),
        "position": range(1, rows + 1),
        **{
            f"X{indicator}": (
                100.0
                + indicator
                + np.arange(rows) * (0.1 + indicator / 100.0)
                + np.sin(np.arange(rows) / (2.0 + indicator / 20.0))
            )
            for indicator in range(1, 51)
        },
    })


def test_directional_features_do_not_change_when_future_values_change():
    frame = _frame()
    changed = frame.copy()
    changed.loc[70:, [f"X{i}" for i in range(1, 51)]] *= 100.0
    before = build_directional_downside_features(
        frame,
        lead_correlation_window=36,
        lead_minimum_pairs=12,
    )
    after = build_directional_downside_features(
        changed,
        lead_correlation_window=36,
        lead_minimum_pairs=12,
    )
    columns = [column for column in before if column.startswith("down_")]
    left = before[before["origin_position"].le(71)].sort_values(
        ["origin_position", "indicator_id"]
    )
    right = after[after["origin_position"].le(71)].sort_values(
        ["origin_position", "indicator_id"]
    )
    np.testing.assert_allclose(
        left[columns].to_numpy(dtype=float),
        right[columns].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_bidirectional_selector_can_choose_down_and_keeps_exact_cap():
    indicators = [f"X{value}" for value in range(1, 21)]
    base = pd.DataFrame({
        "origin_position": [120] * 20,
        "indicator_id": indicators,
        "p_up": [0.60] * 20,
        "p_up_calibrated": [0.60] * 20,
        "predicted_direction": ["Up"] * 20,
        "accepted": [value <= 15 for value in range(1, 21)],
        "level_c_ready": [True] * 20,
        "selection_score": [0.60] * 20,
        "selection_rank": [
            float(value) if value <= 15 else np.nan
            for value in range(1, 21)
        ],
        "rejection_reason": [
            "" if value <= 15 else "cap"
            for value in range(1, 21)
        ],
        "y_true": [0.0, 0.0, *([1.0] * 18)],
    })
    downside = pd.DataFrame({
        "origin_position": [120] * 20,
        "indicator_id": indicators,
        "p_down_global": [0.90, 0.85, *([0.30] * 18)],
        "p_down_local": [0.90, 0.85, *([0.30] * 18)],
        "p_down_pattern": [0.90, 0.85, *([0.30] * 18)],
        "p_down_indicator_prior": [0.40] * 20,
        "local_model_available": [True] * 20,
        "pattern_history_rows": [10] * 20,
        "down_fit_through_origin": [118] * 20,
    })
    result = apply_bidirectional_selector(
        base,
        downside,
        local_weight=0.25,
        pattern_weight=0.25,
        down_threshold=0.55,
        down_margin=0.0,
        cap=15,
    )
    selected = result[result["accepted"]]
    assert len(selected) == 15
    assert selected["indicator_id"].nunique() == 15
    assert set(selected[selected["predicted_direction"].eq("Down")][
        "indicator_id"
    ]) == {"X1", "X2"}
    assert selected["correctness_lcb"].isna().all()
    assert selected["correctness_probability"].isna().all()
    assert selected["directional_score"].equals(
        selected["directional_confidence"]
    )
