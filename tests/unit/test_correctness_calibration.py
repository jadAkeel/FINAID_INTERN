from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_select.correctness_calibration import (
    CORRECTNESS_STATUS,
    apply_correctness_semantics,
    decision_correctness,
    wilson_lower_bound,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for origin in range(20):
        for indicator, direction, target in [
            ("X1", "Up", float(origin % 3 != 0)),
            ("X2", "Down", float(origin % 4 == 0)),
        ]:
            rows.append({
                "origin_position": origin,
                "indicator_id": indicator,
                "accepted": True,
                "predicted_direction": direction,
                "y_true": target,
                "directional_confidence": 0.6 + origin / 1000,
                "correctness_probability": 0.6 + origin / 1000,
                "correctness_lcb": np.nan,
            })
    return pd.DataFrame(rows)


def test_decision_correctness_handles_up_down_and_unknown() -> None:
    frame = pd.DataFrame({
        "predicted_direction": ["Up", "Up", "Down", "Down", "Hold"],
        "y_true": [1.0, 0.0, 0.0, 1.0, 1.0],
    })
    result = decision_correctness(frame)
    assert result.iloc[:4].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert np.isnan(result.iloc[4])


def test_semantics_do_not_publish_an_individual_correctness_probability() -> None:
    result = apply_correctness_semantics(
        _predictions(), minimum_history_months=3, label_lag_origins=2
    )
    assert result["correctness_probability"].isna().all()
    assert result["correctness_lcb"].isna().all()
    assert result["legacy_correctness_probability"].notna().all()
    assert result["directional_score"].equals(
        result["directional_confidence"]
    )
    assert set(result["correctness_probability_status"]) == {
        CORRECTNESS_STATUS
    }
    ready = result[result["correctness_fit_through_origin"].notna()]
    assert not ready.empty
    assert (
        ready["correctness_fit_through_origin"]
        <= ready["origin_position"] - 2
    ).all()


def test_cohort_monitoring_is_invariant_to_current_and_future_outcomes() -> None:
    original = _predictions()
    changed = original.copy()
    changed.loc[changed["origin_position"].ge(10), "y_true"] = (
        1.0 - changed.loc[changed["origin_position"].ge(10), "y_true"]
    )
    before = apply_correctness_semantics(
        original, minimum_history_months=3, label_lag_origins=2
    )
    after = apply_correctness_semantics(
        changed, minimum_history_months=3, label_lag_origins=2
    )
    mask = before["origin_position"].le(11)
    pd.testing.assert_series_equal(
        before.loc[mask, "cohort_correctness_probability"],
        after.loc[mask, "cohort_correctness_probability"],
    )


def test_wilson_lower_bound_is_conservative() -> None:
    lower = wilson_lower_bound(70, 100)
    assert 0.60 < lower < 0.70
