"""Historical confidence evidence must respect month and origin boundaries."""

import pandas as pd
import pytest

from forecast_select.delivery_reliability import historical_reliability


def _history():
    return pd.DataFrame(
        [
            {
                "origin_position": origin,
                "accepted": True,
                "predicted_direction": "Up" if rank % 2 else "Down",
                "y_true": int(rank % 3 == 0),
            }
            for origin in range(180, 220)
            for rank in range(15)
        ]
    )


def test_historical_evidence_is_cohort_only_and_ignores_future_outcomes():
    history = _history()
    original = historical_reliability(history, replicates=200)
    later = history.copy()
    later["origin_position"] = 220
    later["y_true"] = 1 - later["y_true"]
    changed = historical_reliability(
        pd.concat([history, later], ignore_index=True), replicates=200
    )
    assert original == changed
    assert original["calls"] == 600
    assert original["individual_correctness_probability"] is None
    assert 0 <= original["monthly_block_bootstrap_p05_accuracy"] <= 1
    assert 0 <= original["monthly_block_bootstrap_p95_accuracy"] <= 1


def test_missing_month_or_insufficient_coverage_is_rejected():
    history = _history()
    with pytest.raises(ValueError, match="every month"):
        historical_reliability(history[history["origin_position"].ne(200)])
    with pytest.raises(ValueError, match="15–20"):
        historical_reliability(history.iloc[1:])
