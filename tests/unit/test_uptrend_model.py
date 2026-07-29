import json

import numpy as np
import pandas as pd
import pytest

from forecast_select.indicator_selection import (
    propagate_correlation_graph,
    select_top_indicators,
)
from forecast_select.uptrend_pipeline import (
    _assert_model_invariants,
    active_model_status,
)


def test_correlation_graph_uses_positive_and_negative_neighbours():
    probability = np.array([0.7, 0.8, 0.2])
    correlation = np.array([
        [0.0, 0.9, -0.8],
        [0.9, 0.0, 0.0],
        [-0.8, 0.0, 0.0],
    ])
    adjusted = propagate_correlation_graph(probability, correlation, alpha=0.35)
    assert adjusted[0] > 0.7
    assert adjusted[1] > 0.5
    assert adjusted[2] < 0.5


def test_top_indicator_selection_is_unique_and_respects_t_minus_two_cutoff():
    history = pd.DataFrame([
        {
            "origin_position": origin,
            "indicator_id": f"X{indicator}",
            "y_true": int(indicator <= 15),
        }
        for origin in range(1, 30)
        for indicator in range(1, 21)
    ])
    current = pd.DataFrame({
        "origin_position": [30] * 20,
        "indicator_id": [f"X{indicator}" for indicator in range(1, 21)],
        "p_up": np.linspace(0.8, 0.4, 20),
        "y_true": np.nan,
    })
    result = select_top_indicators(
        history,
        current,
        cap=15,
        prior_window=24,
        prior_weight=0.5,
        minimum_history_months=12,
        minimum_indicator_history=12,
        availability_lag=1,
    )
    accepted = result[result["accepted"]]
    assert len(accepted) == 15
    assert accepted["indicator_id"].nunique() == 15
    assert (accepted["calibration_fit_through_origin"] == 28).all()


def test_model_status_is_json_serializable():
    payload = active_model_status()
    assert payload["registered_result_matches"] is True
    json.dumps(payload)


def test_model_invariants_reject_non_unique_monthly_selection():
    frame = pd.DataFrame({
        "origin_position": [120] * 15,
        "indicator_id": ["X1"] * 15,
        "accepted": [True] * 15,
        "run_id": ["uptrend_selector_research"] * 15,
        "model_id": ["uptrend_selector"] * 15,
        "model_version": ["initial_release"] * 15,
        "calibration_fit_through_origin": [118] * 15,
        "reliability_fit_through_origin": [118] * 15,
    })
    config = {
        "selection_origins": [120, 120],
        "model_id": "uptrend_selector",
        "model_release": "initial_release",
        "selection": {"monthly_selection_count": 15},
    }
    with pytest.raises(AssertionError, match="15 unique"):
        _assert_model_invariants(frame, config)
