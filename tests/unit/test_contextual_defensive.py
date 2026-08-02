import numpy as np
import pandas as pd

from forecast_select.contextual_defensive import (
    apply_contextual_defensive_selector,
    build_causal_market_regime,
)


def test_market_regime_is_invariant_to_future_values():
    frame = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=12, freq="ME"),
        "position": range(1, 13),
        "X1": np.arange(12, dtype=float) + 10,
        "X2": np.arange(12, dtype=float) * -0.5 + 20,
        "X3": np.arange(12, dtype=float) * 2 + 30,
    })
    changed = frame.copy()
    changed.loc[8:, ["X1", "X2", "X3"]] *= 100
    baseline = build_causal_market_regime(frame, availability_lag=1)
    perturbed = build_causal_market_regime(changed, availability_lag=1)
    columns = ["breadth_up", "breadth_mean_3"]
    np.testing.assert_allclose(
        baseline.loc[:8, columns].to_numpy(dtype=float),
        perturbed.loc[:8, columns].to_numpy(dtype=float),
        equal_nan=True,
    )


def _base_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "origin_position": [10] * 4 + [11] * 4,
        "indicator_id": ["X1", "X2", "X3", "X44"] * 2,
        "accepted": [True, True, False, False] * 2,
        "predicted_direction": ["Up", "Up", "Up", "Down"] * 2,
        "p_up": [0.8, 0.6, 0.55, 0.4] * 2,
        "p_up_calibrated": [0.8, 0.6, 0.55, 0.4] * 2,
        "selection_score": [0.8, 0.6, 0.55, 0.4] * 2,
        "selection_rank": [1.0, 2.0, np.nan, np.nan] * 2,
        "rejection_reason": ["", "", "monthly_cap", "monthly_cap"] * 2,
        "level_c_ready": [True] * 8,
        "eligible": [True] * 8,
        "data_quality_ok": [True] * 8,
        "y_true": [1.0, 0.0, 1.0, 1.0] * 2,
    })


def test_contextual_role_replaces_lowest_selection_only_during_stress():
    regime = pd.DataFrame({
        "origin_position": [10, 11],
        "breadth_up": [0.2, 0.8],
        "breadth_mean_3": [0.3, 0.7],
        "regime_observation_through_origin": [9, 10],
    })
    result = apply_contextual_defensive_selector(
        _base_predictions(),
        regime,
        stress_threshold=0.45,
        role_indicators=["X44"],
        cap=2,
    )
    stress = result[result["origin_position"].eq(10)]
    normal = result[result["origin_position"].eq(11)]
    assert set(stress.loc[stress["accepted"], "indicator_id"]) == {
        "X1",
        "X44",
    }
    assert stress.loc[
        stress["indicator_id"].eq("X44"),
        "predicted_direction",
    ].item() == "Up"
    assert set(normal.loc[normal["accepted"], "indicator_id"]) == {
        "X1",
        "X2",
    }
    assert normal.loc[
        normal["indicator_id"].eq("X44"),
        "predicted_direction",
    ].item() == "Down"


def test_contextual_selector_keeps_unique_monthly_cap():
    regime = pd.DataFrame({
        "origin_position": [10, 11],
        "breadth_up": [0.2, 0.2],
        "breadth_mean_3": [0.3, 0.3],
        "regime_observation_through_origin": [9, 10],
    })
    result = apply_contextual_defensive_selector(
        _base_predictions(),
        regime,
        stress_threshold=0.45,
        role_indicators=["X3", "X44"],
        cap=2,
    )
    counts = result[result["accepted"]].groupby(
        "origin_position"
    )["indicator_id"].agg(["count", "nunique"])
    assert counts["count"].eq(2).all()
    assert counts["nunique"].eq(2).all()
