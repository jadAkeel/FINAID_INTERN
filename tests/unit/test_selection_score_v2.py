import numpy as np
import pandas as pd

from forecast_select.selection_score_v2 import (
    build_selection_features,
    fit_selection_score_v2,
    score_selection_candidates,
    select_with_existing_caps,
)


def _panel() -> pd.DataFrame:
    rows = []
    for origin in range(120, 190):
        for rank in range(1, 26):
            p_up = 0.75 - rank * 0.01
            rows.append({
                "origin_position": origin,
                "indicator_id": f"X{rank}",
                "y_true": float((origin + rank) % 4 != 0),
                "predicted_direction": "Up",
                "p_up_base": p_up,
                "p_down_base": 1.0 - p_up,
                "indicator_prior": p_up - 0.01,
                "indicator_history_rows": origin - 100,
                "asset_group_relative_logit": 0.1,
                "p_up_generalized_calibrated": p_up,
                "p_up_calibrated": p_up - 0.01,
                "regime_stress": 0.4,
                "market_dispersion": 0.2,
                "forecast_market_breadth": 0.6,
                "p_down_global": 0.3,
                "p_down_local": 0.31,
                "p_down_pattern": 0.29,
                "p_down_indicator_prior": 0.3,
                "regime_cap": 15 + origin % 6,
                "level_c_ready": True,
                "adaptive_data_quality_excluded": False,
            })
    return pd.DataFrame(rows)


def test_feature_contract_is_numeric():
    features = build_selection_features(_panel())
    assert len(features.columns) == 14
    assert features.select_dtypes(include=[np.number]).shape[1] == 14


def test_v2_selection_respects_each_existing_cap():
    panel = _panel()
    fitted = fit_selection_score_v2(panel)
    selected = select_with_existing_caps(score_selection_candidates(panel, fitted))
    counts = selected[selected["accepted_v2"]].groupby("origin_position").size()
    expected = panel.groupby("origin_position")["regime_cap"].first().astype(int)
    pd.testing.assert_series_equal(counts, expected, check_names=False)
