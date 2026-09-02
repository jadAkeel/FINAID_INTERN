import pandas as pd

from forecast_select.directional_ranker_v1 import (
    build_directional_features,
    fit_directional_ranker,
    score_directional_candidates,
    select_with_existing_caps,
)


def _panel() -> pd.DataFrame:
    rows = []
    for origin in range(120, 190):
        for rank in range(1, 26):
            p_up = 0.8 - rank * 0.02
            rows.append({
                "origin_position": origin,
                "indicator_id": f"X{rank}",
                "asset_group": f"g{rank % 4}",
                "y_true": float((origin + rank) % 5 != 0),
                "p_up_raw": p_up,
                "p_up_calibrated": p_up,
                "p_up_generalized_calibrated": p_up,
                "indicator_prior": p_up - 0.01,
                "indicator_history_rows": origin - 100,
                "asset_group_relative_logit": 0.1,
                "p_down_global": 1.0 - p_up,
                "p_down_local": 1.0 - p_up + 0.01,
                "p_down_pattern": 1.0 - p_up,
                "p_down_indicator_prior": 1.0 - p_up - 0.01,
                "down_return_1": 0.02,
                "down_momentum_3": 0.03,
                "regime_stress": 0.4,
                "market_breadth": 0.6,
                "market_breadth_change_3": 0.01,
                "market_dispersion": 0.2,
                "forecast_market_breadth": 0.6,
                "regime_cap": 15 + origin % 6,
                "level_c_ready": True,
                "adaptive_data_quality_excluded": False,
            })
    return pd.DataFrame(rows)


def test_directional_feature_contract_has_expected_columns():
    features = build_directional_features(_panel())
    assert "indicator_id" in features
    assert "down_disagreement" in features
    assert "regime_uncertainty" in features


def test_directional_selection_respects_existing_caps():
    panel = _panel()
    fitted = fit_directional_ranker(panel)
    selected = select_with_existing_caps(score_directional_candidates(panel, fitted))
    counts = selected[selected["accepted_directional_v1"]].groupby("origin_position").size()
    expected = panel.groupby("origin_position")["regime_cap"].first().astype(int)
    pd.testing.assert_series_equal(counts, expected, check_names=False)
