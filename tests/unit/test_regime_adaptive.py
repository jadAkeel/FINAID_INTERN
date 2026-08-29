import numpy as np
import pandas as pd
import pytest

from forecast_select import regime_adaptive_pipeline
from forecast_select.regime_adaptive import (
    apply_accuracy_first_selector,
    apply_regime_adaptive_selector,
    build_nonselected_peer_features,
    build_nonselected_indicator_warnings,
    build_regime_features,
    cap_for_forward_breadth,
    cap_for_stress,
)
from forecast_select.regime_adaptive_pipeline import (
    _build_causal_group_relative_strength,
    _selection_row,
)


@pytest.mark.parametrize("cap", [14, 21])
def test_build_rejects_cap_outside_configured_range(
    monkeypatch,
    tmp_path,
    cap,
):
    monkeypatch.setattr(
        regime_adaptive_pipeline,
        "_read_yaml",
        lambda _: {
            "selection": {
                "minimum_selection_count": 15,
                "maximum_selection_count": 20,
            }
        },
    )
    with pytest.raises(ValueError, match="cap must be between 15 and 20"):
        regime_adaptive_pipeline.build_regime_adaptive_selector(
            tmp_path,
            cap=cap,
        )


def _base_frame(cap: int = 4) -> pd.DataFrame:
    indicators = [f"X{i}" for i in range(1, 9)]
    return pd.DataFrame({
        "origin_position": [120] * len(indicators),
        "indicator_id": indicators,
        "p_up": [0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56],
        "p_up_calibrated": [0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56],
        "predicted_direction": ["Up"] * len(indicators),
        "accepted": [True] * cap + [False] * (len(indicators) - cap),
        "level_c_ready": [True] * len(indicators),
        "y_true": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "selection_score": [0.70] * len(indicators),
    })


def _downside_frame(base: pd.DataFrame) -> pd.DataFrame:
    result = base[["origin_position", "indicator_id"]].copy()
    result["p_down_global"] = [0.90, 0.85, 0.30, 0.35, 0.80, 0.75, 0.30, 0.25]
    result["p_down_local"] = result["p_down_global"]
    result["p_down_pattern"] = result["p_down_global"]
    result["p_down_indicator_prior"] = result["p_down_global"]
    result["down_exhaustion_flag"] = 0.0
    result["down_fit_through_origin"] = 118
    return result


def _regime_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "origin_position": [120],
        "market_mean_return": [-0.02],
        "market_breadth": [0.20],
        "market_breadth_3": [0.25],
        "market_breadth_change_3": [-0.20],
        "market_dispersion": [0.10],
        "peer_nonselected_count": [4],
        "peer_nonselected_available_count": [4],
        "peer_nonselected_breadth_up": [0.25],
        "peer_nonselected_mean_return": [-0.02],
        "peer_nonselected_median_return": [-0.02],
        "peer_nonselected_dispersion": [0.10],
        "peer_nonselected_negative_share": [0.75],
        "peer_nonselected_weak_momentum_share": [0.75],
        "peer_nonselected_mean_momentum_3": [-0.05],
        "peer_nonselected_exhaustion_share": [0.50],
        "peer_nonselected_lead_negative_share": [0.75],
        "previous_shock": [0.25],
        "previous_shock_share": [0.25],
        "market_mean_return_stress": [0.80],
        "market_breadth_stress": [0.80],
        "market_breadth_3_stress": [0.80],
        "market_breadth_change_3_stress": [0.80],
        "market_dispersion_stress": [0.80],
        "peer_nonselected_breadth_up_stress": [0.80],
        "peer_nonselected_mean_return_stress": [0.80],
        "peer_nonselected_negative_share_stress": [0.80],
        "peer_nonselected_weak_momentum_share_stress": [0.80],
        "peer_nonselected_exhaustion_share_stress": [0.80],
        "peer_nonselected_lead_negative_share_stress": [0.80],
        "previous_shock_share_stress": [0.80],
        "market_stress": [0.80],
        "peer_stress": [0.80],
        "shock_stress": [0.80],
        "regime_stress": [0.80],
        "regime_label": ["stressed"],
    })


def test_nonselected_peer_features_exclude_base_selected_rows():
    base = _base_frame()
    panel = base[["origin_position", "indicator_id"]].copy()
    panel["down_return_1"] = [-0.01] * 4 + [0.02, -0.03, 0.01, -0.02]
    panel["down_momentum_3"] = [-0.01] * 4 + [-0.02, 0.01, -0.01, 0.02]
    panel["down_exhaustion_flag"] = 0.0
    panel["down_lead_negative_consensus"] = 0.0
    result = build_nonselected_peer_features(panel, base)
    assert result.loc[0, "peer_nonselected_count"] == 4
    assert result.loc[0, "peer_nonselected_breadth_up"] == 0.5


def test_cap_for_stress_is_bounded_and_monotone():
    assert cap_for_stress(0.20, 15, 20, 0.35, 0.75) == 15
    assert cap_for_stress(0.55, 15, 20, 0.35, 0.75) == 18
    assert cap_for_stress(0.90, 15, 20, 0.35, 0.75) == 20
    assert cap_for_stress(0.20, 15, 20, 0.35, 0.75, False) == 20
    assert cap_for_stress(0.90, 15, 20, 0.35, 0.75, False) == 15


def test_forward_breadth_expands_only_in_predicted_calm_regime():
    assert cap_for_forward_breadth(0.64, 15, 20, 0.65) == 15
    assert cap_for_forward_breadth(0.65, 15, 20, 0.65) == 20
    assert cap_for_forward_breadth(float("nan"), 15, 20, 0.65) == 15

    with pytest.raises(ValueError, match="Expansion threshold"):
        cap_for_forward_breadth(0.70, 15, 20, 1.0)


def test_nonselected_warning_panel_keeps_indicator_level_reasons():
    base = _base_frame()
    panel = base[["origin_position", "indicator_id"]].copy()
    panel["down_return_1"] = [-0.01] * 4 + [0.02, -0.03, 0.01, -0.02]
    panel["down_momentum_3"] = [-0.01] * 4 + [-0.02, 0.01, -0.01, 0.02]
    panel["down_negative_share_3"] = [0.0] * 4 + [0.5, 1.0, 0.0, 0.0]
    panel["down_exhaustion_flag"] = [0.0] * 4 + [0.0, 1.0, 0.0, 0.0]
    warnings = build_nonselected_indicator_warnings(panel, base)
    assert set(warnings["indicator_id"]) == {"X5", "X6", "X7", "X8"}
    x6 = warnings.loc[warnings["indicator_id"].eq("X6")].iloc[0]
    assert x6["nonselected_warning_score"] == 0.75
    assert "negative_return" in x6["nonselected_warning_reason"]


def test_regime_adaptive_selector_can_choose_down_and_respects_variable_cap():
    base = _base_frame()
    downside = _downside_frame(base)
    result = apply_regime_adaptive_selector(
        base,
        downside,
        _regime_frame(),
        cap=4,
        down_threshold=0.55,
        down_margin=0.0,
        stress_trigger=0.50,
        maximum_down_share=0.75,
        regime_down_bonus=0.0,
        shock_down_bonus=0.0,
        hard_down_threshold=0.80,
    )
    selected = result[result["accepted"]]
    assert len(selected) == 4
    assert selected["indicator_id"].nunique() == 4
    assert set(selected[selected["predicted_direction"].eq("Down")][
        "indicator_id"
    ]) == {"X1", "X2"}


def test_forward_expansion_keeps_positions_above_minimum_up():
    base = _base_frame()
    regime = _regime_frame()
    regime["forecast_market_breadth"] = 0.70
    result = apply_regime_adaptive_selector(
        base,
        _downside_frame(base),
        regime,
        cap=None,
        minimum_cap=4,
        maximum_cap=6,
        cap_stress_low=0.35,
        cap_stress_high=0.75,
        stress_increases_cap=False,
        forward_breadth_threshold=0.65,
        down_threshold=0.55,
        down_margin=0.0,
        stress_trigger=0.50,
        maximum_down_share=0.75,
        regime_down_bonus=0.0,
        shock_down_bonus=0.0,
        hard_down_threshold=0.80,
    )
    selected = result[result["accepted"]].sort_values(
        ["p_up_selection_score", "indicator_id"],
        ascending=[False, True],
    )
    assert len(selected) == 6
    assert selected.iloc[4:]["predicted_direction"].eq("Up").all()


def test_causal_cap_schedule_supports_15_17_20_style_expansion():
    base = _base_frame()
    result = apply_regime_adaptive_selector(
        base,
        _downside_frame(base),
        _regime_frame(),
        cap=None,
        minimum_cap=4,
        maximum_cap=6,
        cap_schedule={120: 5},
        down_threshold=0.55,
        down_margin=0.0,
        stress_trigger=0.50,
        maximum_down_share=0.75,
        regime_down_bonus=0.0,
        shock_down_bonus=0.0,
        hard_down_threshold=0.80,
    )
    selected = result[result["accepted"]].sort_values("p_up_selection_score")
    assert len(selected) == 5
    assert selected["regime_cap"].eq(5).all()
    assert selected.iloc[0]["predicted_direction"] == "Up"


def test_cap_schedule_rejects_missing_origin():
    with pytest.raises(ValueError, match="missing origin 120"):
        apply_regime_adaptive_selector(
            _base_frame(),
            _downside_frame(_base_frame()),
            _regime_frame(),
            cap=None,
            minimum_cap=4,
            maximum_cap=6,
            cap_schedule={},
            down_threshold=0.55,
            down_margin=0.0,
            stress_trigger=0.50,
            maximum_down_share=0.75,
            regime_down_bonus=0.0,
            shock_down_bonus=0.0,
            hard_down_threshold=0.80,
        )


def test_up_only_fallback_disables_hard_down_calls_and_replacements():
    base = _base_frame()
    result = apply_regime_adaptive_selector(
        base,
        _downside_frame(base),
        _regime_frame(),
        cap=4,
        down_threshold=0.55,
        down_margin=0.0,
        stress_trigger=0.50,
        maximum_down_share=0.75,
        regime_down_bonus=0.0,
        shock_down_bonus=0.0,
        hard_down_threshold=0.80,
        replacement_margin=0.0,
        maximum_replacements=2,
        allow_down_predictions=False,
    )
    selected = result[result["accepted"]]
    assert selected["predicted_direction"].eq("Up").all()
    assert not result["regime_replacement"].any()
    assert selected["selection_mode"].eq(
        "regime_adaptive_up_only_fallback"
    ).all()


def test_regime_adaptive_selector_can_replace_a_weak_selected_up():
    base = _base_frame()
    downside = _downside_frame(base)
    result = apply_regime_adaptive_selector(
        base,
        downside,
        _regime_frame(),
        cap=4,
        down_threshold=0.55,
        down_margin=0.0,
        stress_trigger=0.50,
        maximum_down_share=0.75,
        regime_down_bonus=0.0,
        shock_down_bonus=0.0,
        hard_down_threshold=0.80,
        replacement_margin=0.05,
        maximum_replacements=1,
    )
    selected = result[result["accepted"]]
    assert len(selected) == 4
    assert "X5" in set(selected["indicator_id"])
    assert "X4" not in set(selected["indicator_id"])
    assert bool(selected.loc[selected["indicator_id"].eq("X5"), "regime_replacement"].iloc[0])
    assert selected.loc[selected["indicator_id"].eq("X5"), "predicted_direction"].iloc[0] == "Down"


def test_regime_selector_uses_optional_group_adjusted_selection_score():
    base = _base_frame()
    base["p_up_selection_score"] = base["p_up_calibrated"]
    base.loc[base["indicator_id"].eq("X5"), "p_up_selection_score"] = 0.90
    downside = _downside_frame(base)
    for column in [
        "p_down_global",
        "p_down_local",
        "p_down_pattern",
        "p_down_indicator_prior",
    ]:
        downside[column] = 0.20
    result = apply_regime_adaptive_selector(
        base,
        downside,
        _regime_frame(),
        cap=4,
        down_threshold=0.65,
        down_margin=0.10,
        stress_trigger=0.50,
        maximum_down_share=0.50,
        regime_down_bonus=0.0,
        shock_down_bonus=0.0,
        hard_down_threshold=0.80,
    )
    selected = set(result.loc[result["accepted"], "indicator_id"])
    assert "X5" in selected
    assert "X4" not in selected


def test_accuracy_first_selector_uses_group_weight_and_requested_cap():
    base = _base_frame()
    base["p_up_generalized_calibrated"] = base["p_up_calibrated"]
    base["asset_group_relative_logit"] = 0.0
    base.loc[
        base["indicator_id"].eq("X5"), "asset_group_relative_logit"
    ] = 2.0
    result = apply_accuracy_first_selector(
        base,
        cap=1,
        group_weight=1.0,
    )
    selected = result[result["accepted"]]
    assert selected["indicator_id"].tolist() == ["X5"]
    assert selected["predicted_direction"].eq("Up").all()
    assert selected["selection_mode"].eq(
        "accuracy_first_fixed_coverage_up_only"
    ).all()
    assert selected["selection_rank"].tolist() == [1.0]


@pytest.mark.parametrize(
    ("cap", "group_weight", "message"),
    [(0, 0.5, "cap must be positive"), (1, 1.1, "group_weight")],
)
def test_accuracy_first_selector_validates_parameters(
    cap,
    group_weight,
    message,
):
    with pytest.raises(ValueError, match=message):
        apply_accuracy_first_selector(
            _base_frame(),
            cap=cap,
            group_weight=group_weight,
        )


def test_group_relative_strength_uses_only_labels_available_by_origin():
    targets = pd.DataFrame({
        "origin_position": np.repeat(np.arange(105, 121), 2),
        "indicator_id": ["X1", "X2"] * 16,
        "y_true": [1.0, 0.0] * 16,
    })
    before = _build_causal_group_relative_strength(
        targets,
        {"group_a": ["X1"], "group_b": ["X2"]},
        set(),
        range(120, 121),
        trailing_months=12,
        label_lag_months=2,
    )
    changed = targets.copy()
    changed.loc[changed["origin_position"].ge(119), "y_true"] = 1.0
    after = _build_causal_group_relative_strength(
        changed,
        {"group_a": ["X1"], "group_b": ["X2"]},
        set(),
        range(120, 121),
        trailing_months=12,
        label_lag_months=2,
    )
    pd.testing.assert_frame_equal(before, after)
    assert before["asset_group_prior_fit_through_origin"].eq(118).all()


def test_regime_features_are_not_allowed_to_use_future_rows():
    rows = pd.DataFrame({
        "origin_position": [120, 121, 122],
        "down_market_mean_return": [-0.01, 0.01, -0.02],
        "down_market_breadth": [0.4, 0.6, 0.2],
        "down_market_breadth_3": [0.4, 0.5, 0.4],
        "down_market_breadth_change_3": [0.0, 0.1, -0.2],
        "down_market_dispersion": [0.1, 0.05, 0.2],
    })
    peer = pd.DataFrame({
        "origin_position": [120, 121, 122],
        **{column: [0.1, 0.2, 0.3] for column in [
            "peer_nonselected_count", "peer_nonselected_available_count",
            "peer_nonselected_breadth_up", "peer_nonselected_mean_return",
            "peer_nonselected_median_return", "peer_nonselected_dispersion",
            "peer_nonselected_negative_share", "peer_nonselected_weak_momentum_share",
            "peer_nonselected_mean_momentum_3", "peer_nonselected_exhaustion_share",
            "peer_nonselected_lead_negative_share",
        ]},
    })
    shock = pd.DataFrame({
        "origin_position": [120, 121, 122],
        "previous_shock": [0.0, 0.0, 1.0],
        "previous_shock_share": [0.0, 0.0, 1.0],
    })
    before = build_regime_features(rows, peer, shock, 0.45, 0.35, 0.20)
    changed = rows.copy()
    changed.loc[2, "down_market_mean_return"] = -999.0
    after = build_regime_features(changed, peer, shock, 0.45, 0.35, 0.20)
    np.testing.assert_allclose(
        before.loc[:1, "regime_stress"],
        after.loc[:1, "regime_stress"],
    )


def test_candidate_must_hold_up_in_each_internal_tuning_window():
    settings = {
        "tuning_origins": [120, 123],
        "validation_origins": [120, 123],
        "selection": {
            "minimum_tuning_down_calls": 2,
            "minimum_tuning_down_precision": 0.5,
            "internal_tuning_windows": [[120, 121], [122, 123]],
            "minimum_internal_down_calls": 1,
            "minimum_internal_down_precision": 0.5,
            "require_nonnegative_internal_hit_delta": True,
            "require_nonnegative_validation_hit_delta": True,
        },
    }
    base = pd.DataFrame({
        "origin_position": [120, 121, 122, 123],
        "accepted": [True] * 4,
        "predicted_direction": ["Up"] * 4,
        "y_true": [0.0, 1.0, 0.0, 1.0],
        "regime_stress": [0.5] * 4,
    })
    stable = base.copy()
    stable["predicted_direction"] = ["Down", "Up", "Down", "Up"]
    stable_row = _selection_row(stable, base, {}, settings, None)
    assert stable_row["qualifying"] is True
    assert stable_row["stable_qualifying"] is True
    assert stable_row["internal_1_hit_delta"] == 1
    assert stable_row["internal_2_hit_delta"] == 1

    unstable = base.copy()
    unstable["predicted_direction"] = ["Down", "Up", "Up", "Down"]
    unstable_row = _selection_row(unstable, base, {}, settings, None)
    assert unstable_row["qualifying"] is True
    assert unstable_row["stable_qualifying"] is False
    assert unstable_row["internal_2_hit_delta"] == -1
