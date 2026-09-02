import numpy as np
import pandas as pd

from forecast_select.down_sensing import (
    apply_guarded_down_policy,
    evaluate_window,
    extreme_down_labels,
    select_variant,
)


def _selected_frame(breadth: float = 0.40) -> pd.DataFrame:
    return pd.DataFrame({
        "origin_position": [100] * 4,
        "indicator_id": ["X1", "X2", "X3", "X4"],
        "p_up_base": [0.70, 0.55, 0.45, 0.62],
        "predicted_direction": ["Up", "Up", "Up", "Up"],
        "y_true": [1.0, 0.0, 1.0, 1.0],
        "forecast_market_breadth": [breadth] * 4,
    })


def _scores() -> pd.DataFrame:
    return pd.DataFrame({
        "origin_position": [100] * 4,
        "indicator_id": ["X1", "X2", "X3", "X4"],
        "p_extreme_down": [0.10, 0.80, 0.90, 0.30],
    })


def test_gate_blocks_replacement_when_breadth_is_high():
    governed = apply_guarded_down_policy(
        _selected_frame(breadth=0.60), _scores(),
        {"breadth_gate": 0.50, "risk_quantile": 0.85,
         "max_replacements": 2, "conviction_ceiling": 0.60},
    )
    assert not governed["policy_replaced"].any()
    assert (governed["policy_direction"] == "Up").all()


def test_replacement_targets_weakest_up_within_ceiling():
    governed = apply_guarded_down_policy(
        _selected_frame(), _scores(),
        {"breadth_gate": 0.50, "risk_quantile": 0.50,
         "max_replacements": 1, "conviction_ceiling": 0.60},
    )
    changed = set(governed.loc[governed["policy_replaced"], "indicator_id"])
    assert changed == {"X3"}
    assert governed.loc[
        governed["indicator_id"] == "X3", "policy_direction"
    ].item() == "Down"
    assert governed.loc[
        governed["indicator_id"] != "X3", "policy_direction"
    ].eq("Up").all()


def test_conviction_ceiling_protects_strong_calls():
    scores = pd.DataFrame({
        "origin_position": [100, 100],
        "indicator_id": ["X1", "X2"],
        "p_extreme_down": [0.95, 0.95],
    })
    selected = pd.DataFrame({
        "origin_position": [100, 100],
        "indicator_id": ["X1", "X2"],
        "p_up_base": [0.65, 0.52],
        "predicted_direction": ["Up", "Up"],
        "y_true": [1.0, 1.0],
        "forecast_market_breadth": [0.40, 0.40],
    })
    governed = apply_guarded_down_policy(
        selected, scores,
        {"breadth_gate": 0.50, "risk_quantile": 0.50,
         "max_replacements": 2, "conviction_ceiling": 0.50},
    )
    assert (governed["policy_direction"] == "Up").all()


def test_evaluate_window_counts_delta_against_baseline():
    panel = pd.DataFrame({
        "origin_position": [100] * 4,
        "accepted": [True, True, True, True],
        **{column: _selected_frame()[column] for column in [
            "indicator_id", "p_up_base", "predicted_direction",
            "y_true", "forecast_market_breadth",
        ]},
    })
    stats = evaluate_window(
        panel, _scores(),
        {"breadth_gate": 0.50, "risk_quantile": 0.50,
         "max_replacements": 2, "conviction_ceiling": 0.60},
    )
    baseline_down_hits = 1
    policy_down_hits = stats["down_hits"]
    assert stats["calls"] == 4
    assert stats["hits_baseline"] == 3
    assert stats["hits_policy"] - stats["hits_baseline"] == (
        policy_down_hits - baseline_down_hits
    )


def test_selection_rule_requires_nonnegative_subwindows():
    tuning = {"a": {"delta_hits": 1}, "b": {"delta_hits": 0}}
    subwindows = {
        "a": [{"delta_hits": -1}, {"delta_hits": 1}],
        "b": [{"delta_hits": 0}, {"delta_hits": 0}],
    }
    chosen, mode = select_variant(tuning, subwindows)
    assert chosen == "b"
    assert mode == "selected_on_tuning_with_internal_stability"


def test_extreme_labels_are_causal_and_boolean():
    shock_frame = pd.DataFrame({
        "origin_position": [1, 2],
        "indicator_id": ["X1", "X1"],
        "target_return": [-0.10, 0.02],
        "shock_lower_tail": [-0.05, -0.05],
    })
    labels = extreme_down_labels(shock_frame)
    assert labels["extreme_down_next"].tolist() == [1.0, 0.0]
    assert np.isfinite(labels["extreme_down_next"]).all()
