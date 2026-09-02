import pandas as pd

from forecast_select.downside_challenger_runner import _expansion_only_mask
from forecast_select.downside_challengers import (
    apply_normalized_bidirectional_selector,
)


def _frames(cap: int = 15):
    rows = 20
    replay = pd.DataFrame({
        "origin_position": [120] * rows,
        "indicator_id": [f"X{i}" for i in range(1, rows + 1)],
        "base_up_rank": list(range(1, rows + 1)),
    })
    baseline = replay.copy()
    baseline["regime_cap"] = cap
    baseline["level_c_ready"] = True
    baseline["p_up_selection_score"] = [
        0.90 - index / 100 for index in range(rows)
    ]
    baseline["p_down"] = [0.20] * 19 + [0.95]
    baseline["regime_stress"] = 0.80
    baseline["accepted"] = [True] * cap + [False] * (rows - cap)
    baseline["predicted_direction"] = "Up"
    baseline["regime_base_accepted"] = baseline["accepted"]
    baseline["regime_base_direction"] = "Up"
    baseline["y_true"] = 1.0
    return replay, baseline


def test_normalized_challenger_makes_at_most_one_guarded_replacement():
    replay, baseline = _frames(cap=15)
    result = apply_normalized_bidirectional_selector(
        replay,
        baseline,
        down_threshold=0.65,
        stress_trigger=0.50,
        hard_down_threshold=0.80,
    )
    selected = result[result["accepted"]]
    down = selected[selected["predicted_direction"].eq("Down")]
    assert len(selected) == 15
    assert selected["indicator_id"].nunique() == 15
    assert down["indicator_id"].tolist() == ["X20"]
    assert down["normalized_replacement"].all()


def test_normalized_challenger_keeps_expansion_positions_up():
    replay, baseline = _frames(cap=20)
    result = apply_normalized_bidirectional_selector(
        replay,
        baseline,
        down_threshold=0.65,
        stress_trigger=0.50,
        hard_down_threshold=0.80,
    )
    expansion = result[result["accepted"] & result["base_up_rank"].between(16, 20)]
    assert expansion["predicted_direction"].eq("Up").all()


def test_normalized_challenger_does_not_force_down_calls():
    replay, baseline = _frames(cap=15)
    baseline["p_down"] = 0.20
    result = apply_normalized_bidirectional_selector(
        replay,
        baseline,
        down_threshold=0.65,
        stress_trigger=0.50,
        hard_down_threshold=0.80,
    )
    assert result.loc[result["accepted"], "predicted_direction"].eq("Up").all()


def test_expansion_mask_excludes_both_replacement_types():
    frame = pd.DataFrame(
        {
            "base_up_rank": [16, 17, 18],
            "regime_replacement": [False, True, False],
            "normalized_replacement": [False, False, True],
        }
    )

    assert _expansion_only_mask(frame).tolist() == [True, False, False]
