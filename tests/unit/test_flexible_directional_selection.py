"""Behavior tests for the research-only joint directional selector."""

import pandas as pd

from research.flexible_directional_selection import select_month


def _month(up, down, accepted=None, cap=20):
    accepted = accepted or set()
    return pd.DataFrame(
        {
            "origin_position": 180,
            "indicator_id": [f"X{i}" for i in range(len(up))],
            "level_c_ready": True,
            "regime_cap": cap,
            "up_cal": up,
            "down_cal": down,
            "accepted": [f"X{i}" in accepted for i in range(len(up))],
            "y_true": [1] * len(up),
        }
    )


def test_overlay_external_and_avoid_weak_up():
    up = [0.70] * 20
    down = [0.30] * 20
    up[0], down[0] = 0.65, 0.90
    up[19], down[19] = 0.20, 0.86
    up[18] = 0.10
    selected = select_month(
        _month(up, down, {"X0", "X18"}),
        down_floor=0.55,
        down_margin=0.05,
        extra_floor=0.60,
        optional_extra=True,
    )
    sources = selected.set_index("indicator_id")["selection_source"]
    assert sources["X0"] == "down_overlay"
    assert sources["X19"] == "down_external"
    assert "X18" not in sources
    assert len(selected) == 19
    assert selected.loc[selected["direction"].eq("Down"), "call_correct"].eq(1).sum() == 0


def test_optional_extra_slots_and_five_down_limit():
    up = [0.70] * 15 + [0.55] * 10
    down = [0.30] * 25
    month = _month(up, down)
    strict = select_month(
        month, down_floor=0.55, down_margin=0.05, extra_floor=0.60,
        optional_extra=True,
    )
    loose = select_month(
        month, down_floor=0.55, down_margin=0.05, extra_floor=0.50,
        optional_extra=True,
    )
    assert len(strict) == 15
    assert len(loose) == 20

    down[:10] = [0.90 - i * 0.01 for i in range(10)]
    mixed = select_month(
        _month(up, down), down_floor=0.55, down_margin=0.05,
        extra_floor=0.50, optional_extra=True,
    )
    assert len(mixed) == 20
    assert mixed["direction"].eq("Down").sum() == 5
    assert mixed["indicator_id"].is_unique
