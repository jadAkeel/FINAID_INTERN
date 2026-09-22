"""Tests for open Up/Down competition and the calibration time boundary."""

import numpy as np
import pandas as pd
import pytest

from research.joint_directional_competition import (
    _confidence_audit,
    fit_calibrators,
    select_month,
)


def _month(up_scores, cap=15):
    return pd.DataFrame(
        {
            "origin_position": 180,
            "indicator_id": [f"X{number:02d}" for number in range(len(up_scores))],
            "level_c_ready": True,
            "regime_cap": cap,
            "joint_p_up": up_scores,
            "joint_p_down": 1 - np.asarray(up_scores),
            "accepted": [number < cap for number in range(len(up_scores))],
            "y_true": [0] * len(up_scores),
        }
    )


def test_down_from_outside_old_up_pool_competes_without_down_quota():
    month = _month([0.65] * 15 + [0.10] * 10)
    selected = select_month(month)
    assert len(selected) == 15
    assert selected["indicator_id"].is_unique
    assert selected["direction"].eq("Down").sum() == 10
    assert selected.loc[selected["indicator_id"].eq("X15"), "direction"].item() == "Down"
    assert selected.loc[selected["indicator_id"].eq("X15"), "accepted"].item() is False


def test_all_up_is_allowed_and_monthly_cap_is_enforced():
    selected = select_month(_month([0.8] * 25, cap=20))
    assert len(selected) == 20
    assert selected["direction"].eq("Up").all()
    with pytest.raises(ValueError, match="Fewer eligible"):
        select_month(_month([0.8] * 14, cap=15))


def test_later_labels_cannot_change_frozen_calibration():
    origins = np.repeat(np.arange(120, 150), 2)
    panel = pd.DataFrame(
        {
            "origin_position": origins,
            "p_up_selection_score": np.tile([0.7, 0.3], 30),
            "logistic": np.tile([0.3, 0.7], 30),
            "y_true": np.tile([1, 0], 30),
        }
    )
    original = fit_calibrators(panel, through_origin=148)
    changed = panel.copy()
    changed.loc[changed["origin_position"].eq(149), "y_true"] = 1 - changed.loc[
        changed["origin_position"].eq(149), "y_true"
    ]
    mutated = fit_calibrators(changed, through_origin=148)
    for direction in ("up", "down"):
        np.testing.assert_allclose(
            original[direction].coef_, mutated[direction].coef_
        )


def test_confidence_audit_rejects_reversed_ranking():
    rows = pd.DataFrame(
        {
            "call_score": np.linspace(0.51, 0.90, 20),
            "call_correct": [1] * 10 + [0] * 10,
        }
    )
    audit = _confidence_audit(rows)
    assert audit["correctness_auc"] < 0.5
    assert not audit["higher_score_more_accurate_by_quintile"]
