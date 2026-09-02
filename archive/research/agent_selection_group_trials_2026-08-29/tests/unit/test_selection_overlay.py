"""Unit tests for the recent-miss + group-stability selection overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_select.selection_overlay import (
    apply_selection_overlay,
    build_group_stability,
    build_recent_misses,
)


def _frame(n_history: int = 30) -> pd.DataFrame:
    """A small frame with three indicators and a long history."""
    rows = []
    # history: X1 always up, X2 always down, X3 mostly up with a few misses
    for origin in range(0, n_history + 5):
        x1_y = 1.0
        x2_y = 0.0
        x3_history = [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
        x3_y = x3_history[origin % len(x3_history)]
        for indicator, group, y in [
            ("X1", "g_a", x1_y),
            ("X2", "g_a", x2_y),
            ("X3", "g_b", x3_y),
        ]:
            rows.append({
                "origin_position": origin,
                "indicator_id": indicator,
                "y_true": y,
                "asset_group": group,
                "p_up": 0.6,
                "p_up_selection_score": 0.6,
            })
    return pd.DataFrame(rows)


def test_recent_misses_use_only_past_labels() -> None:
    frame = _frame()
    misses = build_recent_misses(frame, window_months=4, label_lag=2)
    row = misses[misses["fit_through_origin"] == 20].iloc[0]
    assert row["fit_through_origin"] == 20
    assert row["recent_call_count"] == 4


def test_recent_misses_separate_strong_and_weak() -> None:
    frame = _frame()
    misses = build_recent_misses(frame, window_months=4, label_lag=2)
    x1 = misses[(misses["indicator_id"] == "X1") & (misses["recent_miss_rate"].notna())]["recent_miss_rate"]
    x2 = misses[(misses["indicator_id"] == "X2") & (misses["recent_miss_rate"].notna())]["recent_miss_rate"]
    assert (x1 < 0.1).all()
    assert (x2 > 0.9).all()


def test_recent_misses_excludes_future_y_true() -> None:
    frame = _frame()
    base = build_recent_misses(frame, window_months=4, label_lag=2)
    mutated = frame.copy()
    # Flip a far-future label: this must not affect any rolling miss
    # at fit_through_origin <= 20 (where the rolling window uses months 17..20).
    mutated.loc[(mutated["origin_position"] == 34) & (mutated["indicator_id"] == "X2"), "y_true"] = 1.0
    new = build_recent_misses(mutated, window_months=4, label_lag=2)
    for fit_through in [10, 14, 18, 20]:
        old = base[base["fit_through_origin"] == fit_through].set_index("indicator_id")["recent_miss_rate"]
        new_v = new[new["fit_through_origin"] == fit_through].set_index("indicator_id")["recent_miss_rate"]
        for ind in ["X1", "X2", "X3"]:
            assert old[ind] == pytest.approx(new_v[ind], abs=1e-9)


def test_group_stability_uses_only_past_labels() -> None:
    frame = _frame()
    stability = build_group_stability(frame, window_months=4, label_lag=2)
    assert (stability["fit_through_origin"] < frame["origin_position"].max()).all()


def test_apply_selection_overlay_penalises_high_miss_indicator() -> None:
    frame = _frame(n_history=30)
    misses = build_recent_misses(frame, window_months=6, label_lag=2)
    stability = build_group_stability(frame, window_months=6, label_lag=2)
    out = apply_selection_overlay(
        frame,
        misses,
        stability,
        base_threshold=0.45,
        stability_bonus=0.0,
    )
    target_origin = 32
    x2 = out[(out["origin_position"] == target_origin) & (out["indicator_id"] == "X2")].iloc[0]
    x1 = out[(out["origin_position"] == target_origin) & (out["indicator_id"] == "X1")].iloc[0]
    assert x2["p_up_selection_score"] < x1["p_up_selection_score"]
    assert x2["selection_overlay_miss_penalty"] > 0.0


def test_apply_selection_overlay_preserves_baseline_score() -> None:
    frame = _frame()
    misses = build_recent_misses(frame, window_months=4, label_lag=2)
    stability = build_group_stability(frame, window_months=4, label_lag=2)
    out = apply_selection_overlay(
        frame,
        misses,
        stability,
        base_threshold=0.45,
        stability_bonus=0.30,
    )
    assert "p_up_selection_score_baseline" in out
    assert np.allclose(
        out["p_up_selection_score_baseline"], frame["p_up_selection_score"]
    )


def test_apply_selection_overlay_is_finite() -> None:
    frame = _frame()
    misses = build_recent_misses(frame, window_months=4, label_lag=2)
    stability = build_group_stability(frame, window_months=4, label_lag=2)
    out = apply_selection_overlay(
        frame,
        misses,
        stability,
        base_threshold=0.45,
        stability_bonus=0.30,
    )
    assert np.isfinite(out["p_up_selection_score"]).all()
    assert out["p_up_selection_score"].between(0.0, 1.0).all()


def test_missing_columns_are_rejected() -> None:
    empty = pd.DataFrame({"origin_position": [1]})
    with pytest.raises(KeyError):
        build_recent_misses(empty, window_months=4)
