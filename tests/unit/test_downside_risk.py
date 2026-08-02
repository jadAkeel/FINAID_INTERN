from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecast_select.downside_risk import (
    add_known_shock_features,
    apply_downside_risk_gate,
    build_downside_feature_panel,
    build_sudden_drop_labels,
)
from forecast_select.targets import build_targets


def _settings() -> dict:
    with Path("configs/downside_risk_gate.yaml").open(
        encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_sudden_drop_labels_do_not_change_when_later_values_change():
    rows = 90
    frame = pd.DataFrame({
        "Dates": pd.date_range("2010-01-31", periods=rows, freq="ME"),
        "position": range(1, rows + 1),
        "X1": 100.0 + np.sin(np.arange(rows) / 3.0) * 5.0 + np.arange(rows),
    })
    changed = frame.copy()
    changed.loc[70:, "X1"] *= 50.0
    settings = _settings()["shock_definition"]
    before = build_sudden_drop_labels(
        build_targets(frame),
        trailing_window=int(settings["trailing_window"]),
        minimum_history=int(settings["minimum_history"]),
        lower_quantile=float(settings["lower_quantile"]),
        robust_z=float(settings["robust_z"]),
    )
    after = build_sudden_drop_labels(
        build_targets(changed),
        trailing_window=int(settings["trailing_window"]),
        minimum_history=int(settings["minimum_history"]),
        lower_quantile=float(settings["lower_quantile"]),
        robust_z=float(settings["robust_z"]),
    )
    columns = [
        "origin_position",
        "shock_lower_tail",
        "shock_robust_threshold",
        "shock_label_valid",
        "sudden_drop",
    ]
    pd.testing.assert_frame_equal(
        before[before["origin_position"].le(69)][columns].reset_index(drop=True),
        after[after["origin_position"].le(69)][columns].reset_index(drop=True),
    )


def test_downside_features_do_not_change_when_future_values_change():
    rows = 90
    frame = pd.DataFrame({
        "Dates": pd.date_range("2010-01-31", periods=rows, freq="ME"),
        "position": range(1, rows + 1),
        **{
            f"X{indicator}": (
                100.0
                + indicator
                + np.arange(rows) * (0.1 + indicator / 100.0)
            )
            for indicator in range(1, 51)
        },
    })
    changed = frame.copy()
    changed.loc[70:, [f"X{i}" for i in range(1, 51)]] *= 100.0
    settings = _settings()
    before = build_downside_feature_panel(frame, settings)
    after = build_downside_feature_panel(changed, settings)
    numeric = [
        column for column in before.columns
        if column.startswith("risk_")
    ]
    left = before[before["origin_position"].le(71)].sort_values(
        ["origin_position", "indicator_id"]
    )
    right = after[after["origin_position"].le(71)].sort_values(
        ["origin_position", "indicator_id"]
    )
    np.testing.assert_allclose(
        left[numeric].to_numpy(dtype=float),
        right[numeric].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_risk_gate_replaces_high_risk_up_candidate_without_duplicates():
    indicators = [f"X{value}" for value in range(1, 21)]
    base = pd.DataFrame({
        "origin_position": [120] * 20,
        "indicator_id": indicators,
        "selection_score": np.linspace(0.80, 0.61, 20),
        "level_c_ready": [True] * 20,
        "predicted_direction": ["Up"] * 20,
        "accepted": [value <= 15 for value in range(1, 21)],
        "selection_rank": [
            float(value) if value <= 15 else np.nan
            for value in range(1, 21)
        ],
        "rejection_reason": ["" if value <= 15 else "monthly_cap" for value in range(1, 21)],
        "y_true": [1] * 20,
    })
    risk = pd.DataFrame({
        "origin_position": [120] * 20,
        "indicator_id": indicators,
        "p_sudden_drop": [0.99, *np.linspace(0.01, 0.20, 19)],
        "sudden_drop": [False] * 20,
        "shock_label_valid": [True] * 20,
        "risk_fit_through_origin": [118] * 20,
    })
    result = apply_downside_risk_gate(
        base,
        risk,
        penalty=1.0,
        cap=15,
    )
    selected = result[result["accepted"]]
    assert len(selected) == 15
    assert selected["indicator_id"].nunique() == 15
    assert "X1" not in set(selected["indicator_id"])
    assert selected["predicted_direction"].eq("Up").all()


def test_known_shock_feature_respects_target_availability_lag():
    panel = pd.DataFrame({
        "origin_position": [1, 2, 3, 4],
        "indicator_id": ["X1"] * 4,
    })
    labels = pd.DataFrame({
        "origin_position": [1, 2, 3, 4],
        "indicator_id": ["X1"] * 4,
        "shock_label_valid": [True] * 4,
        "sudden_drop": [True, False, False, False],
    })
    settings = _settings()
    result = add_known_shock_features(panel, labels, settings)
    at_origin_three = result[result["origin_position"].eq(3)].iloc[0]
    assert at_origin_three["risk_previous_shock"] == 1.0
    assert at_origin_three["risk_previous_shock_share"] == 1.0
