import pandas as pd

from forecast_select.unified_controller import apply_unified_controller


def _frame() -> pd.DataFrame:
    rows = []
    for origin in [10, 11]:
        for index, indicator in enumerate(["X1", "X2", "X3", "X4"]):
            rows.append({
                "origin_position": origin,
                "indicator_id": indicator,
                "selection_score": 0.8 - index * 0.05,
                "predicted_direction": "Down" if indicator == "X4" else "Up",
                "accepted": indicator in {"X1", "X2"},
                "level_c_ready": True,
                "eligible": True,
                "data_quality_ok": True,
                "context_stress": origin == 11,
                "context_role_indicators": "X3",
                "risk_percentile": 0.9 if indicator == "X1" else 0.1,
                "y_true": 1 if indicator in {"X1", "X3"} else 0,
            })
    return pd.DataFrame(rows)


def test_controller_keeps_cap_and_uses_risk_and_context_layers():
    result = apply_unified_controller(
        _frame(),
        risk_penalty=0.2,
        down_risk_bonus=0.2,
        context_role_bonus=0.2,
        cap=2,
    )
    selected = result[result["accepted"]]
    counts = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    assert counts["count"].eq(2).all()
    assert counts["nunique"].eq(2).all()
    assert "X1" not in set(
        selected[selected["origin_position"].eq(10)]["indicator_id"]
    )
    assert result["unified_context_role"].eq(
        result["indicator_id"].eq("X3")
    ).all()


def test_controller_can_require_risk_data():
    frame = _frame()
    frame.loc[frame["indicator_id"].eq("X4"), "risk_percentile"] = None
    result = apply_unified_controller(
        frame,
        risk_penalty=0.0,
        down_risk_bonus=0.0,
        context_role_bonus=0.0,
        cap=2,
        require_risk_data=True,
    )
    assert result.loc[result["indicator_id"].eq("X4"), "accepted"].eq(False).all()

