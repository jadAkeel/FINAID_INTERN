from __future__ import annotations

import numpy as np
import pandas as pd


def apply_unified_controller(
    predictions: pd.DataFrame,
    risk_penalty: float,
    down_risk_bonus: float,
    context_role_bonus: float,
    cap: int,
    require_risk_data: bool = True,
) -> pd.DataFrame:
    """Apply the causal meta-layer without changing the active model."""
    if min(risk_penalty, down_risk_bonus, context_role_bonus) < 0:
        raise ValueError("Controller weights must be non-negative")
    if cap < 1:
        raise ValueError("cap must be positive")
    required = {
        "origin_position",
        "indicator_id",
        "selection_score",
        "predicted_direction",
        "accepted",
        "level_c_ready",
        "eligible",
        "data_quality_ok",
        "context_stress",
        "context_role_indicators",
        "risk_percentile",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Controller input missing columns: {missing}")

    result = predictions.copy()
    result["unified_base_accepted"] = (
        result["accepted"].fillna(False).astype(bool)
    )
    result["unified_base_direction"] = result["predicted_direction"].astype(str)
    result["unified_risk_data_available"] = result["risk_percentile"].notna()
    risk = pd.to_numeric(result["risk_percentile"], errors="coerce")
    result["unified_risk_percentile"] = risk.fillna(1.0).clip(0.0, 1.0)
    result["unified_context_role"] = False
    for index, row in result[["indicator_id", "context_role_indicators"]].iterrows():
        roles = {
            value.strip()
            for value in str(row["context_role_indicators"]).split(",")
            if value.strip()
        }
        result.at[index, "unified_context_role"] = row["indicator_id"] in roles

    direction = result["predicted_direction"].astype(str)
    up = direction.eq("Up")
    down = direction.eq("Down")
    stress_role = (
        result["context_stress"].fillna(False).astype(bool)
        & result["unified_context_role"]
        & up
    )
    result["unified_risk_penalty"] = float(risk_penalty)
    result["unified_down_risk_bonus"] = float(down_risk_bonus)
    result["unified_context_role_bonus"] = float(context_role_bonus)
    result["unified_score"] = pd.to_numeric(
        result["selection_score"], errors="coerce"
    )
    result.loc[up, "unified_score"] = (
        result.loc[up, "unified_score"]
        - float(risk_penalty) * result.loc[up, "unified_risk_percentile"]
    )
    result.loc[down, "unified_score"] = (
        result.loc[down, "unified_score"]
        + float(down_risk_bonus) * result.loc[down, "unified_risk_percentile"]
    )
    result.loc[stress_role, "unified_score"] = (
        result.loc[stress_role, "unified_score"] + float(context_role_bonus)
    )
    result["accepted"] = False
    result["selection_rank"] = np.nan
    result["rejection_reason"] = "unified_monthly_cap"

    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        current = result.loc[positions]
        eligible = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["eligible"].fillna(False).astype(bool)
            & current["data_quality_ok"].fillna(False).astype(bool)
            & current["unified_score"].notna()
        ]
        if require_risk_data:
            eligible = eligible[eligible["unified_risk_data_available"]]
        if len(eligible) < cap:
            raise AssertionError(
                f"Origin {origin} has fewer than {cap} controller candidates"
            )
        accepted = eligible.sort_values(
            ["unified_score", "indicator_id"],
            ascending=[False, True],
        ).head(cap).index
        result.loc[accepted, "accepted"] = True
        result.loc[accepted, "selection_rank"] = np.arange(1, cap + 1)
        result.loc[accepted, "rejection_reason"] = ""

    result["selection_score"] = result["unified_score"]
    result["selection_mode"] = "unified_forecast_controller"
    result["unified_selection_changed"] = (
        result["unified_base_accepted"]
        != result["accepted"].fillna(False).astype(bool)
    )
    return result


def summarize_unified_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    correct = selected["predicted_direction"].eq(
        selected["y_true"].astype(int).map({1: "Up", 0: "Down"})
    )
    down = selected[selected["predicted_direction"].eq("Down")]
    down_correct = down["y_true"].eq(0)
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(len(down)),
        "down_hits": int(down_correct.sum()),
        "down_accuracy": float(down_correct.mean()) if len(down) else np.nan,
        "changed_calls": int(result_count_changed(predictions)),
    }


def result_count_changed(predictions: pd.DataFrame) -> int:
    return int(
        predictions["unified_selection_changed"].fillna(False).astype(bool).sum()
        // 2
    )
