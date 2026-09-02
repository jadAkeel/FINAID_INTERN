from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="coerce").clip(1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def build_causal_group_reliability(
    frame: pd.DataFrame,
    *,
    window_months: int,
    label_lag_months: int = 2,
    ridge: float = 1.0,
    maximum_absolute_weight: float = 0.75,
    positive_only: bool = True,
) -> pd.DataFrame:
    """Estimate a causal residual weight for the existing asset-group signal.

    Each historical month contributes equally, so the estimate is not dominated
    by groups containing more indicators. At origin ``t`` only labels through
    ``t - label_lag_months`` are used.
    """
    required = {
        "origin_position",
        "y_true",
        "p_up_generalized_calibrated",
        "asset_group_relative_logit",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing group-reliability columns: {sorted(missing)}")
    if window_months < 1 or label_lag_months < 1:
        raise ValueError("Window and label lag must be positive")
    if ridge < 0.0 or maximum_absolute_weight < 0.0:
        raise ValueError("Ridge and maximum weight must be nonnegative")

    work = frame[list(required)].copy()
    work["base_probability"] = pd.to_numeric(
        work["p_up_generalized_calibrated"], errors="coerce"
    )
    work["group_signal"] = pd.to_numeric(
        work["asset_group_relative_logit"], errors="coerce"
    )
    work["target"] = pd.to_numeric(work["y_true"], errors="coerce")
    work = work.dropna(subset=["base_probability", "group_signal", "target"])
    work["group_signal_centered"] = work["group_signal"] - work.groupby(
        "origin_position"
    )["group_signal"].transform("mean")
    work["residual"] = work["target"] - work["base_probability"]
    work["numerator"] = work["group_signal_centered"] * work["residual"]
    work["denominator"] = work["group_signal_centered"].pow(2)
    monthly = work.groupby("origin_position", as_index=False).agg(
        numerator=("numerator", "mean"),
        denominator=("denominator", "mean"),
    )

    rows: list[dict[str, float | int]] = []
    for origin in sorted(pd.to_numeric(frame["origin_position"]).dropna().unique()):
        fit_through = int(origin) - int(label_lag_months)
        fit_start = fit_through - int(window_months) + 1
        history = monthly[monthly["origin_position"].between(fit_start, fit_through)]
        denominator = float(history["denominator"].sum()) + float(ridge)
        raw_weight = (
            float(history["numerator"].sum()) / denominator
            if denominator > 0.0 and not history.empty
            else 0.0
        )
        lower = 0.0 if positive_only else -float(maximum_absolute_weight)
        weight = float(np.clip(raw_weight, lower, maximum_absolute_weight))
        rows.append(
            {
                "origin_position": int(origin),
                "group_reliability_weight": weight,
                "group_reliability_raw_weight": raw_weight,
                "group_reliability_fit_through_origin": fit_through,
                "group_reliability_history_months": int(len(history)),
            }
        )
    return pd.DataFrame(rows)


def apply_reliability_gated_group_score(
    frame: pd.DataFrame,
    reliability: pd.DataFrame,
    *,
    stress_trigger: float = 0.5,
) -> pd.DataFrame:
    """Apply the adaptive group weight without changing Regime Adaptive logic."""
    result = frame.merge(
        reliability,
        on="origin_position",
        how="left",
        validate="many_to_one",
    )
    weight = pd.to_numeric(
        result["group_reliability_weight"], errors="coerce"
    ).fillna(0.0)
    base = pd.to_numeric(
        result["p_up_generalized_calibrated"], errors="coerce"
    ).fillna(pd.to_numeric(result["p_up"], errors="coerce"))
    signal = pd.to_numeric(
        result["asset_group_relative_logit"], errors="coerce"
    ).fillna(0.0)
    adjusted_logit = _logit(base) + weight * signal
    result["p_up_group_reliability"] = 1.0 / (1.0 + np.exp(-adjusted_logit))

    stress = pd.to_numeric(result["regime_stress"], errors="coerce").fillna(0.0)
    stress_excess = ((stress - stress_trigger) / max(1e-6, 1.0 - stress_trigger)).clip(
        0.0, 1.0
    )
    p_down = pd.to_numeric(result["p_down"], errors="coerce").fillna(0.5)
    result["selection_score_group_reliability"] = (
        result["p_up_group_reliability"] * (1.0 - 0.25 * stress_excess)
        - 0.15 * p_down
    ).clip(1e-6, 1.0)
    return result
