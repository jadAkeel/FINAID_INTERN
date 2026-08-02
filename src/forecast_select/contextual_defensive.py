from __future__ import annotations

import numpy as np
import pandas as pd


def build_causal_market_regime(
    frame: pd.DataFrame,
    availability_lag: int,
) -> pd.DataFrame:
    """Build a market-breadth signal from observations available at origin."""
    if availability_lag < 1:
        raise ValueError("availability_lag must be at least one month")
    indicators = [
        column for column in frame.columns if column.startswith("X")
    ]
    source = frame[indicators].shift(availability_lag)
    returns = source.pct_change(fill_method=None)
    breadth = (
        returns.gt(0).sum(axis=1)
        / returns.notna().sum(axis=1).replace(0, np.nan)
    )
    return pd.DataFrame({
        "origin_position": frame["position"],
        "breadth_up": breadth,
        "breadth_mean_3": breadth.rolling(3, min_periods=2).mean(),
        "regime_observation_through_origin": (
            frame["position"] - availability_lag
        ),
    })


def apply_contextual_defensive_selector(
    base_predictions: pd.DataFrame,
    regime: pd.DataFrame,
    stress_threshold: float,
    role_indicators: list[str],
    cap: int,
) -> pd.DataFrame:
    """Force selected defensive roles Up only when past breadth signals stress."""
    if not 0.0 <= stress_threshold <= 1.0:
        raise ValueError("stress_threshold must be between zero and one")
    if cap < 1:
        raise ValueError("cap must be positive")
    if not role_indicators or len(set(role_indicators)) != len(role_indicators):
        raise ValueError("role_indicators must be non-empty and unique")

    regime_columns = [
        "origin_position",
        "breadth_up",
        "breadth_mean_3",
        "regime_observation_through_origin",
    ]
    monthly_regime = regime[regime_columns].drop_duplicates(
        "origin_position"
    )
    result = base_predictions.merge(
        monthly_regime,
        on="origin_position",
        how="left",
        validate="many_to_one",
    )
    result["context_base_accepted"] = (
        result["accepted"].fillna(False).astype(bool)
    )
    result["context_base_predicted_direction"] = result[
        "predicted_direction"
    ].astype(str)
    result["context_base_p_up"] = pd.to_numeric(
        result["p_up"],
        errors="coerce",
    )
    result["context_stress"] = result["breadth_mean_3"].le(
        float(stress_threshold)
    )
    result["context_stress_threshold"] = float(stress_threshold)
    result["context_role_indicators"] = ",".join(role_indicators)
    result["context_forced_role"] = False
    result["context_replaced_by_role"] = ""

    for origin, positions in result.groupby(
        "origin_position",
        sort=True,
    ).groups.items():
        positions = list(positions)
        if bool(result.loc[positions, "context_stress"].iloc[0]):
            for role in role_indicators:
                role_rows = result.loc[positions]
                role_rows = role_rows[
                    role_rows["indicator_id"].eq(role)
                ]
                if role_rows.empty:
                    continue
                role_index = role_rows.index[0]
                readiness_values = [
                    result.at[role_index, "level_c_ready"],
                    result.at[role_index, "eligible"],
                    result.at[role_index, "data_quality_ok"],
                ]
                role_ready = all(
                    pd.notna(value) and bool(value)
                    for value in readiness_values
                )
                if not role_ready:
                    continue
                result.at[role_index, "predicted_direction"] = "Up"
                result.at[role_index, "p_up"] = max(
                    float(result.at[role_index, "p_up"]),
                    0.500001,
                )
                result.at[role_index, "p_up_calibrated"] = max(
                    float(result.at[role_index, "p_up_calibrated"]),
                    0.500001,
                )
                result.at[role_index, "context_forced_role"] = True
                if not bool(result.at[role_index, "accepted"]):
                    current = result.loc[positions]
                    removable = current[
                        current["accepted"].fillna(False).astype(bool)
                        & ~current["indicator_id"].isin(role_indicators)
                    ].sort_values(
                        ["selection_score", "indicator_id"],
                        ascending=[True, True],
                    )
                    if removable.empty:
                        raise AssertionError(
                            f"Origin {origin} has no removable selected row"
                        )
                    removed_index = removable.index[0]
                    removed_indicator = str(
                        result.at[removed_index, "indicator_id"]
                    )
                    result.at[removed_index, "accepted"] = False
                    result.at[removed_index, "selection_rank"] = np.nan
                    result.at[
                        removed_index,
                        "rejection_reason",
                    ] = "contextual_role_replacement"
                    result.at[
                        removed_index,
                        "context_replaced_by_role",
                    ] = role
                    result.at[role_index, "accepted"] = True
                    result.at[role_index, "rejection_reason"] = ""
                    result.at[
                        role_index,
                        "context_replaced_by_role",
                    ] = removed_indicator

        selected = result.loc[positions]
        selected = selected[
            selected["accepted"].fillna(False).astype(bool)
        ].sort_values(
            ["selection_score", "indicator_id"],
            ascending=[False, True],
        )
        if len(selected) != cap or selected["indicator_id"].nunique() != cap:
            raise AssertionError(
                f"Origin {origin} must contain {cap} unique selections"
            )
        result.loc[selected.index, "selection_rank"] = np.arange(
            1,
            cap + 1,
        )

    result["context_selection_changed"] = (
        result["context_base_accepted"]
        != result["accepted"].fillna(False).astype(bool)
    )
    result["context_direction_changed"] = (
        result["context_base_predicted_direction"]
        != result["predicted_direction"].astype(str)
    )
    return result


def contextual_selection_summary(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    direction = selected["predicted_direction"].eq("Up").astype(int)
    selected["correct"] = direction.eq(selected["y_true"].astype(int))
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(selected["predicted_direction"].eq("Down").sum()),
        "changed_calls": int(
            predictions[
                predictions["context_selection_changed"]
            ].shape[0]
            // 2
        ),
        "forced_direction_rows": int(
            predictions["context_direction_changed"].sum()
        ),
    }
