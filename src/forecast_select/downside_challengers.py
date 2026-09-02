from __future__ import annotations

import numpy as np
import pandas as pd


def apply_normalized_bidirectional_selector(
    replay_inputs: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    down_threshold: float,
    stress_trigger: float,
    hard_down_threshold: float,
    normalized_margin: float = 0.05,
    maximum_down_actions: int = 1,
    minimum_cap: int = 15,
) -> pd.DataFrame:
    """Compare Up and Down cross-sectional percentile ranks on one scale."""
    if maximum_down_actions not in {0, 1}:
        raise ValueError("Normalized challenger supports at most one Down action")
    if normalized_margin < 0:
        raise ValueError("normalized_margin must be non-negative")
    keys = ["origin_position", "indicator_id"]
    if replay_inputs.duplicated(keys).any() or baseline_predictions.duplicated(keys).any():
        raise ValueError("Normalized challenger inputs must have unique keys")
    result = baseline_predictions.copy()
    if "base_up_rank" not in result:
        result = result.merge(
            replay_inputs[keys + ["base_up_rank"]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
    result["up_percentile"] = np.nan
    result["down_percentile"] = np.nan
    result["normalized_directional_score"] = np.nan
    result["normalized_replacement"] = False
    result["accepted"] = False
    result["predicted_direction"] = "Up"
    result["selection_rank"] = np.nan
    result["rejection_reason"] = "normalized_bidirectional_monthly_cap"
    result["selection_mode"] = "normalized_percentile_bidirectional"

    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        positions = list(positions)
        current = result.loc[positions].copy()
        cap_values = pd.to_numeric(current["regime_cap"], errors="coerce").dropna()
        if cap_values.empty:
            raise ValueError(f"Baseline cap is unavailable at origin {origin}")
        origin_cap = int(cap_values.iloc[0])
        ready = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["p_up_selection_score"].notna()
            & current["p_down"].notna()
        ].copy()
        if len(ready) < origin_cap:
            raise ValueError(f"Origin {origin} has too few ready candidates")
        ready["up_percentile"] = ready["p_up_selection_score"].rank(
            method="average", pct=True
        )
        ready["down_percentile"] = ready["p_down"].rank(
            method="average", pct=True
        )
        stress = float(pd.to_numeric(ready["regime_stress"], errors="coerce").iloc[0])
        ready["guarded_down"] = (
            ready["p_down"].ge(float(down_threshold))
            & (
                stress >= float(stress_trigger)
                or ready["p_down"].ge(float(hard_down_threshold))
            )
        )
        ready["down_advantage"] = (
            ready["down_percentile"] - ready["up_percentile"]
        )
        base_pool = ready.sort_values(
            ["p_up_selection_score", "indicator_id"],
            ascending=[False, True],
        ).head(origin_cap)
        core = base_pool[base_pool["base_up_rank"].le(minimum_cap)]
        selected_indices = set(base_pool.index)
        down_index = None
        victim_index = None

        if maximum_down_actions:
            core_options = core[
                core["guarded_down"]
                & core["down_advantage"].ge(float(normalized_margin))
            ].sort_values(
                ["down_advantage", "down_percentile", "indicator_id"],
                ascending=[False, False, True],
            )
            replacement_pool = ready[
                ~ready.index.isin(base_pool.index)
                & ready["guarded_down"]
            ].copy()
            weakest_core = core.sort_values(
                ["up_percentile", "indicator_id"],
                ascending=[True, True],
            ).head(1)
            replacement_option = replacement_pool.iloc[0:0]
            replacement_advantage = -np.inf
            if not weakest_core.empty and not replacement_pool.empty:
                weakest_score = float(weakest_core["up_percentile"].iloc[0])
                replacement_pool["replacement_advantage"] = (
                    replacement_pool["down_percentile"] - weakest_score
                )
                replacement_option = replacement_pool[
                    replacement_pool["replacement_advantage"].ge(
                        float(normalized_margin)
                    )
                ].sort_values(
                    [
                        "replacement_advantage",
                        "down_percentile",
                        "indicator_id",
                    ],
                    ascending=[False, False, True],
                ).head(1)
                if not replacement_option.empty:
                    replacement_advantage = float(
                        replacement_option["replacement_advantage"].iloc[0]
                    )
            core_advantage = (
                float(core_options["down_advantage"].iloc[0])
                if not core_options.empty
                else -np.inf
            )
            if replacement_advantage > core_advantage:
                down_index = replacement_option.index[0]
                victim_index = weakest_core.index[0]
                selected_indices.remove(victim_index)
                selected_indices.add(down_index)
            elif not core_options.empty:
                down_index = core_options.index[0]

        result.loc[ready.index, "up_percentile"] = ready["up_percentile"]
        result.loc[ready.index, "down_percentile"] = ready["down_percentile"]
        result.loc[ready.index, "normalized_directional_score"] = ready[
            "up_percentile"
        ]
        if down_index is not None:
            result.loc[down_index, "predicted_direction"] = "Down"
            result.loc[down_index, "normalized_directional_score"] = ready.loc[
                down_index, "down_percentile"
            ]
        if victim_index is not None:
            result.loc[down_index, "normalized_replacement"] = True
        result.loc[list(selected_indices), "accepted"] = True
        selected = result.loc[list(selected_indices)].sort_values(
            ["normalized_directional_score", "indicator_id"],
            ascending=[False, True],
        )
        result.loc[selected.index, "selection_rank"] = np.arange(
            1, len(selected) + 1
        )
        result.loc[selected.index, "rejection_reason"] = ""

    result["regime_replacement"] = result["normalized_replacement"]
    result["regime_selection_changed"] = (
        result["regime_base_accepted"].fillna(False).astype(bool)
        != result["accepted"].fillna(False).astype(bool)
    )
    result["regime_direction_changed"] = (
        result["regime_base_direction"].astype(str)
        != result["predicted_direction"].astype(str)
    )
    return result
