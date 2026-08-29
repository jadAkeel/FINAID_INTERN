from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_percentile(values: pd.Series, minimum_history: int = 12) -> pd.Series:
    """Rank each value against earlier values only."""
    history: list[float] = []
    result: list[float] = []
    for value in pd.to_numeric(values, errors="coerce"):
        if not np.isfinite(value) or len(history) < minimum_history:
            result.append(0.5)
        else:
            result.append(float(np.mean(np.asarray(history) <= value)))
        if np.isfinite(value):
            history.append(float(value))
    return pd.Series(result, index=values.index, dtype=float)


def build_nonselected_peer_features(
    downside_panel: pd.DataFrame,
    base_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize currently non-selected indicators using past-only features."""
    accepted = base_predictions[[
        "origin_position",
        "indicator_id",
        "accepted",
    ]].copy()
    accepted["base_accepted"] = accepted["accepted"].fillna(False).astype(bool)
    current = downside_panel.merge(
        accepted[["origin_position", "indicator_id", "base_accepted"]],
        on=["origin_position", "indicator_id"],
        how="inner",
        validate="one_to_one",
    )
    current = current[~current["base_accepted"]].copy()
    if current.empty:
        raise ValueError("No non-selected indicators are available")

    def share(frame: pd.DataFrame, column: str, predicate) -> float:
        values = pd.to_numeric(frame[column], errors="coerce")
        values = values.where(values.notna())
        if values.notna().sum() == 0:
            return np.nan
        return float(predicate(values).mean())

    rows = []
    for origin, group in current.groupby("origin_position", sort=True):
        returns = pd.to_numeric(group["down_return_1"], errors="coerce")
        momentum = pd.to_numeric(group["down_momentum_3"], errors="coerce")
        lead_column = (
            "down_lead_negative_consensus"
            if "down_lead_negative_consensus" in group
            else "down_negative_share_3"
        )
        rows.append({
            "origin_position": int(origin),
            "peer_nonselected_count": int(len(group)),
            "peer_nonselected_available_count": int(returns.notna().sum()),
            "peer_nonselected_breadth_up": share(
                group, "down_return_1", lambda values: values.gt(0)
            ),
            "peer_nonselected_mean_return": float(returns.mean()),
            "peer_nonselected_median_return": float(returns.median()),
            "peer_nonselected_dispersion": float(returns.std()),
            "peer_nonselected_negative_share": share(
                group, "down_return_1", lambda values: values.lt(0)
            ),
            "peer_nonselected_weak_momentum_share": share(
                group, "down_momentum_3", lambda values: values.lt(0)
            ),
            "peer_nonselected_mean_momentum_3": float(momentum.mean()),
            "peer_nonselected_exhaustion_share": share(
                group,
                "down_exhaustion_flag",
                lambda values: values.eq(1.0),
            ),
            "peer_nonselected_lead_negative_share": share(
                group,
                lead_column,
                lambda values: values.ge(0.5),
            ),
        })
    return pd.DataFrame(rows)


def build_nonselected_indicator_warnings(
    downside_panel: pd.DataFrame,
    base_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Rank each non-selected indicator's current causal weakness."""
    accepted = base_predictions[[
        "origin_position",
        "indicator_id",
        "accepted",
    ]].copy()
    current = downside_panel.merge(
        accepted,
        on=["origin_position", "indicator_id"],
        how="inner",
        validate="one_to_one",
    )
    current = current[~current["accepted"].fillna(False).astype(bool)].copy()
    if current.empty:
        return pd.DataFrame(columns=[
            "origin_position",
            "indicator_id",
            "nonselected_warning_score",
            "nonselected_warning_reason",
        ])
    def flag(column: str, predicate) -> pd.Series:
        values = pd.to_numeric(current[column], errors="coerce")
        return predicate(values).where(values.notna())

    weakness = pd.DataFrame({
        "negative_return": flag("down_return_1", lambda values: values.lt(0)),
        "weak_momentum": flag("down_momentum_3", lambda values: values.lt(0)),
        "negative_streak": flag(
            "down_negative_share_3", lambda values: values.ge(0.5)
        ),
        "exhaustion": flag(
            "down_exhaustion_flag", lambda values: values.eq(1.0)
        ),
    }).astype(float)
    current["nonselected_warning_score"] = weakness.mean(axis=1)
    current["nonselected_warning_reason"] = weakness.apply(
        lambda row: ",".join(
            name for name, value in row.items()
            if pd.notna(value) and bool(value)
        ),
        axis=1,
    )
    return current[[
        "origin_position",
        "indicator_id",
        "nonselected_warning_score",
        "nonselected_warning_reason",
    ]]


def build_regime_features(
    downside_panel: pd.DataFrame,
    peer_features: pd.DataFrame,
    shock_features: pd.DataFrame,
    market_weight: float,
    peer_weight: float,
    shock_weight: float,
) -> pd.DataFrame:
    """Create a causal market-pressure score from market, peers, and shocks."""
    if min(market_weight, peer_weight, shock_weight) < 0:
        raise ValueError("Stress weights must be non-negative")
    total = market_weight + peer_weight + shock_weight
    if total <= 0:
        raise ValueError("At least one stress weight must be positive")

    market = downside_panel.groupby("origin_position", sort=True).agg(
        market_mean_return=("down_market_mean_return", "first"),
        market_breadth=("down_market_breadth", "first"),
        market_breadth_3=("down_market_breadth_3", "first"),
        market_breadth_change_3=("down_market_breadth_change_3", "first"),
        market_dispersion=("down_market_dispersion", "first"),
    ).reset_index()
    result = market.merge(peer_features, on="origin_position", how="left")
    result = result.merge(shock_features, on="origin_position", how="left")
    result = result.sort_values("origin_position").reset_index(drop=True)

    low_rank_columns = [
        "market_mean_return",
        "market_breadth",
        "market_breadth_3",
        "peer_nonselected_breadth_up",
        "peer_nonselected_mean_return",
    ]
    high_rank_columns = [
        "market_breadth_change_3",
        "market_dispersion",
        "peer_nonselected_negative_share",
        "peer_nonselected_weak_momentum_share",
        "peer_nonselected_exhaustion_share",
        "peer_nonselected_lead_negative_share",
        "previous_shock_share",
    ]
    for column in low_rank_columns:
        result[f"{column}_stress"] = 1.0 - _causal_percentile(
            result[column]
        )
    for column in high_rank_columns:
        result[f"{column}_stress"] = _causal_percentile(result[column])

    market_parts = [
        result["market_mean_return_stress"],
        result["market_breadth_stress"],
        result["market_breadth_3_stress"],
        result["market_dispersion_stress"],
    ]
    peer_parts = [
        result["peer_nonselected_breadth_up_stress"],
        result["peer_nonselected_negative_share_stress"],
        result["peer_nonselected_weak_momentum_share_stress"],
        result["peer_nonselected_lead_negative_share_stress"],
    ]
    shock_parts = [
        result["previous_shock_share_stress"],
        result["peer_nonselected_exhaustion_share_stress"],
    ]
    result["market_stress"] = pd.concat(market_parts, axis=1).mean(axis=1)
    result["peer_stress"] = pd.concat(peer_parts, axis=1).mean(axis=1)
    result["shock_stress"] = pd.concat(shock_parts, axis=1).mean(axis=1)
    result["regime_stress"] = (
        market_weight * result["market_stress"]
        + peer_weight * result["peer_stress"]
        + shock_weight * result["shock_stress"]
    ) / total
    result["regime_label"] = np.select(
        [result["regime_stress"].lt(0.35), result["regime_stress"].gt(0.65)],
        ["calm", "stressed"],
        default="mixed",
    )
    return result


def cap_for_stress(
    stress: float,
    minimum_cap: int,
    maximum_cap: int,
    low_stress: float,
    high_stress: float,
    stress_increases_cap: bool = True,
) -> int:
    """Map causal regime stress to an inclusive integer selection cap."""
    if minimum_cap < 1 or maximum_cap < minimum_cap:
        raise ValueError("Selection cap bounds are invalid")
    if not 0.0 <= low_stress < high_stress <= 1.0:
        raise ValueError("Stress cap bounds must satisfy 0 <= low < high <= 1")
    fraction = (float(stress) - low_stress) / (high_stress - low_stress)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if not stress_increases_cap:
        fraction = 1.0 - fraction
    return int(round(minimum_cap + (maximum_cap - minimum_cap) * fraction))


def cap_for_forward_breadth(
    breadth: float,
    minimum_cap: int,
    maximum_cap: int,
    expansion_threshold: float,
) -> int:
    """Use maximum coverage only for a forecasted broad Up regime."""
    if minimum_cap < 1 or maximum_cap < minimum_cap:
        raise ValueError("Selection cap bounds are invalid")
    if not 0.0 < expansion_threshold < 1.0:
        raise ValueError("Expansion threshold must be between zero and one")
    if not np.isfinite(float(breadth)):
        return int(minimum_cap)
    return (
        int(maximum_cap)
        if float(breadth) >= expansion_threshold
        else int(minimum_cap)
    )


def cap_for_forward_breadth_graduated(
    breadth: float,
    minimum_cap: int,
    maximum_cap: int,
    low_threshold: float,
    high_threshold: float,
) -> int:
    """Map forecasted breadth linearly to 15..20 so every intermediate cap is reachable."""
    if minimum_cap < 1 or maximum_cap < minimum_cap:
        raise ValueError("Selection cap bounds are invalid")
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("Graduated breadth thresholds must satisfy 0 <= low < high <= 1")
    if not np.isfinite(float(breadth)):
        return int(minimum_cap)
    fraction = (float(breadth) - low_threshold) / (high_threshold - low_threshold)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return int(round(minimum_cap + (maximum_cap - minimum_cap) * fraction))


def apply_dynamic_base_cap(
    base_predictions: pd.DataFrame,
    regime_features: pd.DataFrame,
    minimum_cap: int,
    maximum_cap: int,
    low_stress: float,
    high_stress: float,
    stress_increases_cap: bool = True,
    forward_breadth_threshold: float | None = None,
    forward_breadth_low: float | None = None,
    forward_breadth_high: float | None = None,
    forward_breadth_cap_mode: str | None = None,
) -> pd.DataFrame:
    """Build the fair all-Up baseline using the same dynamic cap."""
    regime_columns = ["origin_position", "regime_stress"]
    if "forecast_market_breadth" in regime_features:
        regime_columns.append("forecast_market_breadth")
    base = base_predictions.drop(
        columns=regime_columns[1:],
        errors="ignore",
    )
    result = base.merge(
        regime_features[regime_columns],
        on="origin_position",
        how="left",
        validate="many_to_one",
    ).copy()
    result["accepted"] = False
    result["predicted_direction"] = "Up"
    result["selection_rank"] = np.nan
    score_column = (
        "p_up_selection_score"
        if "p_up_selection_score" in result
        else "p_up_calibrated"
    )
    result["selection_score"] = pd.to_numeric(
        result[score_column], errors="coerce"
    ).fillna(pd.to_numeric(result["p_up"], errors="coerce"))
    result["rejection_reason"] = "dynamic_base_monthly_cap"
    result["regime_cap"] = np.nan
    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        current = result.loc[list(positions)].copy()
        stress = float(pd.to_numeric(current["regime_stress"], errors="coerce").iloc[0])
        use_graduated = (
            forward_breadth_cap_mode == "graduated_15_to_20"
            and forward_breadth_low is not None
            and forward_breadth_high is not None
            and "forecast_market_breadth" in current
        )
        if use_graduated:
            origin_cap = cap_for_forward_breadth_graduated(
                float(current["forecast_market_breadth"].iloc[0]),
                minimum_cap,
                maximum_cap,
                float(forward_breadth_low),
                float(forward_breadth_high),
            )
        elif (
            forward_breadth_threshold is not None
            and "forecast_market_breadth" in current
        ):
            origin_cap = cap_for_forward_breadth(
                float(current["forecast_market_breadth"].iloc[0]),
                minimum_cap,
                maximum_cap,
                forward_breadth_threshold,
            )
        else:
            origin_cap = cap_for_stress(
                stress,
                minimum_cap,
                maximum_cap,
                low_stress,
                high_stress,
                stress_increases_cap,
            )
        ready = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["selection_score"].notna()
        ].sort_values(["selection_score", "indicator_id"], ascending=[False, True])
        if len(ready) < origin_cap:
            raise AssertionError(f"Origin {origin} has fewer than {origin_cap} eligible candidates")
        selected = ready.head(origin_cap)
        result.loc[selected.index, "accepted"] = True
        result.loc[selected.index, "selection_rank"] = np.arange(1, origin_cap + 1)
        result.loc[selected.index, "rejection_reason"] = ""
        result.loc[list(positions), "regime_cap"] = origin_cap
    return result


def apply_accuracy_first_selector(
    base_predictions: pd.DataFrame,
    cap: int,
    group_weight: float,
) -> pd.DataFrame:
    """Select a fixed-coverage Up-only cohort from causal adaptive rankings."""
    if cap < 1:
        raise ValueError("cap must be positive")
    if not 0.0 <= group_weight <= 1.0:
        raise ValueError("group_weight must be between zero and one")

    result = base_predictions.copy()
    probability_column = (
        "p_up_generalized_calibrated"
        if "p_up_generalized_calibrated" in result
        else "p_up_calibrated"
    )
    probability = pd.to_numeric(
        result[probability_column], errors="coerce"
    ).fillna(pd.to_numeric(result["p_up"], errors="coerce")).clip(
        1e-6, 1.0 - 1e-6
    )
    relative_logit = pd.to_numeric(
        result.get(
            "asset_group_relative_logit",
            pd.Series(0.0, index=result.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    base_logit = np.log(probability / (1.0 - probability))
    adjusted_logit = base_logit + float(group_weight) * relative_logit
    score = 1.0 / (1.0 + np.exp(-adjusted_logit))

    result["accuracy_first_base_accepted"] = result["accepted"].fillna(
        False
    ).astype(bool)
    result["p_up_fixed_coverage_selection_score"] = pd.to_numeric(
        result.get("p_up_selection_score", probability), errors="coerce"
    )
    result["p_up_selection_score"] = score
    result["p_up_accuracy_first_score"] = score
    result["accuracy_first_group_weight"] = float(group_weight)
    result["accuracy_first_cap"] = int(cap)
    result["predicted_direction"] = "Up"
    result["directional_confidence"] = score
    result["directional_score"] = score
    result["correctness_probability"] = np.nan
    result["correctness_lcb"] = np.nan
    result["selection_score"] = score
    result["selection_mode"] = "accuracy_first_fixed_coverage_up_only"
    result["accepted"] = False
    result["selection_rank"] = np.nan
    result["rejection_reason"] = "accuracy_first_monthly_cap"
    result["regime_cap"] = int(cap)

    for origin, positions in result.groupby(
        "origin_position", sort=True
    ).groups.items():
        current = result.loc[list(positions)]
        ready = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["selection_score"].notna()
        ].sort_values(
            ["selection_score", "indicator_id"],
            ascending=[False, True],
        )
        if len(ready) < cap:
            raise AssertionError(
                f"Origin {origin} has fewer than {cap} eligible candidates"
            )
        selected = ready.head(cap)
        result.loc[selected.index, "accepted"] = True
        result.loc[selected.index, "selection_rank"] = np.arange(
            1, cap + 1
        )
        result.loc[selected.index, "rejection_reason"] = ""

    result["accuracy_first_selection_changed"] = (
        result["accuracy_first_base_accepted"]
        != result["accepted"].fillna(False).astype(bool)
    )
    return result


def apply_regime_adaptive_selector(
    base_predictions: pd.DataFrame,
    downside_predictions: pd.DataFrame,
    regime_features: pd.DataFrame,
    cap: int | None,
    down_threshold: float,
    down_margin: float,
    stress_trigger: float,
    maximum_down_share: float,
    regime_down_bonus: float,
    shock_down_bonus: float,
    hard_down_threshold: float,
    minimum_cap: int | None = None,
    maximum_cap: int | None = None,
    cap_stress_low: float = 0.35,
    cap_stress_high: float = 0.75,
    stress_increases_cap: bool = True,
    forward_breadth_threshold: float | None = None,
    forward_breadth_low: float | None = None,
    forward_breadth_high: float | None = None,
    forward_breadth_cap_mode: str | None = None,
    replacement_margin: float = 0.0,
    maximum_replacements: int = 0,
    allow_down_predictions: bool = True,
    cap_schedule: dict[int, int] | pd.Series | None = None,
) -> pd.DataFrame:
    """Select a variable Up/Down mix without forcing low-confidence Down calls."""
    dynamic_cap = cap is None
    scheduled_cap = cap_schedule is not None
    if scheduled_cap and not dynamic_cap:
        raise ValueError("A cap schedule cannot be combined with a fixed cap")
    if not dynamic_cap and cap < 1:
        raise ValueError("cap must be positive")
    if dynamic_cap and (minimum_cap is None or maximum_cap is None):
        raise ValueError("Dynamic cap requires minimum_cap and maximum_cap")
    if scheduled_cap:
        schedule = {
            int(origin): int(origin_cap)
            for origin, origin_cap in dict(cap_schedule).items()
        }
        invalid = [
            origin_cap
            for origin_cap in schedule.values()
            if not int(minimum_cap) <= origin_cap <= int(maximum_cap)
        ]
        if invalid:
            raise ValueError("Scheduled caps must stay inside the configured range")
    else:
        schedule = {}
    if not 0.5 <= down_threshold < 1.0:
        raise ValueError("down_threshold must be in [0.5, 1.0)")
    if not 0.0 <= maximum_down_share <= 1.0:
        raise ValueError("maximum_down_share must be between zero and one")
    if not 0.0 <= stress_trigger < 1.0:
        raise ValueError("stress_trigger must be in [0, 1)")
    if not 0.0 <= hard_down_threshold < 1.0:
        raise ValueError("hard_down_threshold must be in [0, 1)")
    if replacement_margin < 0.0:
        raise ValueError("replacement_margin must be non-negative")
    if maximum_replacements < 0:
        raise ValueError("maximum_replacements must be non-negative")

    downside_predictions = downside_predictions.copy()
    if "p_down" not in downside_predictions:
        fallback_columns = [
            "p_down_global",
            "p_down_local",
            "p_down_pattern",
            "p_down_indicator_prior",
        ]
        downside_predictions["p_down"] = downside_predictions[
            fallback_columns
        ].mean(axis=1)

    merge_columns = [
        "origin_position",
        "indicator_id",
        "p_down",
        "p_down_global",
        "p_down_local",
        "p_down_pattern",
        "p_down_indicator_prior",
        "down_exhaustion_flag",
        "down_fit_through_origin",
    ]
    result = base_predictions.merge(
        downside_predictions[merge_columns],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        regime_features,
        on="origin_position",
        how="left",
        validate="many_to_one",
    )
    result["regime_base_accepted"] = result["accepted"].fillna(False).astype(bool)
    result["regime_base_direction"] = result["predicted_direction"].astype(str)
    probability_column = (
        "p_up_generalized_calibrated"
        if "p_up_generalized_calibrated" in result
        else "p_up_calibrated"
    )
    p_up = pd.to_numeric(result[probability_column], errors="coerce").fillna(
        pd.to_numeric(result["p_up"], errors="coerce")
    ).clip(1e-6, 1.0 - 1e-6)
    p_up_selection = pd.to_numeric(
        result.get("p_up_selection_score", p_up), errors="coerce"
    ).fillna(p_up).clip(1e-6, 1.0 - 1e-6)
    down_parts = [
        pd.to_numeric(result[column], errors="coerce")
        for column in [
            "p_down_global",
            "p_down_local",
            "p_down_pattern",
            "p_down_indicator_prior",
        ]
    ]
    p_down_base = pd.to_numeric(
        result["p_down"], errors="coerce"
    ).fillna(pd.concat(down_parts, axis=1).mean(axis=1)).fillna(0.5)
    stress_excess = (
        (pd.to_numeric(result["regime_stress"], errors="coerce") - stress_trigger)
        / max(1e-6, 1.0 - stress_trigger)
    ).clip(0.0, 1.0).fillna(0.0)
    previous_shock = pd.to_numeric(
        result["previous_shock"], errors="coerce"
    ).fillna(0.0)
    previous_shock_share = pd.to_numeric(
        result["previous_shock_share"], errors="coerce"
    ).fillna(0.0)
    result["p_up_base"] = p_up
    result["p_up_selection_score"] = p_up_selection
    result["p_down_base"] = p_down_base
    result["p_down"] = (
        p_down_base
        + regime_down_bonus * stress_excess
        + shock_down_bonus * (0.75 * previous_shock + 0.25 * previous_shock_share)
    ).clip(1e-6, 1.0 - 1e-6)
    result["risk_adjusted_up_score"] = (
        p_up_selection * (1.0 - 0.25 * stress_excess)
        - 0.15 * result["p_down"]
    ).clip(1e-6, 1.0)
    result["down_margin_effective"] = (
        float(down_margin) - 0.05 * stress_excess
    )
    result["down_candidate"] = (
        result["p_down"].ge(float(down_threshold))
        & result["p_down"].ge(
            result["p_up_base"] + result["down_margin_effective"]
        )
    )

    result["predicted_direction"] = np.where(
        result["p_down"].gt(result["p_up_base"]), "Down", "Up"
    )
    result["directional_confidence"] = np.maximum(
        result["p_down"], result["p_up_base"]
    )
    result["directional_score"] = result["directional_confidence"]
    result["correctness_probability"] = np.nan
    result["correctness_lcb"] = np.nan
    result["selection_score"] = result["risk_adjusted_up_score"]
    result["selection_mode"] = (
        "regime_adaptive_bidirectional"
        if allow_down_predictions
        else "regime_adaptive_up_only_fallback"
    )
    result["accepted"] = False
    result["selection_rank"] = np.nan
    result["rejection_reason"] = "regime_adaptive_monthly_cap"
    result["regime_cap"] = np.nan
    result["regime_replacement"] = False
    result["replacement_score"] = np.nan

    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        positions = list(positions)
        current = result.loc[positions].copy()
        stress = float(pd.to_numeric(current["regime_stress"], errors="coerce").iloc[0])
        if scheduled_cap:
            if int(origin) not in schedule:
                raise ValueError(f"Cap schedule is missing origin {origin}")
            origin_cap = schedule[int(origin)]
        elif not dynamic_cap:
            origin_cap = int(cap)
        elif (
            forward_breadth_cap_mode == "graduated_15_to_20"
            and forward_breadth_low is not None
            and forward_breadth_high is not None
            and "forecast_market_breadth" in current
        ):
            origin_cap = cap_for_forward_breadth_graduated(
                float(current["forecast_market_breadth"].iloc[0]),
                int(minimum_cap),
                int(maximum_cap),
                float(forward_breadth_low),
                float(forward_breadth_high),
            )
        elif (
            forward_breadth_threshold is not None
            and "forecast_market_breadth" in current
        ):
            origin_cap = cap_for_forward_breadth(
                float(current["forecast_market_breadth"].iloc[0]),
                int(minimum_cap),
                int(maximum_cap),
                forward_breadth_threshold,
            )
        else:
            origin_cap = cap_for_stress(
                stress,
                int(minimum_cap),
                int(maximum_cap),
                cap_stress_low,
                cap_stress_high,
                stress_increases_cap,
            )
        ready = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["selection_score"].notna()
            & current["p_down"].notna()
        ].copy()
        if len(ready) < origin_cap:
            raise AssertionError(f"Origin {origin} has fewer than {origin_cap} eligible candidates")
        base_pool = ready.sort_values(
            ["p_up_selection_score", "indicator_id"],
            ascending=[False, True],
        ).head(origin_cap)
        directional_pool = base_pool
        has_forward_cap = (
            forward_breadth_threshold is not None
            or (
                forward_breadth_cap_mode == "graduated_15_to_20"
                and forward_breadth_low is not None
                and forward_breadth_high is not None
            )
        )
        if (
            dynamic_cap
            and (has_forward_cap or scheduled_cap)
            and origin_cap > int(minimum_cap)
        ):
            # The extra positions are admitted only when the forward breadth
            # model forecasts a broad Up month. Keep the directional overlay
            # on the invariant core so it cannot contradict that expansion.
            directional_pool = base_pool.head(int(minimum_cap))
        excess = max(0.0, (stress - stress_trigger) / max(1e-6, 1.0 - stress_trigger))
        quota = int(round(origin_cap * maximum_down_share * excess))
        if allow_down_predictions:
            hard = directional_pool[
                directional_pool["p_down"].ge(hard_down_threshold)
            ]
            down_pool = directional_pool[
                directional_pool["down_candidate"]
                | directional_pool["p_down"].ge(hard_down_threshold)
            ].sort_values(["p_down", "indicator_id"], ascending=[False, True])
            down_count = min(origin_cap, max(quota, len(hard)))
            down_selected = down_pool.head(down_count)
        else:
            down_selected = base_pool.iloc[0:0]
        selected = base_pool.sort_values(
            ["directional_confidence", "indicator_id"],
            ascending=[False, True],
        )
        down_indices = down_selected.index.intersection(selected.index)
        replacement_pool = ready.loc[
            ~ready.index.isin(base_pool.index)
        ].copy()
        if allow_down_predictions:
            replacement_pool = replacement_pool[
                (
                    replacement_pool["down_candidate"]
                    | replacement_pool["p_down"].ge(hard_down_threshold)
                )
                & replacement_pool["p_down"].ge(
                    replacement_pool["p_up_base"] + float(replacement_margin)
                )
            ].sort_values(
                ["p_down", "indicator_id"], ascending=[False, True]
            )
        else:
            replacement_pool = replacement_pool.iloc[0:0]
        victims = base_pool.sort_values(
            ["p_up_selection_score", "indicator_id"], ascending=[True, True]
        ).head(
            min(
                int(maximum_replacements),
                len(replacement_pool),
                len(base_pool),
            )
        )
        replacement_indices = replacement_pool.head(len(victims)).index
        selected_indices = selected.index.difference(victims.index).union(
            replacement_indices
        )
        up_indices = selected_indices.difference(down_indices).difference(
            replacement_indices
        )
        result.loc[down_indices, "predicted_direction"] = "Down"
        result.loc[up_indices, "predicted_direction"] = "Up"
        result.loc[replacement_indices, "predicted_direction"] = "Down"
        result.loc[selected_indices, "accepted"] = True
        selected = result.loc[list(selected_indices)].sort_values(
            ["directional_confidence", "indicator_id"],
            ascending=[False, True],
        )
        result.loc[selected.index, "selection_rank"] = np.arange(1, len(selected) + 1)
        result.loc[selected.index, "rejection_reason"] = ""
        result.loc[replacement_indices, "regime_replacement"] = True
        result.loc[replacement_indices, "replacement_score"] = (
            result.loc[replacement_indices, "p_down"]
            - result.loc[replacement_indices, "p_up_base"]
        )
        result.loc[positions, "regime_cap"] = origin_cap

    result["regime_selection_changed"] = (
        result["regime_base_accepted"]
        != result["accepted"].fillna(False).astype(bool)
    )
    result["regime_direction_changed"] = (
        result["regime_base_direction"] != result["predicted_direction"]
    )
    result["regime_down_quota"] = (
        result["regime_stress"] - float(stress_trigger)
    ).clip(lower=0.0) * float(maximum_down_share) / max(1e-6, 1.0 - stress_trigger)
    return result


def summarize_regime_adaptive_predictions(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    if selected.empty:
        raise ValueError("No selected labeled predictions are available")
    selected["correct"] = np.where(
        selected["predicted_direction"].eq("Up"),
        selected["y_true"].eq(1.0),
        selected["y_true"].eq(0.0),
    )
    down = selected[selected["predicted_direction"].eq("Down")]
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(len(down)),
        "down_hits": int(down["y_true"].eq(0.0).sum()),
        "down_precision": float(down["y_true"].eq(0.0).mean()) if len(down) else np.nan,
        "average_regime_stress": float(selected["regime_stress"].mean()),
        "changed_calls": int(
            predictions["regime_selection_changed"].sum()
            if "regime_selection_changed" in predictions
            else 0
        ),
        "direction_changed_calls": int(
            predictions.loc[
                predictions["accepted"].fillna(False).astype(bool),
                "regime_direction_changed",
            ].sum()
            if "regime_direction_changed" in predictions
            else 0
        ),
        "replacement_calls": int(
            predictions["regime_replacement"].fillna(False).sum()
            if "regime_replacement" in predictions
            else 0
        ),
        "replacement_months": int(
            predictions.loc[
                predictions.get("regime_replacement", pd.Series(False, index=predictions.index)).fillna(False),
                "origin_position",
            ].nunique()
            if "regime_replacement" in predictions
            else 0
        ),
    }
