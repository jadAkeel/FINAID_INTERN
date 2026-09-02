from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .correctness_calibration import apply_correctness_semantics
from .directional_downside_pipeline import (
    _load_or_build_downside_probabilities,
    directional_downside_predictions_artifact,
)
from .downside_risk import build_sudden_drop_labels
from .forward_regime import build_forward_market_breadth_forecast
from .selection_overlay import (
    apply_selection_overlay,
    build_group_stability as build_overlay_group_stability,
    build_recent_misses as build_overlay_recent_misses,
)
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .io import load_workbook
from .indicator_selection import (
    propagate_correlation_graph,
    reliability_weighted_correlation,
    summarize_selected_predictions,
)
from .regime_adaptive import (
    apply_accuracy_first_selector,
    apply_regime_adaptive_selector,
    apply_dynamic_base_cap,
    build_nonselected_peer_features,
    build_nonselected_indicator_warnings,
    build_regime_features,
    summarize_regime_adaptive_predictions,
)
from .schemas import validate_oof_columns
from .targets import build_targets
from .uptrend_pipeline import ROOT
from .validation import assert_target_alignment


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _logit(values: pd.Series | float) -> pd.Series | float:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def _build_causal_group_relative_strength(
    targets: pd.DataFrame,
    indicator_groups: dict[str, list[str]],
    excluded_indicators: set[str],
    origins: range,
    trailing_months: int,
    label_lag_months: int,
) -> pd.DataFrame:
    """Estimate recent group strength using labels known before each origin."""
    if trailing_months < 1 or label_lag_months < 1:
        raise ValueError("Group-prior windows and label lag must be positive")
    group_by_indicator = {
        str(indicator): str(group)
        for group, indicators in indicator_groups.items()
        for indicator in indicators
    }
    history = targets[["origin_position", "indicator_id", "y_true"]].copy()
    history["asset_group"] = history["indicator_id"].astype(str).map(
        group_by_indicator
    ).fillna("ungrouped")
    history = history[
        ~history["indicator_id"].astype(str).isin(excluded_indicators)
    ].copy()
    rows = []
    available_groups = sorted(set(group_by_indicator.values()) | {"ungrouped"})
    for origin in origins:
        fit_through = int(origin) - int(label_lag_months)
        fit_start = fit_through - int(trailing_months) + 1
        current = history[
            history["origin_position"].between(fit_start, fit_through)
            & history["y_true"].notna()
        ]
        if current.empty:
            raise ValueError(f"No causal group-prior history is available at {origin}")
        market_prior = float(current["y_true"].mean())
        group_priors = current.groupby("asset_group")["y_true"].mean()
        for group in available_groups:
            group_prior = float(group_priors.get(group, market_prior))
            rows.append({
                "origin_position": int(origin),
                "asset_group": group,
                "asset_group_prior": group_prior,
                "asset_group_market_prior": market_prior,
                "asset_group_relative_logit": float(
                    _logit(group_prior) - _logit(market_prior)
                ),
                "asset_group_prior_fit_through_origin": fit_through,
            })
    return pd.DataFrame(rows)


def _build_generalized_correlation_overlay(
    frame: pd.DataFrame,
    base_predictions: pd.DataFrame,
    settings: dict,
    excluded_indicators: set[str],
    indicator_prior_weight: float,
) -> pd.DataFrame:
    """Rebuild the signed graph at each origin using past percentage returns."""
    window = int(settings["window_months"])
    minimum_pairs = int(settings["minimum_pairs"])
    alpha = float(settings["alpha"])
    if window < minimum_pairs:
        raise ValueError("Generalized correlation window is too short")
    if settings["transform"] != "percentage_return":
        raise ValueError("Only percentage_return correlation is supported")
    indicators = [
        column
        for column in frame.columns
        if column.startswith("X") and column not in excluded_indicators
    ]
    prices = frame.set_index("position")[indicators].apply(
        pd.to_numeric, errors="coerce"
    )
    changes = prices.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    rows = []
    for origin, current in base_predictions.groupby(
        "origin_position", sort=True
    ):
        history = changes.loc[
            (changes.index >= int(origin) - window)
            & (changes.index <= int(origin) - 1)
        ]
        graph = reliability_weighted_correlation(
            history, minimum_pairs=minimum_pairs
        )
        ordered = graph.reindex(
            index=current["indicator_id"],
            columns=current["indicator_id"],
        ).fillna(0.0)
        raw = pd.to_numeric(
            current["p_up_raw"], errors="coerce"
        ).fillna(0.5)
        graph_probability = propagate_correlation_graph(
            raw.to_numpy(dtype=float),
            ordered.to_numpy(dtype=float),
            alpha=alpha,
        )
        prior = pd.to_numeric(
            current["indicator_prior"], errors="coerce"
        )
        calibrated = (
            indicator_prior_weight * prior
            + (1.0 - indicator_prior_weight) * graph_probability
        ).fillna(pd.Series(graph_probability, index=current.index))
        rows.append(pd.DataFrame({
            "origin_position": current["origin_position"].astype(int),
            "indicator_id": current["indicator_id"].astype(str),
            "p_up_generalized_graph": graph_probability,
            "p_up_generalized_calibrated": calibrated.to_numpy(dtype=float),
            "generalized_graph_fit_through_origin": int(origin) - 1,
            "generalized_graph_window_months": window,
            "generalized_graph_minimum_pairs": minimum_pairs,
            "generalized_graph_alpha": alpha,
        }))
    return pd.concat(rows, ignore_index=True)


def regime_adaptive_experiment_root(root: Path = ROOT) -> Path:
    return root / "research/regime_adaptive_selector"


def regime_adaptive_predictions_artifact(root: Path = ROOT) -> Path:
    return regime_adaptive_experiment_root(root) / "artifacts/predictions.parquet"


def regime_adaptive_accuracy_first_artifact(root: Path = ROOT) -> Path:
    return (
        regime_adaptive_experiment_root(root)
        / "artifacts/accuracy_first_predictions.parquet"
    )


def regime_adaptive_summary_path(root: Path = ROOT) -> Path:
    return regime_adaptive_experiment_root(root) / "metrics/summary.json"


def _configuration_hash(root: Path, cap: int | None) -> str:
    payload = {
        "project": _read_yaml(root / "configs/config.yaml"),
        "uptrend_model": _read_yaml(root / "configs/uptrend_model.yaml"),
        "directional_downside": _read_yaml(
            root / "configs/directional_downside_model.yaml"
        ),
        "downside_risk": _read_yaml(root / "configs/downside_risk_gate.yaml"),
        "regime_adaptive": _read_yaml(
            root / "configs/regime_adaptive_selector.yaml"
        ),
        "effective_cap": "dynamic" if cap is None else int(cap),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shock_features(
    targets: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    labels = build_sudden_drop_labels(
        targets,
        trailing_window=int(settings["shock_definition"]["trailing_window"]),
        minimum_history=int(settings["shock_definition"]["minimum_history"]),
        lower_quantile=float(settings["shock_definition"]["lower_quantile"]),
        robust_z=float(settings["shock_definition"]["robust_z"]),
    )
    lag = int(settings["availability_lag_months"])
    known = labels[[
        "origin_position",
        "indicator_id",
        "shock_label_valid",
        "sudden_drop",
    ]].copy()
    known["origin_position"] = known["origin_position"] + lag + 1
    known["previous_shock"] = known["sudden_drop"].astype(float).where(
        known["shock_label_valid"].astype(bool)
    )
    valid = known[known["shock_label_valid"].astype(bool)]
    market = valid.groupby("origin_position")["sudden_drop"].mean()
    rows = known.groupby("origin_position", sort=True).agg(
        previous_shock=("previous_shock", "mean"),
    ).reset_index()
    rows["previous_shock_share"] = rows["origin_position"].map(market)
    return rows


def _build_lightweight_regime_panel(
    frame: pd.DataFrame,
    base_panel: pd.DataFrame,
    availability_lag: int,
) -> pd.DataFrame:
    """Build only the causal fields needed by the adaptive overlay."""
    indicators = [column for column in frame.columns if column.startswith("X")]
    source = frame[indicators].shift(availability_lag)
    returns = source.pct_change(fill_method=None)
    market = pd.DataFrame({
        "origin_position": frame["position"],
        "down_market_mean_return": returns.mean(axis=1),
        "down_market_breadth": returns.gt(0).sum(axis=1).div(
            returns.notna().sum(axis=1).replace(0, np.nan)
        ),
        "down_market_dispersion": returns.std(axis=1),
    })
    market["down_market_breadth_3"] = market["down_market_breadth"].rolling(
        3, min_periods=2
    ).mean()
    market["down_market_breadth_change_3"] = market["down_market_breadth"].diff(3)
    parts = []
    for indicator in indicators:
        level = source[indicator]
        current_return = returns[indicator]
        volatility_24 = current_return.rolling(24, min_periods=12).std()
        rise_before_stall = level.shift(2).div(
            level.shift(8).replace(0, np.nan)
        ).sub(1.0)
        stall_return = level.div(level.shift(2).replace(0, np.nan)).sub(1.0)
        rise_score = rise_before_stall.div(
            volatility_24.mul(np.sqrt(6.0)).replace(0, np.nan)
        )
        stall_score = stall_return.abs().div(
            volatility_24.mul(np.sqrt(2.0)).replace(0, np.nan)
        )
        part = pd.DataFrame({
            "origin_position": frame["position"],
            "indicator_id": indicator,
            "down_return_1": current_return,
            "down_momentum_3": level.div(
                level.shift(3).replace(0, np.nan)
            ).sub(1.0),
            "down_negative_share_3": current_return.lt(0).where(
                current_return.notna()
            ).rolling(3, min_periods=2).mean(),
            "down_exhaustion_flag": (
                rise_before_stall.gt(0)
                & rise_score.ge(0.5)
                & stall_score.le(0.5)
            ).astype(float),
        }).merge(
            market,
            on="origin_position",
            how="left",
            validate="one_to_one",
        )
        parts.append(part)
    features = pd.concat(parts, ignore_index=True)
    return features.merge(
        base_panel[[
            "origin_position",
            "indicator_id",
            "origin_date",
            "target_date",
            "y_true",
            "eligible",
            "data_quality_ok",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )


def _build_inputs(
    root: Path,
    settings: dict,
    cap: int,
    base_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    start = int(settings["tuning_origins"][0])
    end = int(settings["confirmation_origins"][1])
    config = _read_yaml(root / "configs/config.yaml")
    frame = load_workbook(
        root / config["data_path"],
        maximum_position=end + 1,
    )
    targets = build_targets(frame)
    assert_target_alignment(targets, frame)
    base_panel = targets[[
        "origin_position",
        "indicator_id",
        "origin_date",
        "target_date",
        "y_true",
        "target_available",
        "value_t",
    ]].copy()
    base_panel["eligible"] = (
        base_panel["target_available"].fillna(False).astype(bool)
        & base_panel["value_t"].notna()
        & base_panel["origin_position"].gt(
            int(config["minimum_history_months"])
        )
    )
    base_panel["data_quality_ok"] = base_panel["eligible"]
    if base_predictions is None:
        base_path = directional_downside_predictions_artifact(root)
        if not base_path.exists():
            raise FileNotFoundError(
                "Frozen directional artifact is missing; "
                "build-directional-downside first"
            )
        base = pd.read_parquet(base_path)
        if (
            "base_accepted" not in base.columns
            or "base_predicted_direction" not in base.columns
        ):
            raise ValueError(
                "Frozen directional artifact lacks the base Uptrend contract"
            )
        if base["locked_evaluation_read"].fillna(True).any():
            raise AssertionError("Frozen directional artifact includes locked evidence")
    else:
        base = base_predictions.copy()
        if int(base["origin_position"].max()) >= int(settings["locked_origins"][0]):
            raise ValueError("Experimental base predictions include locked origins")
        base["base_accepted"] = base["accepted"].fillna(False).astype(bool)
        base["base_predicted_direction"] = base["predicted_direction"].astype(str)
        base["locked_evaluation_read"] = False
    expected_data_hash = sha256_file(root / config["data_path"])
    if set(base["data_hash"].dropna().astype(str).unique()) != {expected_data_hash}:
        raise ValueError("Frozen directional artifact data hash does not match current data")
    expected_origins = set(range(start, end + 1))
    if set(base["origin_position"].astype(int).unique()) != expected_origins:
        raise ValueError("Frozen directional artifact does not cover the required origins")
    base_counts = base[base["base_accepted"].fillna(False)].groupby(
        "origin_position"
    )["indicator_id"].agg(["count", "nunique"])
    if not base_counts["count"].eq(15).all() or not base_counts["nunique"].eq(15).all():
        raise ValueError("Frozen directional artifact does not contain the expected base top-15")
    frozen_base_config_hash = str(base["config_hash"].dropna().iloc[0])
    base["accepted"] = base["base_accepted"].fillna(False).astype(bool)
    base["predicted_direction"] = base["base_predicted_direction"].astype(str)
    base = base[base["origin_position"].between(start, end)].copy()
    downside = _load_or_build_downside_probabilities(root)
    downside = downside[downside["origin_position"].between(start, end)].copy()
    directional_settings = _read_yaml(
        root / "configs/directional_downside_model.yaml"
    )
    directional_panel = _build_lightweight_regime_panel(
        frame,
        base_panel,
        availability_lag=int(directional_settings["availability_lag_months"]),
    )
    directional_panel["down_target"] = 1.0 - directional_panel["y_true"]
    directional_panel = directional_panel[
        directional_panel["origin_position"].between(start, end)
    ].copy()
    peer = build_nonselected_peer_features(directional_panel, base)
    warnings = build_nonselected_indicator_warnings(directional_panel, base)
    downside_risk_settings = _read_yaml(
        root / "configs/downside_risk_gate.yaml"
    )
    shock = _shock_features(targets, downside_risk_settings)
    regime_settings = settings["stress"]
    regime = build_regime_features(
        directional_panel,
        peer,
        shock,
        market_weight=float(regime_settings["market_weight"]),
        peer_weight=float(regime_settings["peer_weight"]),
        shock_weight=float(regime_settings["shock_weight"]),
    )
    forward_settings = settings.get("forward_regime", {})
    if bool(forward_settings.get("enabled", False)):
        excluded = {
            str(value)
            for value in downside_risk_settings.get(
                "excluded_indicators", []
            )
        }
        model_settings = dict(forward_settings["model"])
        model_settings["seed"] = int(config["seed"])
        forward = build_forward_market_breadth_forecast(
            frame,
            targets,
            downside_risk_settings["indicator_groups"],
            excluded,
            range(start, end + 1),
            availability_lag=int(settings["availability_lag_months"]),
            model_settings=model_settings,
        )
        regime = regime.merge(
            forward,
            on="origin_position",
            how="left",
            validate="one_to_one",
        )
    downside_columns = [
        "p_down_global",
        "p_down_local",
        "p_down_pattern",
        "p_down_indicator_prior",
        "down_exhaustion_flag",
        "down_fit_through_origin",
    ]
    base = base.drop(columns=downside_columns, errors="ignore")
    base = base.merge(
        downside[[
            "origin_position",
            "indicator_id",
            *downside_columns,
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    if "p_down" not in base:
        base["p_down"] = base[[
            "p_down_global",
            "p_down_local",
            "p_down_pattern",
            "p_down_indicator_prior",
        ]].mean(axis=1)
    result = base.merge(
        directional_panel[[
            "origin_position",
            "indicator_id",
            "down_return_1",
            "down_momentum_3",
            "down_market_breadth",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        warnings,
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        regime,
        on="origin_position",
        how="left",
        validate="many_to_one",
    )
    if result["origin_position"].nunique() != end - start + 1:
        raise AssertionError("Regime-adaptive inputs do not cover all origins")
    excluded_indicators = {
        str(value)
        for value in downside_risk_settings.get("excluded_indicators", [])
    }
    result["adaptive_data_quality_excluded"] = result["indicator_id"].astype(
        str
    ).isin(excluded_indicators)
    result["adaptive_exclusion_reason"] = np.where(
        result["adaptive_data_quality_excluded"],
        "excluded_by_downside_data_quality_gate",
        "",
    )
    result["level_c_ready_before_adaptive_exclusion"] = result[
        "level_c_ready"
    ].fillna(False).astype(bool)
    result["level_c_ready"] = (
        result["level_c_ready_before_adaptive_exclusion"]
        & ~result["adaptive_data_quality_excluded"]
    )
    correlation_settings = settings["generalized_correlation_overlay"]
    if bool(correlation_settings["enabled"]):
        uptrend_settings = _read_yaml(root / "configs/uptrend_model.yaml")
        correlation_overlay = _build_generalized_correlation_overlay(
            frame,
            result,
            correlation_settings,
            excluded_indicators,
            indicator_prior_weight=float(
                uptrend_settings["selection"]["indicator_prior_weight"]
            ),
        )
        result = result.merge(
            correlation_overlay,
            on=["origin_position", "indicator_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        result["p_up_generalized_graph"] = result["p_up"]
        result["p_up_generalized_calibrated"] = result[
            "p_up_calibrated"
        ]
        result["generalized_graph_fit_through_origin"] = np.nan
        result["generalized_graph_window_months"] = np.nan
        result["generalized_graph_minimum_pairs"] = np.nan
        result["generalized_graph_alpha"] = 0.0
    overlay_settings = settings["asset_group_overlay"]
    group_by_indicator = {
        str(indicator): str(group)
        for group, indicators in downside_risk_settings[
            "indicator_groups"
        ].items()
        for indicator in indicators
    }
    result["asset_group"] = result["indicator_id"].astype(str).map(
        group_by_indicator
    ).fillna("ungrouped")
    group_strength = _build_causal_group_relative_strength(
        targets,
        downside_risk_settings["indicator_groups"],
        excluded_indicators,
        range(start, end + 1),
        trailing_months=int(overlay_settings["trailing_months"]),
        label_lag_months=int(
            overlay_settings["label_availability_lag_months"]
        ),
    )
    result = result.merge(
        group_strength,
        on=["origin_position", "asset_group"],
        how="left",
        validate="many_to_one",
    )
    base_probability = pd.to_numeric(
        result["p_up_generalized_calibrated"], errors="coerce"
    ).fillna(pd.to_numeric(result["p_up"], errors="coerce"))
    group_weight = (
        float(overlay_settings["weight"])
        if bool(overlay_settings["enabled"])
        else 0.0
    )
    adjusted_logit = _logit(base_probability) + group_weight * result[
        "asset_group_relative_logit"
    ].fillna(0.0)
    result["p_up_selection_score"] = 1.0 / (1.0 + np.exp(-adjusted_logit))
    result["asset_group_overlay_weight"] = group_weight
    overlay_settings = settings.get("selection_overlay", {})
    if bool(overlay_settings.get("enabled", False)):
        overlay_window = int(overlay_settings.get("recent_miss_window_months", 6))
        overlay_label_lag = int(overlay_settings.get("label_lag_months", 2))
        overlay_misses = build_overlay_recent_misses(
            targets,
            window_months=overlay_window,
            label_lag=overlay_label_lag,
        )
        targets_for_stability = targets.merge(
            result[["origin_position", "indicator_id", "asset_group"]],
            on=["origin_position", "indicator_id"],
            how="left",
            validate="many_to_one",
        )
        overlay_stability = build_overlay_group_stability(
            targets_for_stability,
            window_months=overlay_window,
            label_lag=overlay_label_lag,
        )
        result = apply_selection_overlay(
            result,
            overlay_misses,
            overlay_stability,
            base_threshold=float(overlay_settings.get("base_threshold", 0.45)),
            threshold_relax=float(overlay_settings.get("threshold_relax", 0.10)),
            history_full=int(overlay_settings.get("history_full", 4)),
            history_zero=int(overlay_settings.get("history_zero", 1)),
            stability_bonus=float(overlay_settings.get("stability_bonus", 0.30)),
            miss_penalty_strength=float(overlay_settings.get("miss_penalty_strength", 0.40)),
            stability_penalty_std=float(overlay_settings.get("stability_penalty_std", 0.5)),
        )
        overlay_cols = [
            "selection_overlay_recent_miss",
            "selection_overlay_recent_calls",
            "selection_overlay_miss_penalty",
            "selection_overlay_history_weight",
            "selection_overlay_stability_bonus",
            "selection_overlay_base_threshold",
            "selection_overlay_stability_bonus_strength",
        ]
        result = result.drop(
            columns=[
                "fit_through_origin_x",
                "fit_through_origin_y",
            ],
            errors="ignore",
        )
        result.attrs["selection_overlay_columns"] = overlay_cols
    result.attrs["frozen_base_config_hash"] = frozen_base_config_hash
    result.attrs["adaptive_excluded_indicators"] = sorted(excluded_indicators)
    return result


def _candidate_grid(settings: dict) -> list[dict[str, float | int]]:
    selection = settings["selection"]
    rows = []
    for threshold in selection["down_threshold_grid"]:
        for margin in selection["down_margin_grid"]:
            for trigger in selection["stress_trigger_grid"]:
                for share in selection["maximum_down_share_grid"]:
                    for regime_bonus in selection["regime_down_bonus_grid"]:
                        for shock_bonus in selection["shock_down_bonus_grid"]:
                            for replacement_margin in selection["replacement_margin_grid"]:
                                for maximum_replacements in selection["maximum_replacements_grid"]:
                                    rows.append({
                                        "down_threshold": float(threshold),
                                        "down_margin": float(margin),
                                        "stress_trigger": float(trigger),
                                        "maximum_down_share": float(share),
                                        "regime_down_bonus": float(regime_bonus),
                                        "shock_down_bonus": float(shock_bonus),
                                        "replacement_margin": float(replacement_margin),
                                        "maximum_replacements": int(maximum_replacements),
                                    })
    return rows


def _build_accuracy_first_candidate(
    inputs: pd.DataFrame,
    settings: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Choose accuracy-first coverage using development origins only."""
    accuracy_settings = settings["accuracy_first"]
    development_bounds = accuracy_settings["development_origins"]
    internal_windows = accuracy_settings["internal_windows"]
    target_accuracy = float(accuracy_settings["target_accuracy"])
    caps = [int(value) for value in accuracy_settings["cap_grid"]]
    if not caps:
        raise ValueError("Accuracy-first cap grid is empty")
    minimum_cap = int(settings["selection"]["minimum_selection_count"])
    if any(cap < minimum_cap for cap in caps):
        raise ValueError(
            "Accuracy-first caps must respect the configured monthly minimum"
        )
    rows = []
    for group_weight in accuracy_settings["group_weight_grid"]:
        ranked = apply_accuracy_first_selector(
            inputs,
            cap=max(caps),
            group_weight=float(group_weight),
        )
        for cap in caps:
            candidate = ranked[
                ranked["selection_rank"].le(cap)
            ].copy()
            development = _window_summary(candidate, development_bounds)
            internal = [
                _window_summary(candidate, bounds)
                for bounds in internal_windows
            ]
            row = {
                "group_weight": float(group_weight),
                "cap": cap,
                "development_calls": int(development["calls"]),
                "development_hits": int(development["hits"]),
                "development_accuracy": float(development["accuracy"]),
                "worst_internal_accuracy": float(
                    min(summary["accuracy"] for summary in internal)
                ),
                "all_internal_windows_meet_target": bool(
                    all(
                        summary["accuracy"] >= target_accuracy
                        for summary in internal
                    )
                ),
            }
            for index, (bounds, summary) in enumerate(
                zip(internal_windows, internal, strict=True), start=1
            ):
                row.update({
                    f"internal_{index}_start": int(bounds[0]),
                    f"internal_{index}_end": int(bounds[1]),
                    f"internal_{index}_calls": int(summary["calls"]),
                    f"internal_{index}_hits": int(summary["hits"]),
                    f"internal_{index}_accuracy": float(
                        summary["accuracy"]
                    ),
                })
            rows.append(row)
    search = pd.DataFrame(rows).sort_values(
        [
            "development_accuracy",
            "worst_internal_accuracy",
            "development_calls",
            "group_weight",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = search.iloc[0].to_dict()
    search["selected"] = False
    search.loc[0, "selected"] = True
    predictions = apply_accuracy_first_selector(
        inputs,
        cap=int(selected["cap"]),
        group_weight=float(selected["group_weight"]),
    )
    return predictions, search, selected


def _apply(
    inputs: pd.DataFrame,
    settings: dict,
    params: dict,
    cap: int | None,
    cap_schedule: dict[int, int] | pd.Series | None = None,
) -> pd.DataFrame:
    if cap_schedule is not None and cap is not None:
        raise ValueError("A cap schedule cannot be combined with a fixed cap")
    downside_columns = [
        "p_down",
        "p_down_global",
        "p_down_local",
        "p_down_pattern",
        "p_down_indicator_prior",
        "down_exhaustion_flag",
        "down_fit_through_origin",
    ]
    regime_columns = [
        "market_mean_return",
        "market_breadth",
        "market_breadth_3",
        "market_breadth_change_3",
        "market_dispersion",
        "peer_nonselected_count",
        "peer_nonselected_available_count",
        "peer_nonselected_breadth_up",
        "peer_nonselected_mean_return",
        "peer_nonselected_median_return",
        "peer_nonselected_dispersion",
        "peer_nonselected_negative_share",
        "peer_nonselected_weak_momentum_share",
        "peer_nonselected_mean_momentum_3",
        "peer_nonselected_exhaustion_share",
        "peer_nonselected_lead_negative_share",
        "previous_shock",
        "previous_shock_share",
        "market_mean_return_stress",
        "market_breadth_stress",
        "market_breadth_3_stress",
        "market_breadth_change_3_stress",
        "market_dispersion_stress",
        "peer_nonselected_breadth_up_stress",
        "peer_nonselected_mean_return_stress",
        "peer_nonselected_negative_share_stress",
        "peer_nonselected_weak_momentum_share_stress",
        "peer_nonselected_exhaustion_share_stress",
        "peer_nonselected_lead_negative_share_stress",
        "previous_shock_share_stress",
        "market_stress",
        "peer_stress",
        "shock_stress",
        "regime_stress",
        "regime_label",
        "forecast_market_breadth",
        "forecast_market_breadth_fit_through_origin",
        "forecast_market_breadth_observation_through_origin",
    ]
    base = inputs.drop(
        columns=[*downside_columns, *regime_columns],
        errors="ignore",
    )
    downside = inputs[["origin_position", "indicator_id", *downside_columns]]
    regime = inputs[["origin_position", *regime_columns]].drop_duplicates(
        "origin_position"
    )
    return apply_regime_adaptive_selector(
        base,
        downside,
        regime,
        cap=cap,
        minimum_cap=(
            int(settings["selection"]["minimum_selection_count"])
            if cap is None
            else None
        ),
        maximum_cap=(
            int(settings["selection"]["maximum_selection_count"])
            if cap is None
            else None
        ),
        cap_stress_low=float(settings["selection"]["cap_stress_low"]),
        cap_stress_high=float(settings["selection"]["cap_stress_high"]),
        stress_increases_cap=bool(settings["selection"]["stress_increases_cap"]),
        forward_breadth_threshold=(
            float(settings["forward_regime"]["expansion_threshold"])
            if bool(settings.get("forward_regime", {}).get("enabled", False))
            and settings.get("forward_regime", {}).get("cap_mode", "binary_15_or_20") == "binary_15_or_20"
            else None
        ),
        forward_breadth_low=(
            float(settings["forward_regime"]["graduated_low"])
            if bool(settings.get("forward_regime", {}).get("enabled", False))
            and settings.get("forward_regime", {}).get("cap_mode") == "graduated_15_to_20"
            else None
        ),
        forward_breadth_high=(
            float(settings["forward_regime"]["graduated_high"])
            if bool(settings.get("forward_regime", {}).get("enabled", False))
            and settings.get("forward_regime", {}).get("cap_mode") == "graduated_15_to_20"
            else None
        ),
        forward_breadth_cap_mode=(
            str(settings["forward_regime"]["cap_mode"])
            if bool(settings.get("forward_regime", {}).get("enabled", False))
            else None
        ),
        hard_down_threshold=float(settings["stress"]["hard_down_threshold"]),
        cap_schedule=cap_schedule,
        **params,
    )


def _selection_row(
    predictions: pd.DataFrame,
    base_predictions: pd.DataFrame,
    params: dict,
    settings: dict,
    cap: int | None,
) -> dict:
    tuning = predictions[
        predictions["origin_position"].between(
            int(settings["tuning_origins"][0]),
            int(settings["tuning_origins"][1]),
        )
    ]
    summary = summarize_regime_adaptive_predictions(tuning)
    minimum_calls = int(settings["selection"]["minimum_tuning_down_calls"])
    minimum_precision = float(settings["selection"]["minimum_tuning_down_precision"])
    qualifying = (
        summary["down_calls"] >= minimum_calls
        and (
            summary["down_precision"] >= minimum_precision
            if summary["down_calls"]
            else False
        )
    )
    internal_rows = []
    for index, bounds in enumerate(
        settings["selection"]["internal_tuning_windows"], start=1
    ):
        candidate_internal = _window_summary(predictions, bounds)
        base_internal = _window_summary(base_predictions, bounds)
        down_precision = candidate_internal["down_precision"]
        internal_qualifying = (
            candidate_internal["down_calls"]
            >= int(settings["selection"]["minimum_internal_down_calls"])
            and pd.notna(down_precision)
            and down_precision
            >= float(settings["selection"]["minimum_internal_down_precision"])
        )
        hit_delta = int(candidate_internal["hits"] - base_internal["hits"])
        if bool(
            settings["selection"]["require_nonnegative_internal_hit_delta"]
        ):
            internal_qualifying = internal_qualifying and hit_delta >= 0
        internal_rows.append({
            f"internal_{index}_start": int(bounds[0]),
            f"internal_{index}_end": int(bounds[1]),
            f"internal_{index}_hit_delta": hit_delta,
            f"internal_{index}_down_calls": int(
                candidate_internal["down_calls"]
            ),
            f"internal_{index}_down_precision": float(down_precision),
            f"internal_{index}_qualifying": bool(internal_qualifying),
        })
    internal_metrics = {
        key: value for row in internal_rows for key, value in row.items()
    }
    stable_qualifying = bool(
        qualifying
        and internal_rows
        and all(
            row[f"internal_{index}_qualifying"]
            for index, row in enumerate(internal_rows, start=1)
        )
    )
    internal_hit_deltas = [
        row[f"internal_{index}_hit_delta"]
        for index, row in enumerate(internal_rows, start=1)
    ]
    validation_summary = _window_summary(
        predictions, settings["validation_origins"]
    )
    validation_base_summary = _window_summary(
        base_predictions, settings["validation_origins"]
    )
    validation_hit_delta = int(
        validation_summary["hits"] - validation_base_summary["hits"]
    )
    development_qualifying = bool(
        stable_qualifying
        and (
            validation_hit_delta >= 0
            if bool(
                settings["selection"][
                    "require_nonnegative_validation_hit_delta"
                ]
            )
            else True
        )
    )
    return {
        **params,
        "cap": "dynamic" if cap is None else int(cap),
        "qualifying": bool(qualifying),
        "stable_qualifying": stable_qualifying,
        "development_qualifying": development_qualifying,
        "worst_internal_hit_delta": min(internal_hit_deltas),
        "validation_hit_delta": validation_hit_delta,
        "validation_accuracy": float(validation_summary["accuracy"]),
        "validation_down_calls": int(validation_summary["down_calls"]),
        "validation_down_precision": float(
            validation_summary["down_precision"]
        ),
        "tuning_accuracy": float(summary["accuracy"]),
        "tuning_down_calls": int(summary["down_calls"]),
        "tuning_down_precision": float(summary["down_precision"]),
        "tuning_hits": int(summary["hits"]),
        **internal_metrics,
    }


def _window_summary(predictions: pd.DataFrame, bounds: list[int]) -> dict:
    start, end = (int(value) for value in bounds)
    return summarize_regime_adaptive_predictions(
        predictions[predictions["origin_position"].between(start, end)]
    )


def _window_bootstrap_summary(
    predictions: pd.DataFrame,
    bounds: list[int],
    project_settings: dict,
) -> dict:
    start, end = (int(value) for value in bounds)
    return summarize_selected_predictions(
        predictions[
            predictions["origin_position"].between(start, end)
        ],
        block_months=int(project_settings["bootstrap_blocks"]),
        bootstrap_replicates=int(
            project_settings["bootstrap_replicates"]
        ),
        seed=int(project_settings["seed"]),
    )


def build_regime_adaptive_selector(
    root: Path = ROOT,
    cap: int | None = None,
) -> Path:
    settings = _read_yaml(root / "configs/regime_adaptive_selector.yaml")
    minimum_cap = int(settings["selection"]["minimum_selection_count"])
    maximum_cap = int(settings["selection"]["maximum_selection_count"])
    if cap is not None and not minimum_cap <= int(cap) <= maximum_cap:
        raise ValueError(
            f"cap must be between {minimum_cap} and {maximum_cap}"
        )
    dynamic_cap = cap is None and bool(settings["selection"]["dynamic_cap_enabled"])
    effective_cap = None if dynamic_cap else int(
        cap if cap is not None else settings["selection"]["monthly_selection_count"]
    )
    inputs = _build_inputs(root, settings, effective_cap or int(
        settings["selection"]["maximum_selection_count"]
    ))
    regime_frame_columns = ["origin_position", "regime_stress"]
    if "forecast_market_breadth" in inputs:
        regime_frame_columns.append("forecast_market_breadth")
    regime_frame = inputs[regime_frame_columns].drop_duplicates(
        "origin_position"
    )
    forward_cap_mode = str(settings.get("forward_regime", {}).get("cap_mode", "binary_15_or_20")) if bool(settings.get("forward_regime", {}).get("enabled", False)) else None
    base_comparison = apply_dynamic_base_cap(
        inputs.drop(columns=["regime_stress"], errors="ignore"),
        regime_frame,
        int(settings["selection"]["minimum_selection_count"])
        if dynamic_cap else int(effective_cap),
        int(settings["selection"]["maximum_selection_count"])
        if dynamic_cap else int(effective_cap),
        float(settings["selection"]["cap_stress_low"]),
        float(settings["selection"]["cap_stress_high"]),
        bool(settings["selection"]["stress_increases_cap"]),
        (
            float(settings["forward_regime"]["expansion_threshold"])
            if forward_cap_mode == "binary_15_or_20"
            else None
        ),
        (
            float(settings["forward_regime"]["graduated_low"])
            if forward_cap_mode == "graduated_15_to_20"
            else None
        ),
        (
            float(settings["forward_regime"]["graduated_high"])
            if forward_cap_mode == "graduated_15_to_20"
            else None
        ),
        forward_cap_mode,
    )
    candidate_rows = []
    for params in _candidate_grid(settings):
        predictions = _apply(inputs, settings, params, effective_cap)
        row = _selection_row(
            predictions,
            base_comparison,
            params,
            settings,
            effective_cap,
        )
        candidate_rows.append(row)
    search = pd.DataFrame(candidate_rows).sort_values(
        [
            "development_qualifying",
            "stable_qualifying",
            "qualifying",
            "worst_internal_hit_delta",
            "tuning_accuracy",
            "tuning_down_precision",
            "tuning_down_calls",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    if search["development_qualifying"].any():
        selected_row = search[search["development_qualifying"]].iloc[0].to_dict()
        selection_mode = "temporally_stable_development_candidate"
        allow_down_predictions = True
    else:
        fallback = settings["selection"]["fallback_parameters"]
        fallback_mask = pd.Series(True, index=search.index)
        for key, value in fallback.items():
            fallback_mask &= pd.to_numeric(
                search[key], errors="coerce"
            ).eq(float(value))
        if not fallback_mask.any():
            raise ValueError("Configured fallback parameters are not in the candidate grid")
        selected_row = search[fallback_mask].iloc[0].to_dict()
        allow_down_predictions = bool(
            settings["selection"].get(
                "fallback_allow_down_predictions", False
            )
        )
        selection_mode = (
            "guarded_bidirectional_fallback_no_stable_candidate"
            if allow_down_predictions
            else "conservative_fallback_no_stable_candidate"
        )
    selected_params = {
        key: float(selected_row[key])
        for key in [
            "down_threshold",
            "down_margin",
            "stress_trigger",
            "maximum_down_share",
            "regime_down_bonus",
            "shock_down_bonus",
            "replacement_margin",
        ]
    }
    selected_params["maximum_replacements"] = int(selected_row["maximum_replacements"])
    search["selected"] = False
    selected_mask = pd.Series(True, index=search.index)
    for key, value in selected_params.items():
        selected_mask &= pd.to_numeric(search[key], errors="coerce").eq(
            float(value)
        )
    search.loc[selected_mask, "selected"] = True
    search["selection_mode"] = np.where(
        search["selected"], selection_mode, "not_selected"
    )
    selected_params["allow_down_predictions"] = allow_down_predictions
    predictions = _apply(inputs, settings, selected_params, effective_cap)
    accuracy_first_predictions, accuracy_first_search, accuracy_first_row = (
        _build_accuracy_first_candidate(inputs, settings)
    )
    validate_oof_columns(predictions.columns.tolist())
    validate_oof_columns(accuracy_first_predictions.columns.tolist())
    if (
        predictions["generalized_graph_fit_through_origin"]
        > predictions["origin_position"] - 1
    ).any():
        raise AssertionError("Generalized correlation graph used future rows")
    project_settings = _read_yaml(root / "configs/config.yaml")
    data_hash = sha256_file(root / project_settings["data_path"])
    config_hash = _configuration_hash(root, effective_cap)
    frozen_base_config_hash = str(
        inputs.attrs.get("frozen_base_config_hash", "unknown")
    )
    predictions["run_id"] = "regime_adaptive_selector_research"
    predictions["model_id"] = settings["experiment_id"]
    predictions["model_version"] = settings["experiment_release"]
    predictions["data_hash"] = data_hash
    predictions["config_hash"] = config_hash
    predictions["locked_evaluation_read"] = False
    predictions["regime_selected_on"] = (
        "down_gate:tuning_120_179_internal_stability;"
        "group_and_correlation_overlays:"
        "tuning_120_179_and_validation_180_219_development"
    )
    predictions["regime_locked_origins"] = "268_315_not_read"
    predictions = apply_correctness_semantics(predictions)
    accuracy_first_predictions["run_id"] = (
        "regime_adaptive_accuracy_first_research"
    )
    accuracy_first_predictions["model_id"] = (
        f"{settings['experiment_id']}_accuracy_first"
    )
    accuracy_first_predictions["model_version"] = settings[
        "accuracy_first"
    ]["experiment_release"]
    accuracy_first_predictions["data_hash"] = data_hash
    accuracy_first_predictions["config_hash"] = config_hash
    accuracy_first_predictions["locked_evaluation_read"] = False
    accuracy_first_predictions["accuracy_first_selected_on"] = (
        "development_120_219_without_confirmation_or_locked_evaluation"
    )
    accuracy_first_predictions["regime_locked_origins"] = "268_315_not_read"
    accuracy_first_predictions = apply_correctness_semantics(
        accuracy_first_predictions
    )
    correctness_ready = predictions[
        predictions["correctness_fit_through_origin"].notna()
    ]
    if (
        correctness_ready["correctness_fit_through_origin"]
        > correctness_ready["origin_position"] - 2
    ).any():
        raise AssertionError("Correctness monitoring used unavailable outcomes")
    accuracy_first_correctness_ready = accuracy_first_predictions[
        accuracy_first_predictions["correctness_fit_through_origin"].notna()
    ]
    if (
        accuracy_first_correctness_ready["correctness_fit_through_origin"]
        > accuracy_first_correctness_ready["origin_position"] - 2
    ).any():
        raise AssertionError(
            "Accuracy-first correctness monitoring used unavailable outcomes"
        )

    base_summary = {
        "tuning": _window_summary(base_comparison, settings["tuning_origins"]),
        "validation": _window_summary(base_comparison, settings["validation_origins"]),
        "confirmation": _window_summary(base_comparison, settings["confirmation_origins"]),
    }
    candidate_summary = {
        "tuning": _window_summary(predictions, settings["tuning_origins"]),
        "validation": _window_summary(predictions, settings["validation_origins"]),
        "confirmation": _window_summary(predictions, settings["confirmation_origins"]),
    }
    accuracy_first_internal = [
        _window_summary(accuracy_first_predictions, bounds)
        for bounds in settings["accuracy_first"]["internal_windows"]
    ]
    accuracy_first_summary = {
        "enabled": bool(settings["accuracy_first"]["enabled"]),
        "selection_mode": "accuracy_first_fixed_coverage_up_only",
        "selected_on": "development_120_219_only",
        "confirmation_used_for_selection": False,
        "selected_cap": int(accuracy_first_row["cap"]),
        "selected_group_weight": float(
            accuracy_first_row["group_weight"]
        ),
        "target_accuracy": float(
            settings["accuracy_first"]["target_accuracy"]
        ),
        "coverage_vs_fixed_top15": float(
            int(accuracy_first_row["cap"])
            / int(settings["selection"]["monthly_selection_count"])
        ),
        "development": _window_summary(
            accuracy_first_predictions,
            settings["accuracy_first"]["development_origins"],
        ),
        "internal_windows": [
            {
                "origins": [int(bounds[0]), int(bounds[1])],
                **window_summary,
            }
            for bounds, window_summary in zip(
                settings["accuracy_first"]["internal_windows"],
                accuracy_first_internal,
                strict=True,
            )
        ],
        "all_internal_windows_meet_target": bool(
            accuracy_first_row["all_internal_windows_meet_target"]
        ),
        "confirmation": _window_summary(
            accuracy_first_predictions, settings["confirmation_origins"]
        ),
        "all_nonlocked": _window_summary(
            accuracy_first_predictions,
            [
                int(settings["tuning_origins"][0]),
                int(settings["confirmation_origins"][1]),
            ],
        ),
        "temporal_block_bootstrap": {
            "block_months": int(project_settings["bootstrap_blocks"]),
            "replicates": int(project_settings["bootstrap_replicates"]),
            "development": _window_bootstrap_summary(
                accuracy_first_predictions,
                settings["accuracy_first"]["development_origins"],
                project_settings,
            ),
            "confirmation": _window_bootstrap_summary(
                accuracy_first_predictions,
                settings["confirmation_origins"],
                project_settings,
            ),
            "all_nonlocked": _window_bootstrap_summary(
                accuracy_first_predictions,
                [
                    int(settings["tuning_origins"][0]),
                    int(settings["confirmation_origins"][1]),
                ],
                project_settings,
            ),
        },
        "active_model_changed": False,
        "promotion_eligible": False,
        "locked_evaluation_read": False,
    }
    selected_monthly_caps = predictions[
        predictions["accepted"].fillna(False).astype(bool)
    ].groupby("origin_position").size()
    summary = {
        "experiment_id": settings["experiment_id"],
        "experiment_name": settings["experiment_name"],
        "experiment_release": settings["experiment_release"],
        "active_model_changed": False,
        "locked_evaluation_read": False,
        "locked_origins": settings["locked_origins"],
        "effective_cap": "dynamic" if dynamic_cap else effective_cap,
        "minimum_cap": int(settings["selection"]["minimum_selection_count"]),
        "maximum_cap": int(settings["selection"]["maximum_selection_count"]),
        "dynamic_cap": dynamic_cap,
        "monthly_cap_distribution": {
            str(int(monthly_cap)): int(count)
            for monthly_cap, count in selected_monthly_caps.value_counts(
                sort=False
            ).sort_index().items()
        },
        "average_monthly_cap": float(selected_monthly_caps.mean()),
        "stress_increases_cap": bool(settings["selection"]["stress_increases_cap"]),
        "selected_parameters": selected_params,
        "selection_mode": selection_mode,
        "selected_candidate_qualifying": bool(selected_row["qualifying"]),
        "selected_candidate_stable_qualifying": bool(
            selected_row["stable_qualifying"]
        ),
        "selected_candidate_development_qualifying": bool(
            selected_row["development_qualifying"]
        ),
        "internal_tuning_windows": settings["selection"][
            "internal_tuning_windows"
        ],
        "excluded_indicators": inputs.attrs.get(
            "adaptive_excluded_indicators", []
        ),
        "asset_group_overlay": settings["asset_group_overlay"],
        "generalized_correlation_overlay": settings[
            "generalized_correlation_overlay"
        ],
        "forward_regime": settings.get("forward_regime", {"enabled": False}),
        "parameters_selected_on": {
            "down_gate": "tuning_120_179_with_internal_stability_gate",
            "asset_group_overlay": settings["asset_group_overlay"]["selected_on"],
            "generalized_correlation_overlay": settings[
                "generalized_correlation_overlay"
            ]["selected_on"],
            "forward_regime": settings.get("forward_regime", {}).get(
                "selected_on", "disabled"
            ),
        },
        "base": base_summary,
        "candidate": candidate_summary,
        "accuracy_first_selective": accuracy_first_summary,
        "validation_accuracy_delta": candidate_summary["validation"]["accuracy"] - base_summary["validation"]["accuracy"],
        "confirmation_accuracy_delta": candidate_summary["confirmation"]["accuracy"] - base_summary["confirmation"]["accuracy"],
        "promotion_eligible": False,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "frozen_base_config_hash": frozen_base_config_hash,
    }
    summary["nonlocked_development_gate_passed"] = bool(
        candidate_summary["validation"]["accuracy"] > base_summary["validation"]["accuracy"]
        and bool(selected_row["qualifying"])
        and bool(selected_row["stable_qualifying"])
        and bool(selected_row["development_qualifying"])
    )
    summary["promotion_eligible"] = False
    summary["promotion_requires_locked_evaluation"] = True
    experiment_root = regime_adaptive_experiment_root(root)
    (experiment_root / "artifacts").mkdir(parents=True, exist_ok=True)
    (experiment_root / "metrics").mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(predictions, regime_adaptive_predictions_artifact(root))
    atomic_write_parquet(
        accuracy_first_predictions,
        regime_adaptive_accuracy_first_artifact(root),
    )
    search.to_csv(experiment_root / "metrics/candidate_search.csv", index=False)
    accuracy_first_search.to_csv(
        experiment_root / "metrics/accuracy_first_search.csv", index=False
    )
    atomic_write_json(summary, regime_adaptive_summary_path(root))
    (experiment_root / "README.md").write_text(
        "\n".join([
            "# Regime Adaptive Bidirectional Selector",
            "",
            "This non-promoting experiment combines the frozen Uptrend Selector, the ordinary Downside probability research, causal market stress, known shock continuation, and the current non-selected peer set.",
            "",
            "## Behavior",
            "",
            f"- Effective monthly cap: `{'dynamic' if dynamic_cap else effective_cap}`.",
            (
                (
                    f"- Dynamic cap is graduated 15..20 linearly from breadth `{settings['forward_regime']['graduated_low']}` to `{settings['forward_regime']['graduated_high']}` (`{settings['selection']['minimum_selection_count']}` at low, `{settings['selection']['maximum_selection_count']}` at high); `--cap` overrides it with a fixed value."
                    if str(settings.get("forward_regime", {}).get("cap_mode")) == "graduated_15_to_20"
                    else f"- Dynamic cap is binary: `{settings['selection']['minimum_selection_count']}` normally and `{settings['selection']['maximum_selection_count']}` when the causal forward market-breadth forecast is at least `{settings['forward_regime']['expansion_threshold']}`; `--cap` overrides it with a fixed value."
                )
                if dynamic_cap
                else f"- The development result uses a fixed top `{effective_cap}` because stress-based expansion was not temporally stable."
            ),
            f"- Observed monthly cap distribution: `{summary['monthly_cap_distribution']}`; average `{summary['average_monthly_cap']:.4f}`.",
            (
                "- Regime stress activates guarded Down calls in the configured fallback when no policy clears the development gate."
                if selected_params["allow_down_predictions"]
                else "- Regime stress still informs Down evidence, but the fallback disables all Down calls when no policy clears the development gate."
            ),
            "- The full-panel replacement search can replace up to the selected maximum with non-selected indicators when their Down evidence clears the replacement margin.",
            f"- Indicators excluded by the downside data-quality gate are removed before selection: `{', '.join(summary['excluded_indicators']) or 'none'}`.",
            f"- Selection ranking adds a causal `{settings['asset_group_overlay']['trailing_months']}`-month asset-group relative-strength prior through `t-2`, with weight `{settings['asset_group_overlay']['weight']}`.",
            f"- The frozen correlation graph is generalized with a rolling `{settings['generalized_correlation_overlay']['window_months']}`-month signed graph over percentage returns through `t-1`, pair-reliability shrinkage, and alpha `{settings['generalized_correlation_overlay']['alpha']}`.",
            "- The non-selected indicators are summarized at each origin and are not treated as future information.",
            "- Each non-selected indicator also gets a causal `nonselected_warning_score` and explainable warning reasons in the prediction artifact.",
            "- Candidate selection requires adequate Down evidence, non-negative hit delta in both internal tuning windows 120-149 and 150-179, and no loss on the declared Validation development window.",
            (
                "- If no candidate passes that stability gate, the configured fallback keeps conservative Down calls enabled without forcing a Down quota or non-selected replacements."
                if selected_params["allow_down_predictions"]
                else "- If no candidate passes that stability gate, selection falls back to an explicit Up-only policy with no replacements."
            ),
            "- The group and generalized-correlation overlays are development-stage changes selected after reviewing Tuning and Validation; Confirmation remains descriptive and only locked origins can provide a clean future test.",
            "- Locked origins 268-315 were not read.",
            "",
            "## Result",
            "",
            f"- Selected parameters: `{json.dumps(selected_params, sort_keys=True)}`",
            f"- Selection mode: `{selection_mode}`",
            f"- Tuning candidate accuracy: `{candidate_summary['tuning']['hits']}/{candidate_summary['tuning']['calls']}` (`{candidate_summary['tuning']['accuracy']:.4%}`).",
            f"- Validation candidate accuracy: `{candidate_summary['validation']['hits']}/{candidate_summary['validation']['calls']}` (`{candidate_summary['validation']['accuracy']:.4%}`).",
            f"- Confirmation base accuracy: `{base_summary['confirmation']['accuracy']:.4%}`",
            f"- Confirmation candidate accuracy: `{candidate_summary['confirmation']['accuracy']:.4%}`",
            f"- Confirmation Down calls / precision: `{candidate_summary['confirmation']['down_calls']} / {'not applicable' if candidate_summary['confirmation']['down_calls'] == 0 else format(candidate_summary['confirmation']['down_precision'], '.4%')}`",
            f"- Promotion eligible: `{summary['promotion_eligible']}`.",
            "",
            "## Fixed-coverage accuracy alternative",
            "",
            "This separate policy searches causal group weights while enforcing the configured minimum of 15 Up-ranked indicators every month.",
            "",
            f"- Selected cap / group weight: `{accuracy_first_summary['selected_cap']} / {accuracy_first_summary['selected_group_weight']}`.",
            f"- Development accuracy: `{accuracy_first_summary['development']['hits']}/{accuracy_first_summary['development']['calls']}` (`{accuracy_first_summary['development']['accuracy']:.4%}`).",
            f"- Confirmation accuracy: `{accuracy_first_summary['confirmation']['hits']}/{accuracy_first_summary['confirmation']['calls']}` (`{accuracy_first_summary['confirmation']['accuracy']:.4%}`).",
            f"- Development temporal block-bootstrap P10: `{accuracy_first_summary['temporal_block_bootstrap']['development']['bootstrap_p10']:.4%}`.",
            f"- Coverage versus fixed Top-15: `{accuracy_first_summary['coverage_vs_fixed_top15']:.4%}`.",
            "- Confirmation and locked origins were not used to select its cap or group weight; the active model is unchanged.",
        ]) + "\n",
        encoding="utf-8",
    )
    return regime_adaptive_predictions_artifact(root)


def regime_adaptive_status(root: Path = ROOT) -> dict:
    path = regime_adaptive_summary_path(root)
    if not path.exists():
        raise FileNotFoundError("Regime-adaptive summary has not been built")
    return json.loads(path.read_text(encoding="utf-8"))
