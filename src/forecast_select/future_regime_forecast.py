from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .directional_downside import (
    build_directional_downside_features,
    fit_directional_downside_model,
    predict_directional_downside,
)
from .directional_downside_pipeline import (
    _configuration_hash as _directional_configuration_hash,
)
from .future_forecast import (
    _forecast_month,
    build_direct_horizon_targets,
    build_direct_monthly_forecasts,
)
from .forward_regime import build_forward_market_breadth_forecast
from .io import atomic_write_json, load_workbook, sha256_file
from .regime_adaptive import (
    build_nonselected_indicator_warnings,
    build_nonselected_peer_features,
    build_regime_features,
)
from .regime_adaptive_pipeline import (
    _apply,
    _build_causal_group_relative_strength,
    _build_generalized_correlation_overlay,
    _configuration_hash,
    _shock_features,
    regime_adaptive_predictions_artifact,
    regime_adaptive_summary_path,
)
from .targets import build_targets
from .uptrend_pipeline import ROOT
from .validation import causal_training_rows


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen research result is missing: {path}. "
            "Run build-directional-downside and build-regime-adaptive first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _observed_values(frame: pd.DataFrame) -> pd.DataFrame:
    indicators = [column for column in frame.columns if column.startswith("X")]
    return frame[["position", *indicators]].melt(
        id_vars="position",
        value_vars=indicators,
        var_name="indicator_id",
        value_name="value_t",
    ).rename(columns={"position": "origin_position"})


def _build_direct_downside_probabilities(
    root: Path,
    frame: pd.DataFrame,
    origin: int,
    horizons: tuple[int, ...],
    base_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _read_yaml(root / "configs/config.yaml")
    settings = _read_yaml(root / "configs/directional_downside_model.yaml")
    feature_settings = settings["features"]
    model_settings = settings["model"]
    availability_lag = int(settings["availability_lag_months"])
    features = build_directional_downside_features(
        frame,
        availability_lag=availability_lag,
        lead_correlation_window=int(feature_settings["lead_correlation_window"]),
        lead_minimum_pairs=int(feature_settings["lead_minimum_pairs"]),
        lead_top_k=int(feature_settings["lead_top_k"]),
    )
    observed = _observed_values(frame)
    directional_summary = _read_json(
        root / "research/directional_downside_selector/metrics/summary.json"
    )
    expected_hash = sha256_file(root / config["data_path"])
    if str(directional_summary.get("data_hash")) != expected_hash:
        raise ValueError("Frozen Down parameters do not match the current workbook")
    if str(directional_summary.get("config_hash")) != (
        _directional_configuration_hash(root)
    ):
        raise ValueError("Frozen Down parameters do not match current configs")
    selected_parameters = directional_summary["selected_parameters"]
    local_weight = float(selected_parameters["local_weight"])
    pattern_weight = float(selected_parameters["pattern_weight"])
    global_weight = 1.0 - local_weight - pattern_weight

    probability_parts = []
    for horizon in sorted(horizons):
        targets = build_direct_horizon_targets(frame, horizon)
        panel = features.merge(
            targets[[
                "origin_position",
                "indicator_id",
                "y_true",
                "target_available",
            ]],
            on=["origin_position", "indicator_id"],
            how="left",
            validate="one_to_one",
        ).merge(
            observed,
            on=["origin_position", "indicator_id"],
            how="left",
            validate="one_to_one",
        )
        panel["down_target"] = 1.0 - panel["y_true"]
        train_eligible = panel[
            panel["target_available"].fillna(False).astype(bool)
            & panel["value_t"].notna()
            & panel["origin_position"].gt(int(config["minimum_history_months"]))
            & panel["down_target"].notna()
        ].copy()
        effective_lag = availability_lag + horizon - 1
        train = causal_training_rows(
            train_eligible,
            origin,
            availability_lag=effective_lag,
        )
        current_indicators = set(
            base_predictions.loc[
                base_predictions["forecast_horizon_months"].eq(horizon),
                "indicator_id",
            ].astype(str)
        )
        test = panel[
            panel["origin_position"].eq(origin)
            & panel["indicator_id"].astype(str).isin(current_indicators)
        ].copy()
        if test.empty:
            raise ValueError(f"No downside candidates are available for horizon {horizon}")
        model = fit_directional_downside_model(
            train,
            seed=int(config["seed"]),
            global_logistic_c=float(model_settings["global_logistic_c"]),
            local_logistic_c=float(model_settings["local_logistic_c"]),
            max_iter=int(model_settings["logistic_max_iter"]),
            minimum_local_rows=int(model_settings["minimum_local_rows"]),
            minimum_local_class_rows=int(
                model_settings["minimum_local_class_rows"]
            ),
        )
        probabilities = predict_directional_downside(
            model,
            train,
            test,
            trailing_prior_window=int(model_settings["trailing_prior_window"]),
            minimum_pattern_rows=int(model_settings["minimum_pattern_rows"]),
        )
        probabilities["p_down"] = (
            global_weight * probabilities["p_down_global"]
            + local_weight * probabilities["p_down_local"]
            + pattern_weight * probabilities["p_down_pattern"]
        ).clip(1e-6, 1.0 - 1e-6)
        probabilities["down_exhaustion_flag"] = test[
            "down_exhaustion_flag"
        ].to_numpy()
        probabilities["down_fit_through_origin"] = origin - effective_lag - 1
        probabilities["forecast_horizon_months"] = horizon
        probability_parts.append(probabilities)
    return pd.concat(probability_parts, ignore_index=True), features


def _current_regime_features(
    root: Path,
    frame: pd.DataFrame,
    origin: int,
    base_predictions: pd.DataFrame,
    current_directional_panel: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    history_path = regime_adaptive_predictions_artifact(root)
    if not history_path.exists():
        raise FileNotFoundError(
            "Frozen regime-adaptive predictions are missing; "
            "run build-regime-adaptive first"
        )
    history = pd.read_parquet(history_path)
    config = _read_yaml(root / "configs/config.yaml")
    expected_hash = sha256_file(root / config["data_path"])
    if set(history["data_hash"].dropna().astype(str).unique()) != {expected_hash}:
        raise ValueError("Frozen regime history does not match the current workbook")
    summary = _read_json(regime_adaptive_summary_path(root))
    if set(history["config_hash"].dropna().astype(str).unique()) != {
        str(summary["config_hash"])
    }:
        raise ValueError("Frozen regime history does not match the frozen summary")
    if history["locked_evaluation_read"].fillna(True).any():
        raise AssertionError("Frozen regime history includes locked evidence")
    history = history.sort_values("origin_position").drop_duplicates(
        "origin_position"
    )

    market_columns = [
        "market_mean_return",
        "market_breadth",
        "market_breadth_3",
        "market_breadth_change_3",
        "market_dispersion",
    ]
    market_history = history[["origin_position", *market_columns]].rename(
        columns={column: f"down_{column}" for column in market_columns}
    )
    current_market = current_directional_panel[[
        "origin_position",
        *[f"down_{column}" for column in market_columns],
    ]].drop_duplicates("origin_position")
    market = pd.concat([market_history, current_market], ignore_index=True)

    peer_columns = [
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
    ]
    peer_history = history[["origin_position", *peer_columns]]
    current_peer = build_nonselected_peer_features(
        current_directional_panel,
        base_predictions,
    )
    peer = pd.concat([peer_history, current_peer], ignore_index=True)

    shock_columns = ["previous_shock", "previous_shock_share"]
    shock_history = history[["origin_position", *shock_columns]]
    shock_settings = _read_yaml(root / "configs/downside_risk_gate.yaml")
    current_shock = _shock_features(build_targets(frame), shock_settings)
    current_shock = current_shock[
        current_shock["origin_position"].eq(origin)
    ][["origin_position", *shock_columns]]
    if current_shock.empty:
        raise ValueError("No causal shock features are available at forecast origin")
    shock = pd.concat([shock_history, current_shock], ignore_index=True)

    stress = settings["stress"]
    regime = build_regime_features(
        market,
        peer,
        shock,
        market_weight=float(stress["market_weight"]),
        peer_weight=float(stress["peer_weight"]),
        shock_weight=float(stress["shock_weight"]),
    )
    return regime[regime["origin_position"].eq(origin)].copy()


def _build_adaptive_inputs(
    root: Path,
    frame: pd.DataFrame,
    origin: int,
    base: pd.DataFrame,
    downside: pd.DataFrame,
    directional_features: pd.DataFrame,
    regime: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    current_panel = directional_features[
        directional_features["origin_position"].eq(origin)
    ].copy()
    warnings = build_nonselected_indicator_warnings(current_panel, base)
    inputs = base.merge(
        downside,
        on=[
            "origin_position",
            "indicator_id",
            "forecast_horizon_months",
        ],
        how="left",
        validate="one_to_one",
    ).merge(
        current_panel[[
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

    risk_settings = _read_yaml(root / "configs/downside_risk_gate.yaml")
    excluded = {
        str(value) for value in risk_settings.get("excluded_indicators", [])
    }
    inputs["adaptive_data_quality_excluded"] = inputs[
        "indicator_id"
    ].astype(str).isin(excluded)
    inputs["adaptive_exclusion_reason"] = np.where(
        inputs["adaptive_data_quality_excluded"],
        "excluded_by_downside_data_quality_gate",
        "",
    )
    inputs["level_c_ready_before_adaptive_exclusion"] = inputs[
        "level_c_ready"
    ].fillna(False).astype(bool)
    inputs["level_c_ready"] = (
        inputs["level_c_ready_before_adaptive_exclusion"]
        & ~inputs["adaptive_data_quality_excluded"]
    )

    correlation_settings = settings["generalized_correlation_overlay"]
    if bool(correlation_settings["enabled"]):
        uptrend_settings = _read_yaml(root / "configs/uptrend_model.yaml")
        correlation_overlay = _build_generalized_correlation_overlay(
            frame,
            inputs,
            correlation_settings,
            excluded,
            indicator_prior_weight=float(
                uptrend_settings["selection"]["indicator_prior_weight"]
            ),
        )
        inputs = inputs.merge(
            correlation_overlay,
            on=["origin_position", "indicator_id"],
            how="left",
            validate="one_to_one",
        )
    else:
        inputs["p_up_generalized_graph"] = inputs["p_up"]
        inputs["p_up_generalized_calibrated"] = inputs["p_up_calibrated"]
        inputs["generalized_graph_fit_through_origin"] = np.nan
        inputs["generalized_graph_window_months"] = np.nan
        inputs["generalized_graph_minimum_pairs"] = np.nan
        inputs["generalized_graph_alpha"] = 0.0

    group_by_indicator = {
        str(indicator): str(group)
        for group, indicators in risk_settings["indicator_groups"].items()
        for indicator in indicators
    }
    inputs["asset_group"] = inputs["indicator_id"].astype(str).map(
        group_by_indicator
    ).fillna("ungrouped")
    overlay = settings["asset_group_overlay"]
    group_strength = _build_causal_group_relative_strength(
        build_targets(frame),
        risk_settings["indicator_groups"],
        excluded,
        range(origin, origin + 1),
        trailing_months=int(overlay["trailing_months"]),
        label_lag_months=int(overlay["label_availability_lag_months"]),
    )
    inputs = inputs.merge(
        group_strength,
        on=["origin_position", "asset_group"],
        how="left",
        validate="many_to_one",
    )
    p_up = pd.to_numeric(
        inputs["p_up_generalized_calibrated"], errors="coerce"
    ).fillna(
        pd.to_numeric(inputs["p_up"], errors="coerce")
    ).clip(1e-6, 1.0 - 1e-6)
    weight = float(overlay["weight"]) if bool(overlay["enabled"]) else 0.0
    adjusted_logit = np.log(p_up / (1.0 - p_up)) + weight * inputs[
        "asset_group_relative_logit"
    ].fillna(0.0)
    inputs["p_up_selection_score"] = 1.0 / (1.0 + np.exp(-adjusted_logit))
    inputs["asset_group_overlay_weight"] = weight
    return inputs


def build_regime_adaptive_monthly_forecasts(
    root: Path = ROOT,
    forecast_origin: int | None = None,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    config = _read_yaml(root / "configs/config.yaml")
    frame = load_workbook(root / config["data_path"])
    origin = int(
        frame["position"].max() if forecast_origin is None else forecast_origin
    )
    if origin not in set(frame["position"].astype(int)):
        raise ValueError("forecast origin is outside the supplied workbook")

    settings = _read_yaml(root / "configs/regime_adaptive_selector.yaml")
    summary = _read_json(regime_adaptive_summary_path(root))
    expected_hash = sha256_file(root / config["data_path"])
    if str(summary.get("data_hash")) != expected_hash:
        raise ValueError("Frozen adaptive parameters do not match the current workbook")
    configured_cap = (
        None if bool(summary["dynamic_cap"]) else int(summary["effective_cap"])
    )
    if str(summary.get("config_hash")) != _configuration_hash(
        root, configured_cap
    ):
        raise ValueError("Frozen adaptive parameters do not match current configs")
    base = build_direct_monthly_forecasts(
        root,
        forecast_origin=origin,
        horizons=horizons,
        include_rejected=True,
    )
    downside, directional_features = _build_direct_downside_probabilities(
        root,
        frame,
        origin,
        horizons,
        base,
    )
    selected_parameters = dict(summary["selected_parameters"])
    cap = configured_cap
    forward_regime = None
    forward_settings = settings.get("forward_regime", {})
    if cap is None and bool(forward_settings.get("enabled", False)):
        risk_settings = _read_yaml(
            root / "configs/downside_risk_gate.yaml"
        )
        model_settings = dict(forward_settings["model"])
        model_settings["seed"] = int(config["seed"])
        forward_regime = build_forward_market_breadth_forecast(
            frame,
            build_targets(frame),
            risk_settings["indicator_groups"],
            {
                str(value)
                for value in risk_settings.get("excluded_indicators", [])
            },
            [origin],
            availability_lag=int(settings["availability_lag_months"]),
            model_settings=model_settings,
        )
    forecast_parts = []
    for horizon in sorted(horizons):
        current_base = base[
            base["forecast_horizon_months"].eq(horizon)
        ].copy()
        current_downside = downside[
            downside["forecast_horizon_months"].eq(horizon)
        ].copy()
        current_panel = directional_features[
            directional_features["origin_position"].eq(origin)
        ].copy()
        regime = _current_regime_features(
            root,
            frame,
            origin,
            current_base,
            current_panel,
            settings,
        )
        if forward_regime is not None:
            regime = regime.merge(
                forward_regime,
                on="origin_position",
                how="left",
                validate="one_to_one",
            )
        inputs = _build_adaptive_inputs(
            root,
            frame,
            origin,
            current_base,
            current_downside,
            directional_features,
            regime,
            settings,
        )
        predictions = _apply(inputs, settings, selected_parameters, cap)
        selected = predictions[predictions["accepted"]].copy()
        expected_cap = int(selected["regime_cap"].iloc[0])
        if len(selected) != expected_cap or selected["indicator_id"].nunique() != expected_cap:
            raise AssertionError(
                f"Adaptive horizon {horizon} must select {expected_cap} unique indicators"
            )
        selected["forecast_horizon_months"] = horizon
        selected["forecast_month"] = _forecast_month(frame, origin, horizon)
        selected["model_scope"] = (
            "frozen_regime_adaptive_one_step"
            if horizon == 1
            else "experimental_direct_horizon_regime_adaptive"
        )
        availability_lag = int(settings["availability_lag_months"])
        selected["training_fit_through_origin"] = (
            origin - (availability_lag + horizon - 1) - 1
        )
        forecast_parts.append(selected)
    return pd.concat(forecast_parts, ignore_index=True).sort_values(
        ["forecast_horizon_months", "selection_rank"]
    ).reset_index(drop=True)


def regime_adaptive_next_three_artifact(root: Path = ROOT) -> Path:
    return root / "reports/regime_adaptive_next_three_forecast.json"


def write_regime_adaptive_next_three_forecast(root: Path = ROOT) -> Path:
    config = _read_yaml(root / "configs/config.yaml")
    settings = _read_yaml(root / "configs/regime_adaptive_selector.yaml")
    summary = _read_json(regime_adaptive_summary_path(root))
    frame = load_workbook(root / config["data_path"])
    origin = int(frame["position"].max())
    forecasts = build_regime_adaptive_monthly_forecasts(
        root,
        forecast_origin=origin,
        horizons=(1, 2, 3),
    )
    information_position = origin - int(settings["availability_lag_months"])
    information_date = frame.loc[
        frame["position"].eq(information_position), "Dates"
    ].iloc[0]
    history = pd.read_parquet(regime_adaptive_predictions_artifact(root))
    payload = {
        "generated_from_data_through": str(
            frame.loc[frame["position"].eq(origin), "Dates"].iloc[0].date()
        ),
        "feature_information_through": str(information_date.date()),
        "forecast_origin_position": origin,
        "availability_lag_months": int(settings["availability_lag_months"]),
        "data_hash": sha256_file(root / config["data_path"]),
        "method": "direct_multi_horizon_frozen_regime_adaptive",
        "selection_mode": str(summary["selection_mode"]),
        "selected_parameters": summary["selected_parameters"],
        "generalized_correlation_overlay": summary[
            "generalized_correlation_overlay"
        ],
        "historical_regime_reference_through_origin": int(
            history["origin_position"].max()
        ),
        "notes": [
            "Horizon one applies the frozen regime-adaptive research policy.",
            "Horizons two and three are experimental direct-horizon extensions.",
            "The causal one-step breadth forecast controls the 15/20 cap and is only a regime reference for experimental horizons two and three.",
            "Each horizon trains separate Up and Down models on causally available labels.",
            "No future indicator values or hidden outcomes were synthesized or read.",
            "The research artifacts and locked evaluation were not modified.",
        ],
        "forecasts": [],
    }
    for (horizon, month), group in forecasts.groupby(
        ["forecast_horizon_months", "forecast_month"], sort=True
    ):
        forecast_payload = {
            "forecast_month": str(month),
            "horizon_months": int(horizon),
            "model_scope": str(group["model_scope"].iloc[0]),
            "training_fit_through_origin": int(
                group["training_fit_through_origin"].iloc[0]
            ),
            "generalized_graph_fit_through_origin": int(
                group["generalized_graph_fit_through_origin"].iloc[0]
            ),
            "regime_label": str(group["regime_label"].iloc[0]),
            "regime_stress": round(float(group["regime_stress"].iloc[0]), 6),
            "selection_cap": int(group["regime_cap"].iloc[0]),
            "selections": [
                {
                    "rank": int(row.selection_rank),
                    "indicator_id": str(row.indicator_id),
                    "direction": str(row.predicted_direction),
                    "selection_score": round(float(row.selection_score), 6),
                    "p_up": round(float(row.p_up_base), 6),
                    "p_down": round(float(row.p_down), 6),
                    "asset_group": str(row.asset_group),
                }
                for row in group.sort_values("selection_rank").itertuples(
                    index=False
                )
            ],
        }
        if "forecast_market_breadth" in group:
            forecast_payload["forecast_market_breadth"] = round(
                float(group["forecast_market_breadth"].iloc[0]), 6
            )
            forecast_payload[
                "forward_regime_fit_through_origin"
            ] = int(
                group[
                    "forecast_market_breadth_fit_through_origin"
                ].iloc[0]
            )
        payload["forecasts"].append(forecast_payload)
    output = regime_adaptive_next_three_artifact(root)
    atomic_write_json(payload, output)
    return output
