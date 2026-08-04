from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .features import build_feature_panel
from .indicator_selection import (
    propagate_correlation_graph,
    select_top_indicators,
)
from .io import atomic_write_json, load_workbook, sha256_file
from .uptrend_model import fit_uptrend_model, predict_uptrend_probability
from .validation import causal_training_rows


ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_direct_horizon_targets(
    frame: pd.DataFrame,
    horizon_months: int,
) -> pd.DataFrame:
    """Build the monthly direction at t+h from features anchored at t.

    Horizon one predicts the direction from t to t+1. Horizon two predicts the
    direction from t+1 to t+2, so future intermediate values are never
    synthesized for a multi-month forecast.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be positive")
    indicators = [column for column in frame.columns if column.startswith("X")]
    rows = []
    for indicator in indicators:
        direction_start = frame[indicator].shift(-(horizon_months - 1))
        direction_end = frame[indicator].shift(-horizon_months)
        change = direction_end - direction_start
        rows.append(pd.DataFrame({
            "origin_position": frame["position"],
            "indicator_id": indicator,
            "direction_start_date": frame["Dates"].shift(
                -(horizon_months - 1)
            ),
            "target_date": frame["Dates"].shift(-horizon_months),
            "y_true": (change > 0).astype("float64").where(change.notna()),
            "target_available": direction_start.notna() & direction_end.notna(),
        }))
    return pd.concat(rows, ignore_index=True)


def _forecast_month(
    frame: pd.DataFrame,
    origin_position: int,
    horizon_months: int,
) -> str:
    origin_date = frame.loc[
        frame["position"].eq(origin_position),
        "Dates",
    ]
    if origin_date.empty:
        raise ValueError("forecast origin is outside the supplied workbook")
    return str(origin_date.iloc[0].to_period("M") + horizon_months)


def build_direct_monthly_forecasts(
    root: Path = ROOT,
    forecast_origin: int | None = None,
    horizons: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Select monthly directions for several direct forecast horizons.

    The registered Uptrend Selector is a one-step model. Horizons above one
    are explicit direct extensions that reuse its features, estimator, graph,
    and selection policy while training on horizon-specific historical labels.
    """
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")

    config = _read_yaml(root / "configs/config.yaml")
    settings = _read_yaml(root / "configs/uptrend_model.yaml")
    frame = load_workbook(root / config["data_path"])
    origin = int(
        frame["position"].max()
        if forecast_origin is None
        else forecast_origin
    )
    if origin not in set(frame["position"].astype(int)):
        raise ValueError("forecast origin is outside the supplied workbook")

    lag = int(settings["availability_lag_months"])
    features = build_feature_panel(
        frame,
        availability_lag=lag,
        include_structured=True,
    )
    graph_settings = settings["graph"]
    indicators = [column for column in frame.columns if column.startswith("X")]
    graph = frame[indicators].diff().iloc[
        :int(graph_settings["estimation_end"])
    ].corr(min_periods=int(graph_settings["minimum_pairs"]))
    np.fill_diagonal(graph.values, 0.0)

    model_settings = settings["model"]
    selection_settings = settings["selection"]
    forecast_parts = []
    for horizon in sorted(horizons):
        targets = build_direct_horizon_targets(frame, horizon)
        panel = features.merge(
            targets,
            on=["origin_position", "indicator_id"],
            how="left",
            validate="one_to_one",
        )
        train_eligible = panel[
            panel["target_available"]
            & panel["observed"].eq(1)
            & panel["origin_position"].gt(
                int(config["minimum_history_months"])
            )
        ].copy()
        effective_lag = lag + horizon - 1
        train = causal_training_rows(
            train_eligible,
            origin,
            availability_lag=effective_lag,
        )
        test = panel[
            panel["origin_position"].eq(origin)
            & panel["observed"].eq(1)
        ].copy()
        if test.empty:
            raise ValueError("no indicators are available at the forecast origin")

        model = fit_uptrend_model(
            train,
            seed=int(config["seed"]),
            logistic_c=float(model_settings["logistic_c"]),
            max_iter=int(model_settings["logistic_max_iter"]),
        )
        raw_probability = predict_uptrend_probability(model, test)
        correlation = graph.reindex(
            index=test["indicator_id"],
            columns=test["indicator_id"],
        ).fillna(0.0).to_numpy(dtype=float)
        graph_probability = propagate_correlation_graph(
            raw_probability,
            correlation,
            alpha=float(graph_settings["alpha"]),
        )
        predictions = test[[
            "origin_position",
            "origin_date",
            "target_date",
            "indicator_id",
            "y_true",
        ]].copy()
        predictions["p_up_raw"] = raw_probability
        predictions["p_up"] = graph_probability
        predictions["predicted_direction"] = np.where(
            graph_probability >= 0.5,
            "Up",
            "Down",
        )

        target_history = targets.loc[
            targets["target_available"] & targets["y_true"].notna(),
            ["origin_position", "indicator_id", "y_true"],
        ]
        selected = select_top_indicators(
            target_history,
            predictions,
            cap=int(selection_settings["monthly_selection_count"]),
            prior_window=int(selection_settings["trailing_target_window"]),
            prior_weight=float(selection_settings["indicator_prior_weight"]),
            minimum_history_months=int(
                selection_settings["minimum_history_months"]
            ),
            minimum_indicator_history=int(
                selection_settings["minimum_indicator_history"]
            ),
            availability_lag=effective_lag,
        )
        selected = selected[selected["accepted"]].copy()
        expected = int(selection_settings["monthly_selection_count"])
        if (
            len(selected) != expected
            or selected["indicator_id"].nunique() != expected
        ):
            raise AssertionError(
                f"Forecast horizon {horizon} must select {expected} unique indicators"
            )
        selected["forecast_horizon_months"] = horizon
        selected["forecast_month"] = _forecast_month(frame, origin, horizon)
        selected["model_scope"] = (
            "registered_one_step"
            if horizon == 1
            else "experimental_direct_horizon"
        )
        selected["training_fit_through_origin"] = origin - effective_lag - 1
        forecast_parts.append(selected)

    return pd.concat(forecast_parts, ignore_index=True).sort_values(
        ["forecast_horizon_months", "selection_rank"]
    ).reset_index(drop=True)


def next_three_forecast_artifact(root: Path = ROOT) -> Path:
    return root / "reports/next_three_month_forecast.json"


def write_next_three_forecast(root: Path = ROOT) -> Path:
    config = _read_yaml(root / "configs/config.yaml")
    settings = _read_yaml(root / "configs/uptrend_model.yaml")
    frame = load_workbook(root / config["data_path"])
    origin = int(frame["position"].max())
    forecasts = build_direct_monthly_forecasts(
        root,
        forecast_origin=origin,
        horizons=(1, 2, 3),
    )
    lag = int(settings["availability_lag_months"])
    information_position = origin - lag
    information_date = frame.loc[
        frame["position"].eq(information_position),
        "Dates",
    ].iloc[0]
    payload = {
        "generated_from_data_through": str(
            frame.loc[frame["position"].eq(origin), "Dates"].iloc[0].date()
        ),
        "feature_information_through": str(information_date.date()),
        "forecast_origin_position": origin,
        "availability_lag_months": lag,
        "data_hash": sha256_file(root / config["data_path"]),
        "method": "direct_multi_horizon_uptrend_selector",
        "notes": [
            "The first month is the registered one-step model scope.",
            "Months two and three are experimental direct-horizon extensions.",
            "No hidden outcomes or synthesized future indicator values were used.",
        ],
        "forecasts": [],
    }
    for (horizon, month), group in forecasts.groupby(
        ["forecast_horizon_months", "forecast_month"],
        sort=True,
    ):
        payload["forecasts"].append({
            "forecast_month": month,
            "horizon_months": int(horizon),
            "model_scope": str(group["model_scope"].iloc[0]),
            "training_fit_through_origin": int(
                group["training_fit_through_origin"].iloc[0]
            ),
            "selections": [
                {
                    "rank": int(row.selection_rank),
                    "indicator_id": str(row.indicator_id),
                    "direction": str(row.predicted_direction),
                    "selection_score": round(float(row.selection_score), 6),
                }
                for row in group.sort_values("selection_rank").itertuples(
                    index=False
                )
            ],
        })
    output = next_three_forecast_artifact(root)
    atomic_write_json(payload, output)
    return output
