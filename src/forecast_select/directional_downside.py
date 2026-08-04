from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DIRECTIONAL_DOWNSIDE_FEATURES = [
    "down_return_1",
    "down_return_lag_1",
    "down_return_lag_2",
    "down_momentum_3",
    "down_momentum_6",
    "down_momentum_12",
    "down_volatility_3",
    "down_volatility_12",
    "down_drawdown_12",
    "down_distance_mean_12",
    "down_negative_share_3",
    "down_rise_6_before_2",
    "down_stall_2",
    "down_rise_score",
    "down_stall_score",
    "down_exhaustion_score",
    "down_near_high_12",
    "down_deceleration_2_vs_4",
    "down_volatility_compression",
    "down_market_mean_return",
    "down_market_breadth",
    "down_market_breadth_3",
    "down_market_breadth_change_3",
    "down_market_dispersion",
    "down_lead_peer_score",
    "down_lead_negative_consensus",
    "down_lead_abs_correlation",
    "down_lead_peer_count",
]


def _lead_peer_features(
    frame: pd.DataFrame,
    availability_lag: int,
    window: int,
    minimum_pairs: int,
    top_k: int,
) -> pd.DataFrame:
    """Estimate which peers lead each indicator by two return periods.

    The two-period alignment matches the project's one-month availability lag:
    at origin t the newest observable return ends at t-1, while the target
    return ends at t+1.
    """
    indicators = [column for column in frame.columns if column.startswith("X")]
    raw_returns = frame[indicators].pct_change(fill_method=None)
    rows: list[dict[str, float | int | str]] = []
    for row_index, origin in enumerate(frame["position"]):
        latest = row_index - availability_lag
        defaults = {
            "down_lead_peer_score": np.nan,
            "down_lead_negative_consensus": np.nan,
            "down_lead_abs_correlation": np.nan,
            "down_lead_peer_count": 0.0,
        }
        if latest < minimum_pairs + 2:
            for indicator in indicators:
                rows.append({
                    "origin_position": int(origin),
                    "indicator_id": indicator,
                    **defaults,
                })
            continue

        predictor_end = latest - 1
        predictor_start = max(1, predictor_end - window)
        predictors = raw_returns.iloc[predictor_start:predictor_end]
        responses = raw_returns.shift(-2).iloc[predictor_start:predictor_end]
        predictor_names = [f"predictor_{indicator}" for indicator in indicators]
        response_names = [f"response_{indicator}" for indicator in indicators]
        combined = pd.concat([
            predictors.set_axis(predictor_names, axis=1),
            responses.set_axis(response_names, axis=1),
        ], axis=1)
        lead_correlation = combined.corr(min_periods=minimum_pairs).loc[
            predictor_names,
            response_names,
        ]
        current = raw_returns.iloc[latest]
        scale = predictors.std(ddof=0).replace(0, np.nan)
        current_z = current.div(scale).clip(-5.0, 5.0)
        for indicator in indicators:
            correlations = lead_correlation[
                f"response_{indicator}"
            ].copy()
            correlations.index = indicators
            correlations = correlations.drop(
                labels=indicator,
                errors="ignore",
            ).dropna()
            pairs = pd.DataFrame({
                "correlation": correlations,
                "current_z": current_z.reindex(correlations.index),
            }).dropna()
            pairs = pairs.reindex(
                pairs["correlation"].abs().sort_values(ascending=False).index
            ).head(top_k)
            if pairs.empty:
                rows.append({
                    "origin_position": int(origin),
                    "indicator_id": indicator,
                    **defaults,
                })
                continue
            weights = pairs["correlation"].abs()
            signed_signal = pairs["correlation"] * pairs["current_z"]
            rows.append({
                "origin_position": int(origin),
                "indicator_id": indicator,
                "down_lead_peer_score": float(
                    -np.average(signed_signal, weights=weights)
                ),
                "down_lead_negative_consensus": float(
                    np.average(signed_signal.lt(0).astype(float), weights=weights)
                ),
                "down_lead_abs_correlation": float(weights.mean()),
                "down_lead_peer_count": float(len(pairs)),
            })
    return pd.DataFrame(rows)


def build_directional_downside_features(
    frame: pd.DataFrame,
    availability_lag: int = 1,
    lead_correlation_window: int = 60,
    lead_minimum_pairs: int = 24,
    lead_top_k: int = 3,
) -> pd.DataFrame:
    """Build causal reversal, regime, and learned peer-leading features."""
    if availability_lag < 1:
        raise ValueError("availability_lag must be at least one month")
    if lead_correlation_window < lead_minimum_pairs:
        raise ValueError("lead correlation window is shorter than minimum pairs")
    if lead_top_k < 1:
        raise ValueError("lead_top_k must be positive")

    indicators = [column for column in frame.columns if column.startswith("X")]
    source = frame[indicators].shift(availability_lag)
    returns = source.pct_change(fill_method=None)
    market_breadth = returns.gt(0).sum(axis=1).div(
        returns.notna().sum(axis=1).replace(0, np.nan)
    )
    market = pd.DataFrame({
        "origin_position": frame["position"],
        "down_market_mean_return": returns.mean(axis=1),
        "down_market_breadth": market_breadth,
        "down_market_breadth_3": market_breadth.rolling(
            3,
            min_periods=2,
        ).mean(),
        "down_market_breadth_change_3": market_breadth.diff(3),
        "down_market_dispersion": returns.std(axis=1),
    })
    parts = []
    for indicator in indicators:
        level = source[indicator]
        indicator_return = returns[indicator]
        volatility_3 = indicator_return.rolling(3, min_periods=2).std()
        volatility_12 = indicator_return.rolling(12, min_periods=6).std()
        volatility_24 = indicator_return.rolling(24, min_periods=12).std()
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
        recent_two = indicator_return.rolling(2, min_periods=2).mean()
        prior_four = indicator_return.shift(2).rolling(4, min_periods=3).mean()
        rolling_high = level.rolling(12, min_periods=6).max()
        rolling_mean = level.rolling(12, min_periods=6).mean()
        part = pd.DataFrame({
            "origin_position": frame["position"],
            "indicator_id": indicator,
            "down_return_1": indicator_return,
            "down_return_lag_1": indicator_return.shift(1),
            "down_return_lag_2": indicator_return.shift(2),
            "down_momentum_3": level.div(
                level.shift(3).replace(0, np.nan)
            ).sub(1.0),
            "down_momentum_6": level.div(
                level.shift(6).replace(0, np.nan)
            ).sub(1.0),
            "down_momentum_12": level.div(
                level.shift(12).replace(0, np.nan)
            ).sub(1.0),
            "down_volatility_3": volatility_3,
            "down_volatility_12": volatility_12,
            "down_drawdown_12": level.div(
                rolling_high.replace(0, np.nan)
            ).sub(1.0),
            "down_distance_mean_12": level.div(
                rolling_mean.replace(0, np.nan)
            ).sub(1.0),
            "down_negative_share_3": indicator_return.lt(0).where(
                indicator_return.notna()
            ).rolling(3, min_periods=2).mean(),
            "down_rise_6_before_2": rise_before_stall,
            "down_stall_2": stall_return,
            "down_rise_score": rise_score,
            "down_stall_score": stall_score,
            "down_exhaustion_score": rise_score - stall_score,
            "down_exhaustion_flag": (
                rise_before_stall.gt(0)
                & rise_score.ge(0.5)
                & stall_score.le(0.5)
            ).astype(float),
            "down_near_high_12": level.ge(0.98 * rolling_high).astype(float),
            "down_deceleration_2_vs_4": prior_four - recent_two,
            "down_volatility_compression": volatility_3.div(
                volatility_12.replace(0, np.nan)
            ),
        })
        parts.append(part.merge(
            market,
            on="origin_position",
            how="left",
            validate="one_to_one",
        ))
    features = pd.concat(parts, ignore_index=True)
    lead = _lead_peer_features(
        frame,
        availability_lag=availability_lag,
        window=lead_correlation_window,
        minimum_pairs=lead_minimum_pairs,
        top_k=lead_top_k,
    )
    return features.merge(
        lead,
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )


def _model_pipeline(
    numeric: list[str],
    logistic_c: float,
    max_iter: int,
    seed: int,
    include_indicator: bool,
) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = [(
        "numeric",
        Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]),
        numeric,
    )]
    if include_indicator:
        transformers.append((
            "indicator",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["indicator_id"],
        ))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        (
            "classifier",
            LogisticRegression(
                C=logistic_c,
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed,
            ),
        ),
    ])


@dataclass
class DirectionalDownsideModel:
    global_model: Any
    local_models: dict[str, Any]
    feature_columns: list[str]


def fit_directional_downside_model(
    train: pd.DataFrame,
    seed: int,
    global_logistic_c: float,
    local_logistic_c: float,
    max_iter: int,
    minimum_local_rows: int,
    minimum_local_class_rows: int,
) -> DirectionalDownsideModel:
    numeric = [
        column for column in DIRECTIONAL_DOWNSIDE_FEATURES
        if column in train.columns
    ]
    if train["down_target"].nunique() < 2:
        raise ValueError("Directional downside model needs both target classes")
    global_model = _model_pipeline(
        numeric,
        logistic_c=global_logistic_c,
        max_iter=max_iter,
        seed=seed,
        include_indicator=True,
    )
    global_model.fit(
        train[[*numeric, "indicator_id"]],
        train["down_target"].astype(int),
    )
    locals_: dict[str, Any] = {}
    for indicator, group in train.groupby("indicator_id", sort=False):
        counts = group["down_target"].value_counts()
        if (
            len(group) < minimum_local_rows
            or len(counts) < 2
            or int(counts.min()) < minimum_local_class_rows
        ):
            continue
        local_model = _model_pipeline(
            numeric,
            logistic_c=local_logistic_c,
            max_iter=max_iter,
            seed=seed,
            include_indicator=False,
        )
        local_model.fit(group[numeric], group["down_target"].astype(int))
        locals_[str(indicator)] = local_model
    return DirectionalDownsideModel(
        global_model=global_model,
        local_models=locals_,
        feature_columns=numeric,
    )


def predict_directional_downside(
    model: DirectionalDownsideModel,
    train: pd.DataFrame,
    test: pd.DataFrame,
    trailing_prior_window: int,
    minimum_pattern_rows: int,
) -> pd.DataFrame:
    numeric = model.feature_columns
    result = test[["origin_position", "indicator_id"]].copy()
    result["p_down_global"] = np.clip(
        model.global_model.predict_proba(
            test[[*numeric, "indicator_id"]]
        )[:, 1],
        1e-6,
        1.0 - 1e-6,
    )
    result["p_down_local"] = result["p_down_global"].to_numpy()
    result["local_model_available"] = False
    for indicator, positions in test.groupby("indicator_id").groups.items():
        local_model = model.local_models.get(str(indicator))
        if local_model is None:
            continue
        result.loc[positions, "p_down_local"] = np.clip(
            local_model.predict_proba(test.loc[positions, numeric])[:, 1],
            1e-6,
            1.0 - 1e-6,
        )
        result.loc[positions, "local_model_available"] = True

    ordered = train.sort_values(["indicator_id", "origin_position"])
    recent = ordered.groupby("indicator_id", sort=False).tail(
        trailing_prior_window
    )
    indicator_prior = recent.groupby("indicator_id")["down_target"].mean()
    result["p_down_indicator_prior"] = result["indicator_id"].map(
        indicator_prior
    ).fillna(float(train["down_target"].mean()))
    pattern = ordered[ordered["down_exhaustion_flag"].eq(1.0)]
    pattern_stats = pattern.groupby("indicator_id")["down_target"].agg(
        ["sum", "count"]
    )
    pattern_probability = (
        pattern_stats["sum"].add(2.0)
        .div(pattern_stats["count"].add(4.0))
    )
    pattern_count = result["indicator_id"].map(pattern_stats["count"]).fillna(0)
    learned_pattern = result["indicator_id"].map(pattern_probability)
    result["p_down_pattern"] = np.where(
        test["down_exhaustion_flag"].eq(1.0).to_numpy()
        & pattern_count.ge(minimum_pattern_rows).to_numpy(),
        learned_pattern.fillna(result["p_down_indicator_prior"]).to_numpy(),
        result["p_down_indicator_prior"].to_numpy(),
    )
    result["pattern_history_rows"] = pattern_count.astype(int)
    return result


def apply_bidirectional_selector(
    base_predictions: pd.DataFrame,
    downside_predictions: pd.DataFrame,
    local_weight: float,
    pattern_weight: float,
    down_threshold: float,
    down_margin: float,
    cap: int,
) -> pd.DataFrame:
    """Choose one direction per indicator and rank the strongest 15 calls."""
    if min(local_weight, pattern_weight, down_margin) < 0:
        raise ValueError("selector weights and margin must be non-negative")
    if local_weight + pattern_weight > 1.0:
        raise ValueError("downside blend weights must sum to at most one")
    if not 0.5 <= down_threshold < 1.0:
        raise ValueError("down_threshold must be in [0.5, 1.0)")
    if cap < 1:
        raise ValueError("cap must be positive")
    merge_columns = [
        "origin_position",
        "indicator_id",
        "p_down_global",
        "p_down_local",
        "p_down_pattern",
        "p_down_indicator_prior",
        "local_model_available",
        "pattern_history_rows",
        "down_fit_through_origin",
    ]
    result = base_predictions.merge(
        downside_predictions[merge_columns],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    global_weight = 1.0 - local_weight - pattern_weight
    result["p_down"] = (
        global_weight * result["p_down_global"]
        + local_weight * result["p_down_local"]
        + pattern_weight * result["p_down_pattern"]
    ).clip(1e-6, 1.0 - 1e-6)
    result["base_predicted_direction"] = result["predicted_direction"]
    result["base_accepted"] = result["accepted"].fillna(False).astype(bool)
    p_up = pd.to_numeric(
        result["p_up_calibrated"],
        errors="coerce",
    ).fillna(pd.to_numeric(result["p_up"], errors="coerce"))
    choose_down = (
        result["p_down"].ge(down_threshold)
        & result["p_down"].ge(p_up + down_margin)
    )
    result["predicted_direction"] = np.where(choose_down, "Down", "Up")
    result["directional_confidence"] = np.where(
        choose_down,
        result["p_down"],
        p_up,
    )
    result["correctness_probability"] = result["directional_confidence"]
    result["correctness_lcb"] = np.nan
    result["selection_score"] = result["directional_confidence"]
    result["selection_mode"] = "bidirectional_downside"
    result["down_local_weight"] = float(local_weight)
    result["down_pattern_weight"] = float(pattern_weight)
    result["down_threshold"] = float(down_threshold)
    result["down_margin"] = float(down_margin)
    result["accepted"] = False
    result["selection_rank"] = np.nan
    result["rejection_reason"] = "bidirectional_monthly_cap"
    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        current = result.loc[positions]
        eligible = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & current["selection_score"].notna()
            & current["p_down"].notna()
        ].sort_values("selection_score", ascending=False)
        if len(eligible) < cap:
            raise AssertionError(
                f"Origin {origin} has fewer than {cap} eligible candidates"
            )
        accepted = eligible.head(cap).index
        result.loc[accepted, "accepted"] = True
        result.loc[accepted, "selection_rank"] = np.arange(1, cap + 1)
        result.loc[accepted, "rejection_reason"] = ""
    result["directional_model_changed"] = (
        result["base_accepted"]
        != result["accepted"].fillna(False).astype(bool)
    ) | (
        result["base_predicted_direction"] != result["predicted_direction"]
    )
    return result


def summarize_bidirectional_predictions(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    predicted = selected["predicted_direction"].eq("Up").astype(int)
    selected["correct"] = predicted.eq(selected["y_true"].astype(int))
    down = selected[selected["predicted_direction"].eq("Down")]
    down_correct = down["y_true"].eq(0)
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(len(down)),
        "down_hits": int(down_correct.sum()),
        "down_accuracy": float(down_correct.mean()) if len(down) else np.nan,
    }
