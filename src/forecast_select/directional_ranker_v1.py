from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "p_up_raw", "p_up_calibrated", "p_up_generalized_calibrated",
    "indicator_prior", "indicator_history_log", "asset_group_relative_logit",
    "p_down_global", "p_down_local", "p_down_pattern",
    "p_down_indicator_prior", "down_disagreement", "down_return_1",
    "down_momentum_3", "regime_stress", "regime_uncertainty",
    "market_breadth", "market_breadth_change_3", "market_dispersion",
    "forecast_market_breadth", "graph_adjustment",
]
CATEGORICAL_FEATURES = ["indicator_id", "asset_group"]


@dataclass(frozen=True)
class DirectionalRankerResult:
    model: Pipeline
    regularization_c: float
    late_tuning_auc: float


def build_directional_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[NUMERIC_FEATURES[:4]].apply(pd.to_numeric, errors="coerce").copy()
    result["indicator_history_log"] = np.log1p(
        pd.to_numeric(frame["indicator_history_rows"], errors="coerce")
    )
    for column in NUMERIC_FEATURES[5:10]:
        result[column] = pd.to_numeric(frame[column], errors="coerce")
    down_columns = [
        "p_down_global", "p_down_local", "p_down_pattern",
        "p_down_indicator_prior",
    ]
    result["down_disagreement"] = frame[down_columns].apply(
        pd.to_numeric, errors="coerce"
    ).std(axis=1, ddof=0)
    for column in NUMERIC_FEATURES[11:]:
        if column == "regime_uncertainty":
            stress = pd.to_numeric(frame["regime_stress"], errors="coerce")
            result[column] = 1.0 - 2.0 * (stress - 0.5).abs()
        elif column == "graph_adjustment":
            generalized = pd.to_numeric(
                frame["p_up_generalized_calibrated"], errors="coerce"
            )
            base = pd.to_numeric(frame["p_up_calibrated"], errors="coerce")
            result[column] = (generalized - base).abs()
        else:
            result[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        result[column] = frame[column].fillna("unknown").astype(str)
    return result[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]


def _pipeline(c_value: float) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(C=float(c_value), max_iter=2000)),
    ])


def fit_directional_ranker(
    frame: pd.DataFrame,
    *,
    early_tuning: tuple[int, int] = (120, 149),
    late_tuning: tuple[int, int] = (150, 179),
    c_grid: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1),
) -> DirectionalRankerResult:
    labeled = frame[frame["y_true"].notna()].copy()
    early = labeled[labeled["origin_position"].between(*early_tuning)]
    late = labeled[labeled["origin_position"].between(*late_tuning)]
    if early["y_true"].nunique() < 2 or late["y_true"].nunique() < 2:
        raise ValueError("Directional tuning windows lack both target classes")
    candidates = []
    for c_value in c_grid:
        model = _pipeline(c_value)
        model.fit(build_directional_features(early), early["y_true"].astype(int))
        probability = model.predict_proba(build_directional_features(late))[:, 1]
        auc = float(roc_auc_score(late["y_true"], probability))
        candidates.append((auc, -float(c_value), float(c_value)))
    late_auc, _, selected_c = max(candidates)
    tuning = labeled[labeled["origin_position"].between(early_tuning[0], late_tuning[1])]
    model = _pipeline(selected_c)
    model.fit(build_directional_features(tuning), tuning["y_true"].astype(int))
    return DirectionalRankerResult(model, selected_c, late_auc)


def score_directional_candidates(
    frame: pd.DataFrame, fitted: DirectionalRankerResult
) -> pd.DataFrame:
    result = frame.copy()
    result["p_up_directional_v1"] = fitted.model.predict_proba(
        build_directional_features(result)
    )[:, 1]
    result["predicted_direction_v1"] = np.where(
        result["p_up_directional_v1"].ge(0.5), "Up", "Down"
    )
    result["directional_selection_score_v1"] = np.maximum(
        result["p_up_directional_v1"], 1.0 - result["p_up_directional_v1"]
    )
    return result


def select_with_existing_caps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["accepted_directional_v1"] = False
    result["selection_rank_directional_v1"] = np.nan
    for origin, indices in result.groupby("origin_position", sort=True).groups.items():
        current = result.loc[list(indices)]
        cap = int(pd.to_numeric(current["regime_cap"], errors="coerce").dropna().iloc[0])
        ready = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & ~current["adaptive_data_quality_excluded"].fillna(False).astype(bool)
        ].sort_values(
            ["directional_selection_score_v1", "indicator_id"],
            ascending=[False, True],
        )
        selected = ready.head(cap)
        if len(selected) != cap:
            raise AssertionError(f"Origin {origin} has fewer than {cap} candidates")
        result.loc[selected.index, "accepted_directional_v1"] = True
        result.loc[selected.index, "selection_rank_directional_v1"] = np.arange(1, cap + 1)
    return result


def directional_metrics(frame: pd.DataFrame, probability_column: str) -> dict:
    labeled = frame[frame["y_true"].notna() & frame[probability_column].notna()]
    y = labeled["y_true"].astype(int)
    probability = pd.to_numeric(labeled[probability_column], errors="coerce").clip(1e-6, 1 - 1e-6)
    predicted = probability.ge(0.5).astype(int)
    return {
        "rows": int(len(labeled)),
        "auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "accuracy": float(predicted.eq(y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
    }


def selection_metrics(
    frame: pd.DataFrame,
    accepted_column: str,
    direction_column: str,
) -> dict:
    selected = frame[frame[accepted_column].fillna(False).astype(bool) & frame["y_true"].notna()]
    correct = np.where(
        selected[direction_column].eq("Up"),
        selected["y_true"].eq(1.0),
        selected["y_true"].eq(0.0),
    )
    return {
        "calls": int(len(selected)), "hits": int(np.sum(correct)),
        "accuracy": float(np.mean(correct)),
        "down_calls": int(selected[direction_column].eq("Down").sum()),
    }
