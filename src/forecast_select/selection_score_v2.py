from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "p_up_base", "p_down_base", "up_down_margin", "indicator_prior",
    "indicator_history_log", "prior_gap", "down_disagreement",
    "asset_group_relative_logit", "graph_adjustment", "regime_stress",
    "regime_uncertainty", "market_dispersion", "forecast_market_breadth",
    "predicted_down",
]


@dataclass(frozen=True)
class SelectionScoreV2Result:
    model: Pipeline
    regularization_c: float
    tuning_auc: float


def decision_correctness(frame: pd.DataFrame) -> pd.Series:
    up = frame["predicted_direction"].eq("Up")
    return ((up & frame["y_true"].eq(1.0)) | (~up & frame["y_true"].eq(0.0))).astype(int)


def build_selection_features(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "p_up_base", "p_down_base", "indicator_prior", "indicator_history_rows",
        "asset_group_relative_logit", "p_up_generalized_calibrated",
        "p_up_calibrated", "regime_stress", "market_dispersion",
        "forecast_market_breadth",
    ]
    result = frame[numeric].apply(pd.to_numeric, errors="coerce").copy()
    down = frame[[
        "p_down_global", "p_down_local", "p_down_pattern",
        "p_down_indicator_prior",
    ]].apply(pd.to_numeric, errors="coerce")
    result["up_down_margin"] = result["p_up_base"] - result["p_down_base"]
    result["indicator_history_log"] = np.log1p(result["indicator_history_rows"])
    result["prior_gap"] = result["p_up_base"] - result["indicator_prior"]
    result["down_disagreement"] = down.std(axis=1, ddof=0)
    result["graph_adjustment"] = (
        result["p_up_generalized_calibrated"] - result["p_up_calibrated"]
    ).abs()
    result["regime_uncertainty"] = 1.0 - 2.0 * (result["regime_stress"] - 0.5).abs()
    result["predicted_down"] = frame["predicted_direction"].eq("Down").astype(float)
    return result[FEATURE_COLUMNS]


def _pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=float(c_value), max_iter=2000)),
    ])


def fit_selection_score_v2(
    frame: pd.DataFrame,
    *,
    early_tuning: tuple[int, int] = (120, 149),
    late_tuning: tuple[int, int] = (150, 179),
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3),
) -> SelectionScoreV2Result:
    labeled = frame[frame["y_true"].notna()].copy()
    labeled["correct"] = decision_correctness(labeled)
    early = labeled[labeled["origin_position"].between(*early_tuning)]
    late = labeled[labeled["origin_position"].between(*late_tuning)]
    if early.empty or late.empty or early["correct"].nunique() < 2:
        raise ValueError("Selection-score tuning windows lack labeled classes")
    candidates = []
    for c_value in c_grid:
        model = _pipeline(c_value)
        model.fit(build_selection_features(early), early["correct"])
        probability = model.predict_proba(build_selection_features(late))[:, 1]
        candidates.append((float(roc_auc_score(late["correct"], probability)), -c_value, c_value))
    tuning_auc, _, selected_c = max(candidates)
    tuning = labeled[labeled["origin_position"].between(early_tuning[0], late_tuning[1])]
    model = _pipeline(selected_c)
    model.fit(build_selection_features(tuning), tuning["correct"])
    return SelectionScoreV2Result(model, float(selected_c), float(tuning_auc))


def score_selection_candidates(frame: pd.DataFrame, fitted: SelectionScoreV2Result) -> pd.DataFrame:
    result = frame.copy()
    result["selection_score_v2"] = fitted.model.predict_proba(build_selection_features(result))[:, 1]
    return result


def select_with_existing_caps(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["accepted_v2"] = False
    result["selection_rank_v2"] = np.nan
    for origin, indices in result.groupby("origin_position", sort=True).groups.items():
        current = result.loc[list(indices)]
        cap = int(pd.to_numeric(current["regime_cap"], errors="coerce").dropna().iloc[0])
        ready = current[
            current["level_c_ready"].fillna(False).astype(bool)
            & ~current["adaptive_data_quality_excluded"].fillna(False).astype(bool)
        ].sort_values(["selection_score_v2", "indicator_id"], ascending=[False, True])
        selected = ready.head(cap)
        if len(selected) != cap:
            raise AssertionError(f"Origin {origin} has fewer than {cap} eligible candidates")
        result.loc[selected.index, "accepted_v2"] = True
        result.loc[selected.index, "selection_rank_v2"] = np.arange(1, cap + 1)
    return result


def score_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    labeled = frame[frame["y_true"].notna() & frame[score_column].notna()].copy()
    correct = decision_correctness(labeled)
    probability = pd.to_numeric(labeled[score_column], errors="coerce").clip(1e-6, 1 - 1e-6)
    return {
        "rows": int(len(labeled)),
        "auc": float(roc_auc_score(correct, probability)),
        "brier": float(brier_score_loss(correct, probability)),
    }


def selection_metrics(frame: pd.DataFrame, accepted_column: str) -> dict[str, float | int]:
    selected = frame[frame[accepted_column].fillna(False).astype(bool) & frame["y_true"].notna()]
    correct = decision_correctness(selected)
    return {"calls": int(len(selected)), "hits": int(correct.sum()), "accuracy": float(correct.mean())}
