from __future__ import annotations

import math

import numpy as np
import pandas as pd


SCORE_SEMANTICS_VERSION = "score_semantics_v2"
CORRECTNESS_STATUS = "unavailable_no_valid_oof_individual_calibrator"
COHORT_VERSION = "causal_laplace_wilson_cohort_v1"


def decision_correctness(predictions: pd.DataFrame) -> pd.Series:
    """Return whether each labeled final direction was correct."""
    labeled = (
        predictions["y_true"].notna()
        & predictions["predicted_direction"].isin(["Up", "Down"])
    )
    correct = pd.Series(np.nan, index=predictions.index, dtype=float)
    correct.loc[labeled] = np.where(
        predictions.loc[labeled, "predicted_direction"].eq("Up"),
        predictions.loc[labeled, "y_true"].eq(1.0),
        predictions.loc[labeled, "y_true"].eq(0.0),
    ).astype(float)
    return correct


def wilson_lower_bound(
    hits: float,
    calls: float,
    z_value: float = 1.6448536269514722,
) -> float:
    """One-sided Wilson lower bound for a Bernoulli success rate."""
    if calls <= 0:
        return np.nan
    rate = float(hits) / float(calls)
    denominator = 1.0 + z_value**2 / calls
    centre = rate + z_value**2 / (2.0 * calls)
    radius = z_value * math.sqrt(
        rate * (1.0 - rate) / calls + z_value**2 / (4.0 * calls**2)
    )
    return float((centre - radius) / denominator)


def apply_correctness_semantics(
    predictions: pd.DataFrame,
    minimum_history_months: int = 12,
    label_lag_origins: int = 2,
) -> pd.DataFrame:
    """Separate ranking scores from unavailable individual correctness odds.

    A causal marginal cohort rate is retained for monitoring, but it is not an
    individualized probability and must not be used to rank or abstain calls.
    """
    if minimum_history_months < 1 or label_lag_origins < 1:
        raise ValueError("Correctness history and label lag must be positive")
    result = predictions.copy()
    if "legacy_correctness_probability" not in result:
        legacy_probability = pd.to_numeric(
            result.get(
                "correctness_probability",
                pd.Series(np.nan, index=result.index),
            ),
            errors="coerce",
        )
        legacy_score = pd.to_numeric(
            result.get(
                "directional_score",
                result.get(
                    "directional_confidence",
                    pd.Series(np.nan, index=result.index),
                ),
            ),
            errors="coerce",
        )
        result["legacy_correctness_probability"] = (
            legacy_probability.fillna(legacy_score)
        )
    if "legacy_correctness_lcb" not in result:
        result["legacy_correctness_lcb"] = pd.to_numeric(
            result.get("correctness_lcb", pd.Series(np.nan, index=result.index)),
            errors="coerce",
        )
    result["directional_score"] = pd.to_numeric(
        result.get(
            "directional_score",
            result.get(
                "directional_confidence",
                pd.Series(np.nan, index=result.index),
            ),
        ),
        errors="coerce",
    )
    result["correctness_probability"] = np.nan
    result["correctness_lcb"] = np.nan
    result["cohort_correctness_probability"] = np.nan
    result["cohort_correctness_lcb"] = np.nan
    result["cohort_correctness_history_calls"] = 0
    result["cohort_correctness_history_months"] = 0
    result["correctness_fit_through_origin"] = np.nan
    result["score_semantics_version"] = SCORE_SEMANTICS_VERSION
    result["p_up_semantics"] = (
        "estimated_up_direction_probability_not_proven_calibrated"
    )
    result["selection_score_semantics"] = "ranking_utility_not_probability"
    result["selection_score_is_probability"] = False
    result["directional_confidence_semantics"] = (
        "legacy_name_for_uncalibrated_directional_score"
    )
    result["directional_confidence_is_calibrated"] = False
    result["correctness_probability_status"] = np.where(
        result["accepted"].fillna(False).astype(bool),
        CORRECTNESS_STATUS,
        "not_applicable_unselected",
    )
    result["correctness_probability_version"] = "unavailable_v1"
    result["cohort_correctness_version"] = COHORT_VERSION
    result["cohort_correctness_semantics"] = (
        "causal_marginal_selected_call_rate_not_individual_probability"
    )

    correctness = decision_correctness(result)
    selected_labeled = result[
        result["accepted"].fillna(False).astype(bool)
        & correctness.notna()
    ].copy()
    selected_labeled["_correct"] = correctness.loc[selected_labeled.index]
    for origin, positions in result.groupby(
        "origin_position", sort=True
    ).groups.items():
        current_indices = pd.Index(positions)
        accepted_indices = current_indices[
            result.loc[current_indices, "accepted"].fillna(False).astype(bool)
        ]
        history = selected_labeled[
            selected_labeled["origin_position"].le(
                int(origin) - int(label_lag_origins)
            )
        ]
        history_months = int(history["origin_position"].nunique())
        result.loc[current_indices, "cohort_correctness_history_months"] = (
            history_months
        )
        result.loc[current_indices, "cohort_correctness_history_calls"] = len(
            history
        )
        if history_months < minimum_history_months or not len(accepted_indices):
            continue
        hits = float(history["_correct"].sum())
        calls = float(len(history))
        posterior_hits = hits + 1.0
        posterior_calls = calls + 2.0
        result.loc[accepted_indices, "cohort_correctness_probability"] = (
            posterior_hits / posterior_calls
        )
        result.loc[accepted_indices, "cohort_correctness_lcb"] = (
            wilson_lower_bound(posterior_hits, posterior_calls)
        )
        result.loc[accepted_indices, "correctness_fit_through_origin"] = int(
            history["origin_position"].max()
        )
    return result
