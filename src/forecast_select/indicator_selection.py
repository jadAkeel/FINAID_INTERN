from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def propagate_correlation_graph(
    probabilities: np.ndarray,
    correlation: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    matrix = np.asarray(correlation, dtype=float).copy()
    if matrix.shape != (len(values), len(values)):
        raise ValueError("correlation matrix shape must match probabilities")
    matrix[~np.isfinite(matrix)] = 0.0
    np.fill_diagonal(matrix, 0.0)
    weights = np.abs(matrix).sum(axis=1)
    neighbour = np.divide(
        matrix @ _logit(values),
        weights,
        out=np.zeros(len(values)),
        where=weights > 0,
    )
    blended_logit = (1.0 - alpha) * _logit(values) + alpha * neighbour
    return np.clip(1.0 / (1.0 + np.exp(-blended_logit)), 1e-6, 1.0 - 1e-6)


def reliability_weighted_correlation(
    changes: pd.DataFrame,
    minimum_pairs: int,
) -> pd.DataFrame:
    """Estimate a signed graph and shrink pairs with limited joint history."""
    if minimum_pairs < 2:
        raise ValueError("minimum_pairs must be at least two")
    if len(changes) < minimum_pairs:
        raise ValueError("Correlation history is shorter than minimum_pairs")
    numeric = changes.apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(min_periods=minimum_pairs).fillna(0.0)
    valid = numeric.notna().astype(float)
    pair_counts = valid.T @ valid
    denominator = max(1, len(numeric) - minimum_pairs + 1)
    reliability = (
        (pair_counts - minimum_pairs + 1) / denominator
    ).clip(0.0, 1.0)
    weighted = (correlation * reliability).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(weighted, 0.0)
    return pd.DataFrame(
        weighted,
        index=correlation.index,
        columns=correlation.columns,
    )


def select_top_indicators(
    target_history: pd.DataFrame,
    predictions: pd.DataFrame,
    cap: int,
    prior_window: int,
    prior_weight: float,
    minimum_history_months: int,
    minimum_indicator_history: int,
    availability_lag: int,
) -> pd.DataFrame:
    if cap < 1 or prior_window < 1:
        raise ValueError("Selection sizes must be positive")
    if not 0.0 <= prior_weight <= 1.0:
        raise ValueError("prior_weight must be between 0 and 1")
    result = predictions.copy()
    defaults = {
        "p_up_calibrated": np.nan,
        "directional_confidence": np.nan,
        "directional_score": np.nan,
        "correctness_probability": np.nan,
        "correctness_lcb": np.nan,
        "indicator_prior": np.nan,
        "indicator_history_rows": 0,
        "selection_score": np.nan,
        "selection_mode": "up_probability",
        "level_c_ready": False,
        "accepted": False,
        "selection_rank": np.nan,
        "rejection_reason": "insufficient_earlier_indicator_history",
        "calibration_fit_through_origin": np.nan,
        "reliability_fit_through_origin": np.nan,
    }
    for column, default in defaults.items():
        result[column] = default
    for origin, positions in result.groupby("origin_position", sort=True).groups.items():
        cutoff = int(origin) - int(availability_lag) - 1
        history = target_history[
            target_history["origin_position"].le(cutoff)
            & target_history["y_true"].notna()
        ].copy()
        current = result.loc[positions].copy()
        if history["origin_position"].nunique() < minimum_history_months:
            continue
        recent = history.sort_values(
            ["indicator_id", "origin_position"]
        ).groupby("indicator_id", sort=False).tail(prior_window)
        stats = recent.groupby("indicator_id")["y_true"].agg(["mean", "count"])
        prior = current["indicator_id"].map(stats["mean"])
        counts = current["indicator_id"].map(stats["count"]).fillna(0).astype(int)
        valid = prior.notna() & counts.ge(minimum_indicator_history)
        model_probability = pd.to_numeric(current["p_up"], errors="coerce").fillna(prior)
        blended = prior_weight * prior + (1.0 - prior_weight) * model_probability
        current["p_up"] = blended.where(valid, current["p_up"])
        current["p_up_calibrated"] = blended
        current["predicted_direction"] = np.where(blended >= 0.5, "Up", "Down")
        current["directional_score"] = np.maximum(blended, 1.0 - blended)
        current["directional_confidence"] = current["directional_score"]
        current["correctness_probability"] = np.nan
        current["indicator_prior"] = prior
        current["indicator_history_rows"] = counts
        current["selection_score"] = blended
        current["level_c_ready"] = valid
        current["calibration_fit_through_origin"] = np.where(valid, cutoff, np.nan)
        current["reliability_fit_through_origin"] = np.where(valid, cutoff, np.nan)
        current["rejection_reason"] = np.where(
            valid,
            "monthly_cap",
            "insufficient_indicator_history",
        )
        accepted_index = current[valid].sort_values(
            "selection_score",
            ascending=False,
        ).head(cap).index
        current.loc[accepted_index, "accepted"] = True
        current.loc[accepted_index, "selection_rank"] = np.arange(
            1,
            len(accepted_index) + 1,
        )
        current.loc[accepted_index, "rejection_reason"] = ""
        result.loc[current.index, current.columns] = current
    return result


def summarize_selected_predictions(
    predictions: pd.DataFrame,
    block_months: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, float | int]:
    accepted = predictions[
        predictions["accepted"] & predictions["y_true"].notna()
    ].copy()
    direction = accepted["predicted_direction"].eq("Up").astype(int)
    accepted["correct"] = direction.eq(accepted["y_true"].astype(int))
    monthly = accepted.groupby("origin_position")["correct"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    block = max(1, min(block_months, len(monthly)))
    samples = []
    for _ in range(max(50, bootstrap_replicates)):
        starts = rng.integers(
            0,
            len(monthly),
            size=max(1, int(np.ceil(len(monthly) / block))),
        )
        sample = np.concatenate([
            np.take(monthly, np.arange(start, start + block) % len(monthly))
            for start in starts
        ])[:len(monthly)]
        samples.append(float(sample.mean()))
    return {
        "months": int(accepted["origin_position"].nunique()),
        "calls": int(len(accepted)),
        "hits": int(accepted["correct"].sum()),
        "accuracy": float(accepted["correct"].mean()),
        "up_calls": int(accepted["predicted_direction"].eq("Up").sum()),
        "down_calls": int(accepted["predicted_direction"].eq("Down").sum()),
        "bootstrap_p10": float(np.quantile(samples, 0.10)),
        "bootstrap_p90": float(np.quantile(samples, 0.90)),
    }
