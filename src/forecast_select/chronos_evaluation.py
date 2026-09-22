from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .chronos_cache import attach_chronos_outcomes

TUNING_WINDOW = (120, 179)
VALIDATION_WINDOW = (180, 219)
CONFIRMATION_WINDOW = (220, 266)
LOCKED_ORIGINS = (268, 315)
MIN_EVALUATION_ORIGIN = 120
MAX_EVALUATION_ORIGIN = 266
ACTIVE_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FROZEN_TUNING_FIT_THROUGH = VALIDATION_WINDOW[0] - 2


def validate_evaluation_dataframe(frame: pd.DataFrame, name: str) -> None:
    """Validate that dataframe respects hard origin constraints and has no locked reads."""
    if frame.empty:
        raise ValueError(f"{name} dataframe is empty")
    required = ["origin_position", "indicator_id"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} dataframe missing required key columns: {missing}")

    if frame.duplicated(required).any():
        raise ValueError(f"{name} dataframe contains duplicate origin_position and indicator_id keys")

    min_orig = int(frame["origin_position"].min())
    max_orig = int(frame["origin_position"].max())
    if min_orig < MIN_EVALUATION_ORIGIN:
        raise ValueError(
            f"{name} min origin {min_orig} is below minimum allowed {MIN_EVALUATION_ORIGIN}"
        )
    if max_orig > MAX_EVALUATION_ORIGIN:
        raise ValueError(
            f"{name} max origin {max_orig} exceeds maximum allowed {MAX_EVALUATION_ORIGIN}; "
            f"origins >= 267 are rejected."
        )

    if "locked_evaluation_read" in frame.columns:
        if frame["locked_evaluation_read"].fillna(False).astype(bool).any():
            raise ValueError(f"{name} contains locked_evaluation_read=True; locked reads are forbidden")


def _assert_complete_origin_coverage(frame: pd.DataFrame, name: str) -> None:
    expected = set(range(MIN_EVALUATION_ORIGIN, MAX_EVALUATION_ORIGIN + 1))
    actual = set(pd.to_numeric(frame["origin_position"], errors="raise").astype(int))
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} must cover every development origin 120..266 exactly; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def extract_active_probability(active: pd.DataFrame) -> pd.Series:
    """
    Extract active model probability:
    Uses finite p_up_calibrated else p_up, never selection_score.
    """
    p_cal = pd.to_numeric(active.get("p_up_calibrated", pd.Series(np.nan, index=active.index)), errors="coerce")
    p_raw = pd.to_numeric(active.get("p_up", pd.Series(np.nan, index=active.index)), errors="coerce")
    active_p = p_cal.where(p_cal.notna() & np.isfinite(p_cal), p_raw)
    return active_p


def compute_brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Compute Brier score loss: mean((p - y)^2)."""
    return float(np.mean((p - y_true) ** 2))


def compute_clipped_log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    """Compute binary cross-entropy with predictions clipped to [eps, 1 - eps]."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    try:
        return float(log_loss(y_true, p_clipped, labels=[0, 1]))
    except Exception:
        return float(-np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)))


def compute_roc_auc(y_true: np.ndarray, p: np.ndarray) -> float | None:
    """Compute ROC AUC when both binary classes are present, else None."""
    if len(np.unique(y_true)) == 2:
        try:
            return float(roc_auc_score(y_true, p))
        except Exception:
            return None
    return None


def compute_fixed_bin_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Compute equal-width 10-bin expected calibration error."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if np.any(mask):
            weight = float(np.sum(mask)) / float(total)
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(p[mask]))
            ece += weight * abs(bin_conf - bin_acc)
    return float(ece)


def compute_probability_metrics(
    y_true: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute standard window metrics on common eligible rows."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    n = len(y)
    if n == 0:
        return {
            "n": 0,
            "hits": 0,
            "accuracy": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "roc_auc": None,
            "ece": np.nan,
        }

    pred_dir = p >= 0.5
    hits = int(np.sum(pred_dir == (y == 1)))
    accuracy = float(hits / n)
    brier = compute_brier_score(y, p)
    loss = compute_clipped_log_loss(y, p)
    auc = compute_roc_auc(y, p)
    ece = compute_fixed_bin_ece(y, p)

    return {
        "n": n,
        "hits": hits,
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": loss,
        "roc_auc": auc,
        "ece": ece,
    }


def compute_diversity_metrics(
    p_chronos: pd.Series | np.ndarray,
    p_active: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Compute diversity metrics between Chronos and Active:
    Pearson/Spearman, disagreement rate/count, accuracies on disagreements,
    active-wrong-Chronos-correct count, double-fault rate.
    """
    p_c = np.asarray(p_chronos, dtype=float)
    p_a = np.asarray(p_active, dtype=float)
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    if n == 0:
        return {
            "n": 0,
            "pearson": np.nan,
            "spearman": np.nan,
            "disagreement_count": 0,
            "disagreement_rate": 0.0,
            "chronos_accuracy_on_disagreements": np.nan,
            "active_accuracy_on_disagreements": np.nan,
            "active_wrong_chronos_correct_count": 0,
            "double_fault_count": 0,
            "double_fault_rate": 0.0,
        }

    s_c = pd.Series(p_c)
    s_a = pd.Series(p_a)
    pearson = float(s_c.corr(s_a, method="pearson"))
    spearman = float(s_c.corr(s_a, method="spearman"))

    dir_c = p_c >= 0.5
    dir_a = p_a >= 0.5
    actual = y == 1

    disagreement = dir_c != dir_a
    dis_count = int(np.sum(disagreement))
    dis_rate = float(dis_count / n)

    if dis_count > 0:
        c_acc_dis = float(np.mean(dir_c[disagreement] == actual[disagreement]))
        a_acc_dis = float(np.mean(dir_a[disagreement] == actual[disagreement]))
    else:
        c_acc_dis = np.nan
        a_acc_dis = np.nan

    chronos_correct = dir_c == actual
    active_correct = dir_a == actual

    active_wrong_c_correct = int(np.sum((~active_correct) & chronos_correct))
    double_fault = int(np.sum((~active_correct) & (~chronos_correct)))
    double_fault_rate = float(double_fault / n)

    return {
        "n": n,
        "pearson": pearson,
        "spearman": spearman,
        "disagreement_count": dis_count,
        "disagreement_rate": dis_rate,
        "chronos_accuracy_on_disagreements": c_acc_dis,
        "active_accuracy_on_disagreements": a_acc_dis,
        "active_wrong_chronos_correct_count": active_wrong_c_correct,
        "double_fault_count": double_fault,
        "double_fault_rate": double_fault_rate,
    }


def _fit_platt_model(
    x_prob: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> LogisticRegression | None:
    """Fit univariate Platt logistic regression on log-odds of probabilities."""
    if len(np.unique(y)) < 2:
        return None
    p_clipped = np.clip(x_prob, eps, 1.0 - eps)
    logit = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        return model
    except Exception:
        return None


def _apply_platt_model(
    model: LogisticRegression | None,
    x_prob: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply fitted Platt model, returning calibrated probabilities clipped to [0.001, 0.999]."""
    if model is None:
        return x_prob.copy()
    p_clipped = np.clip(x_prob, eps, 1.0 - eps)
    logit = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
    try:
        cal = model.predict_proba(logit)[:, 1]
        return np.clip(cal, 0.001, 0.999)
    except Exception:
        return x_prob.copy()


def apply_causal_calibration(
    joined_df: pd.DataFrame,
    min_history_origins: int = 12,
) -> pd.DataFrame:
    """
    Apply strictly causal calibration to Chronos probabilities:
    - Tuning window (120-179): prequential per origin t fits Platt only on earlier tuning rows
      origin <= t - 2 with minimum history and both classes; else raw fallback.
      Explicit status and fit_through_origin <= origin - 2.
    - Frozen Platt fits Tuning labels only through origin 178, the latest label available at
      Validation origin 180 under the t-2 rule, and applies to Validation (180-219) and
      Confirmation (220-266). Labels at origin 179 or later must not affect it.
    """
    result = joined_df.copy()
    result["p_up_chronos_calibrated"] = result["p_up_chronos_raw"]
    result["calibration_status"] = "raw_fallback"
    result["calibration_fit_through_origin"] = None

    # Tuning mask: origin between 120 and 179
    tuning_mask = result["origin_position"].between(TUNING_WINDOW[0], TUNING_WINDOW[1])
    # Eligible rows for calibration fitting: common_eligible and y_true valid
    tuning_eligible = (
        tuning_mask
        & result["common_eligible"]
        & result["y_true"].notna()
        & result["p_up_chronos_raw"].notna()
    )

    tuning_origins = sorted(result.loc[tuning_mask, "origin_position"].unique())

    # 1. Prequential fitting across tuning origins
    for t in tuning_origins:
        orig_indices = result.index[result["origin_position"] == t]
        history_mask = tuning_eligible & (result["origin_position"] <= t - 2)
        history_df = result.loc[history_mask]
        num_hist_origins = history_df["origin_position"].nunique()
        y_hist = history_df["y_true"].to_numpy(dtype=int)

        current_raw = result.loc[orig_indices, "p_up_chronos_raw"].to_numpy(dtype=float)

        if num_hist_origins >= min_history_origins and len(np.unique(y_hist)) >= 2:
            model = _fit_platt_model(
                history_df["p_up_chronos_raw"].to_numpy(dtype=float),
                y_hist,
            )
            if model is not None:
                cal_p = _apply_platt_model(model, current_raw)
                result.loc[orig_indices, "p_up_chronos_calibrated"] = cal_p
                result.loc[orig_indices, "calibration_status"] = "prequential_platt"
                result.loc[orig_indices, "calibration_fit_through_origin"] = int(history_df["origin_position"].max())
                continue

        # Fallback to raw
        result.loc[orig_indices, "p_up_chronos_calibrated"] = current_raw
        if num_hist_origins < min_history_origins:
            result.loc[orig_indices, "calibration_status"] = "fallback_insufficient_history"
        else:
            result.loc[orig_indices, "calibration_status"] = "fallback_single_class"
        result.loc[orig_indices, "calibration_fit_through_origin"] = (
            int(history_df["origin_position"].max()) if not history_df.empty else None
        )

    # 2. Freeze at the information set available for the first Validation forecast.
    # At origin 180, labels are available only through 178. Using origin 179 here would leak.
    frozen_history_mask = tuning_eligible & result["origin_position"].le(
        FROZEN_TUNING_FIT_THROUGH
    )
    all_tuning_history = result.loc[frozen_history_mask]
    y_tuning = all_tuning_history["y_true"].to_numpy(dtype=int)
    frozen_model = None
    if len(np.unique(y_tuning)) >= 2:
        frozen_model = _fit_platt_model(
            all_tuning_history["p_up_chronos_raw"].to_numpy(dtype=float),
            y_tuning,
        )

    frozen_fit_through = (
        int(all_tuning_history["origin_position"].max())
        if not all_tuning_history.empty
        else FROZEN_TUNING_FIT_THROUGH
    )

    out_of_tuning_mask = result["origin_position"].gt(TUNING_WINDOW[1])
    for orig in sorted(result.loc[out_of_tuning_mask, "origin_position"].unique()):
        orig_indices = result.index[result["origin_position"] == orig]
        raw_p = result.loc[orig_indices, "p_up_chronos_raw"].to_numpy(dtype=float)
        if frozen_model is not None:
            cal_p = _apply_platt_model(frozen_model, raw_p)
            result.loc[orig_indices, "p_up_chronos_calibrated"] = cal_p
            result.loc[orig_indices, "calibration_status"] = "frozen_tuning_platt"
            result.loc[orig_indices, "calibration_fit_through_origin"] = frozen_fit_through
        else:
            result.loc[orig_indices, "p_up_chronos_calibrated"] = raw_p
            result.loc[orig_indices, "calibration_status"] = "fallback_frozen_single_class"
            result.loc[orig_indices, "calibration_fit_through_origin"] = frozen_fit_through

    return result


def select_active_weight(
    tuning_df: pd.DataFrame,
    grid: Sequence[float] = ACTIVE_WEIGHT_GRID,
) -> tuple[float, list[dict[str, float]]]:
    """
    Select active_weight from fixed grid [0, .25, .5, .75, 1] on Tuning OOF calibrated Chronos
    + active probabilities minimizing Brier score.
    Ties prefer 1 (largest weight).
    Validation/Confirmation labels must never affect it.
    """
    valid_tuning = tuning_df[
        tuning_df["common_eligible"]
        & tuning_df["y_true"].notna()
        & tuning_df["p_up_active"].notna()
        & tuning_df["p_up_chronos_calibrated"].notna()
    ]
    if valid_tuning.empty:
        raise ValueError("Cannot select active_weight: no valid common eligible tuning rows with labels")

    y = valid_tuning["y_true"].to_numpy(dtype=int)
    p_active = valid_tuning["p_up_active"].to_numpy(dtype=float)
    p_chronos = valid_tuning["p_up_chronos_calibrated"].to_numpy(dtype=float)

    grid_results = []
    for w in grid:
        blend_p = float(w) * p_active + (1.0 - float(w)) * p_chronos
        brier = compute_brier_score(y, blend_p)
        grid_results.append({
            "weight": float(w),
            "brier": float(brier),
        })

    # Minimize brier; for ties, prefer 1.0 (larger weight)
    best_candidate = min(grid_results, key=lambda item: (item["brier"], -item["weight"]))
    return float(best_candidate["weight"]), grid_results


def compute_matched_coverage(
    eval_df: pd.DataFrame,
    active_predictions: pd.DataFrame,
    origin_range: tuple[int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Compute matched coverage on requested origin range:
    - For each origin:
      Exact active accepted count k.
      Candidate ranks common eligible by abs(blend - 0.5) descending with stable indicator_id ascending tie.
      Active comparator uses original accepted rows and directions.
      Asserts equal calls per origin: calls_candidate == calls_active == k.
    """
    min_orig, max_orig = origin_range
    window_origins = sorted([
        o for o in eval_df["origin_position"].unique()
        if min_orig <= o <= max_orig
    ])

    per_origin_records = []
    selected_candidate_rows = []

    for o in window_origins:
        # Active accepted rows for this origin
        active_orig = active_predictions[
            (active_predictions["origin_position"] == o)
            & active_predictions["accepted"].fillna(False).astype(bool)
        ]
        k = len(active_orig)
        if k == 0:
            continue

        # Active comparator calls and hits
        active_actual = active_orig["y_true"].to_numpy(dtype=int)
        active_dir = active_orig["predicted_direction"].to_numpy(dtype=str)
        active_hits = int(np.sum(active_dir == np.where(active_actual == 1, "Up", "Down")))

        # Candidate pool: common eligible rows at this origin
        cand_pool = eval_df[
            (eval_df["origin_position"] == o)
            & eval_df["common_eligible"]
            & eval_df["y_true"].notna()
        ].copy()

        if len(cand_pool) < k:
            raise AssertionError(
                f"Origin {o}: common eligible rows ({len(cand_pool)}) fewer than active accepted calls ({k})"
            )

        cand_pool["confidence"] = (cand_pool["p_up_blend"] - 0.5).abs()
        # Stable sort: confidence descending, indicator_id ascending
        cand_sorted = cand_pool.sort_values(
            by=["confidence", "indicator_id"],
            ascending=[False, True],
        )
        cand_selected = cand_sorted.iloc[:k].copy()

        cand_actual = cand_selected["y_true"].to_numpy(dtype=int)
        cand_p = cand_selected["p_up_blend"].to_numpy(dtype=float)
        cand_dir = np.where(cand_p >= 0.5, "Up", "Down")
        cand_selected["predicted_direction"] = cand_dir
        cand_hits = int(np.sum(cand_dir == np.where(cand_actual == 1, "Up", "Down")))

        # Hard assertion: equal calls per origin
        if len(cand_selected) != k:
            raise AssertionError(
                f"Origin {o}: Candidate calls {len(cand_selected)} != Active calls {k}"
            )

        delta = cand_hits - active_hits
        per_origin_records.append({
            "origin_position": int(o),
            "calls": int(k),
            "candidate_hits": int(cand_hits),
            "active_hits": int(active_hits),
            "hit_delta": int(delta),
        })
        selected_candidate_rows.append(cand_selected)

    matched_comparison_df = (
        pd.concat(selected_candidate_rows, ignore_index=True)
        if selected_candidate_rows
        else pd.DataFrame()
    )
    return matched_comparison_df, per_origin_records


def origin_blocked_paired_bootstrap(
    per_origin_deltas: Sequence[float | int],
    block_size: int = 6,
    replicates: int = 500,
    seed: int = 20260727,
) -> dict[str, float]:
    """
    Origin-blocked paired bootstrap of per-origin hit delta:
    - Contiguous blocks of size 6
    - 500 replicates deterministic with seed
    """
    deltas = np.asarray(per_origin_deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return {
            "months": 0,
            "total_hit_delta": 0.0,
            "mean_monthly_hit_delta": 0.0,
            "bootstrap_p10": 0.0,
            "bootstrap_p50": 0.0,
            "bootstrap_p90": 0.0,
        }

    if block_size < 1 or replicates < 1:
        raise ValueError("block_size and replicates must be positive")
    effective_block_size = min(block_size, n)
    blocks = [
        deltas[i : i + effective_block_size]
        for i in range(0, n - effective_block_size + 1)
    ]
    num_blocks = len(blocks)
    draws_per_replicate = int(np.ceil(n / effective_block_size))
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(replicates):
        idx = rng.choice(num_blocks, size=draws_per_replicate, replace=True)
        sampled = np.concatenate([blocks[i] for i in idx])[:n]
        samples.append(float(np.sum(sampled)))

    return {
        "months": int(n),
        "total_hit_delta": float(np.sum(deltas)),
        "mean_monthly_hit_delta": float(np.mean(deltas)),
        "bootstrap_p10": float(np.quantile(samples, 0.10)),
        "bootstrap_p50": float(np.quantile(samples, 0.50)),
        "bootstrap_p90": float(np.quantile(samples, 0.90)),
    }


def evaluate_chronos_gate(
    validation_matched_bootstrap: dict[str, float],
    validation_metrics: dict[str, Any],
    confirmation_matched_bootstrap: dict[str, float],
    locked_evaluation_read: bool = False,
    max_origin: int = 266,
) -> dict[str, Any]:
    """
    Evaluate the non-locked research gate:
    Requires:
    - validation hit delta > 0
    - validation bootstrap p10 >= 0
    - validation blend Brier < active
    - confirmation hit delta >= 0
    - no locked reads (locked_evaluation_read=False and max origin <= 266)

    Hard barrier:
    - promotion_eligible is always False
    - promotion_requires_locked_evaluation is True
    """
    val_delta = validation_matched_bootstrap.get("total_hit_delta", 0.0)
    val_p10 = validation_matched_bootstrap.get("bootstrap_p10", -1.0)
    val_blend_brier = validation_metrics.get("blend", {}).get("brier", np.nan)
    val_active_brier = validation_metrics.get("active", {}).get("brier", np.nan)
    conf_delta = confirmation_matched_bootstrap.get("total_hit_delta", -1.0)

    no_locked_reads = (not locked_evaluation_read) and (max_origin <= MAX_EVALUATION_ORIGIN)

    criteria = {
        "validation_hit_delta_positive": bool(val_delta > 0),
        "validation_bootstrap_p10_nonnegative": bool(val_p10 >= 0.0),
        "validation_blend_brier_improves_active": bool(val_blend_brier < val_active_brier),
        "confirmation_hit_delta_nonnegative": bool(conf_delta >= 0),
        "no_locked_reads": bool(no_locked_reads),
    }

    nonlocked_gate_passed = bool(all(criteria.values()))

    return {
        "nonlocked_gate_passed": nonlocked_gate_passed,
        "criteria": criteria,
        "promotion_eligible": False,
        "promotion_requires_locked_evaluation": True,
        "allow_promotion": False,
        "active_model_changed": False,
    }


def evaluate_chronos_zero_shot(
    chronos_inputs: pd.DataFrame,
    chronos_outcomes: pd.DataFrame,
    active_predictions: pd.DataFrame,
    config: dict[str, Any] | None = None,
    min_history_origins: int = 12,
    bootstrap_block_size: int = 6,
    bootstrap_replicates: int = 500,
    seed: int = 20260727,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Pure evaluator for Chronos-2 zero-shot directional evaluation:
    1. Pure exact-key join validation for Chronos inputs/outcomes/active.
    2. Active probability extraction (p_up_calibrated else p_up, never selection_score).
    3. Common eligible rows determination.
    4. Causal calibration: tuning prequential Platt, frozen Tuning Platt on Validation/Confirmation.
    5. Active blend weight selection on Tuning OOF calibrated Chronos + active probabilities minimizing Brier.
    6. Matched coverage comparison against active comparator with origin-blocked bootstrap.
    7. Window metrics & diversity metrics on common eligible rows.
    8. Gate evaluation (nonlocked_gate_passed, promotion_eligible=False).
    """
    # 1. Validate dataframes
    validate_evaluation_dataframe(chronos_inputs, "chronos_inputs")
    validate_evaluation_dataframe(chronos_outcomes, "chronos_outcomes")
    validate_evaluation_dataframe(active_predictions, "active_predictions")
    _assert_complete_origin_coverage(chronos_inputs, "chronos_inputs")
    _assert_complete_origin_coverage(chronos_outcomes, "chronos_outcomes")
    _assert_complete_origin_coverage(active_predictions, "active_predictions")

    # Combine Chronos inputs and outcomes
    chronos_combined = attach_chronos_outcomes(chronos_inputs, chronos_outcomes)

    # 2. Extract active probabilities and properties
    active_df = active_predictions.copy()
    active_df["p_up_active"] = extract_active_probability(active_df)

    active_sub = active_df[[
        "origin_position",
        "indicator_id",
        "p_up_active",
        "accepted",
        "predicted_direction",
        "y_true",
    ]].rename(columns={
        "accepted": "active_accepted",
        "predicted_direction": "active_predicted_direction",
        "y_true": "active_y_true",
    })
    if "eligible" in active_df.columns:
        active_sub["active_eligible"] = active_df["eligible"].fillna(False).astype(bool)
    else:
        active_sub["active_eligible"] = active_sub["p_up_active"].notna() & np.isfinite(active_sub["p_up_active"])

    keys = ["origin_position", "indicator_id"]
    chronos_keys = pd.MultiIndex.from_frame(chronos_combined[keys])
    active_keys = pd.MultiIndex.from_frame(active_sub[keys])
    missing_active_keys = active_keys.difference(chronos_keys)
    if len(missing_active_keys):
        raise ValueError(
            "Active predictions contain keys absent from Chronos inputs: "
            f"{missing_active_keys.tolist()[:5]}"
        )
    joined = chronos_combined.merge(active_sub, on=keys, how="inner", validate="one_to_one")

    comparable_labels = joined["y_true"].notna() & joined["active_y_true"].notna()
    if not np.array_equal(
        joined.loc[comparable_labels, "y_true"].to_numpy(dtype=float),
        joined.loc[comparable_labels, "active_y_true"].to_numpy(dtype=float),
    ):
        raise ValueError("Active and Chronos outcome labels disagree on shared keys")

    # Determine eligibility
    joined["chronos_eligible"] = joined["eligible"].fillna(False).astype(bool)
    joined["common_eligible"] = (
        joined["chronos_eligible"]
        & joined["active_eligible"]
        & joined["p_up_active"].notna()
        & np.isfinite(joined["p_up_active"])
        & joined["p_up"].notna()
        & np.isfinite(joined["p_up"])
    )
    joined["p_up_chronos_raw"] = joined["p_up"]

    # 3. Apply causal Platt calibration
    calibrated = apply_causal_calibration(
        joined,
        min_history_origins=min_history_origins,
    )

    # 4. Select active weight from fixed grid on Tuning OOF only
    tuning_slice = calibrated[
        calibrated["origin_position"].between(
            TUNING_WINDOW[0], FROZEN_TUNING_FIT_THROUGH
        )
    ]
    active_weight, grid_searches = select_active_weight(tuning_slice, grid=ACTIVE_WEIGHT_GRID)

    # Blend probability
    calibrated["active_weight"] = active_weight
    calibrated["p_up_blend"] = (
        active_weight * calibrated["p_up_active"]
        + (1.0 - active_weight) * calibrated["p_up_chronos_calibrated"]
    )
    calibrated["blend_predicted_direction"] = np.where(
        calibrated["p_up_blend"] >= 0.5, "Up", "Down"
    )

    # 5. Window metrics & Diversity metrics
    windows = {
        "tuning": TUNING_WINDOW,
        "validation": VALIDATION_WINDOW,
        "confirmation": CONFIRMATION_WINDOW,
        "overall_nonlocked": (MIN_EVALUATION_ORIGIN, MAX_EVALUATION_ORIGIN),
    }

    window_metrics: dict[str, Any] = {}
    diversity_metrics: dict[str, Any] = {}
    matched_coverage_summaries: dict[str, Any] = {}
    all_matched_comparisons: list[pd.DataFrame] = []

    for win_name, win_bounds in windows.items():
        start_o, end_o = win_bounds
        mask = (
            calibrated["origin_position"].between(start_o, end_o)
            & calibrated["common_eligible"]
            & calibrated["y_true"].notna()
        )
        sub = calibrated.loc[mask]

        # Probability metrics
        y_vals = sub["y_true"].to_numpy(dtype=int)
        raw_m = compute_probability_metrics(y_vals, sub["p_up_chronos_raw"].to_numpy(dtype=float))
        cal_m = compute_probability_metrics(y_vals, sub["p_up_chronos_calibrated"].to_numpy(dtype=float))
        act_m = compute_probability_metrics(y_vals, sub["p_up_active"].to_numpy(dtype=float))
        bln_m = compute_probability_metrics(y_vals, sub["p_up_blend"].to_numpy(dtype=float))

        window_metrics[win_name] = {
            "months": int(sub["origin_position"].nunique()),
            "common_eligible_rows": int(len(sub)),
            "chronos_raw": raw_m,
            "chronos_calibrated": cal_m,
            "active": act_m,
            "blend": bln_m,
        }

        # Diversity metrics
        diversity_metrics[win_name] = compute_diversity_metrics(
            sub["p_up_chronos_raw"].to_numpy(dtype=float),
            sub["p_up_active"].to_numpy(dtype=float),
            y_vals,
        )

        # Matched coverage on this window (excluding overall to avoid duplication in records)
        if win_name in {"tuning", "validation", "confirmation"}:
            matched_df, orig_records = compute_matched_coverage(
                calibrated,
                active_predictions,
                origin_range=win_bounds,
            )
            all_matched_comparisons.append(matched_df)
            deltas = [r["hit_delta"] for r in orig_records]
            b_res = origin_blocked_paired_bootstrap(
                deltas,
                block_size=bootstrap_block_size,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            matched_coverage_summaries[win_name] = {
                "origin_count": len(orig_records),
                "total_calls": sum(r["calls"] for r in orig_records),
                "candidate_hits": sum(r["candidate_hits"] for r in orig_records),
                "active_hits": sum(r["active_hits"] for r in orig_records),
                "total_hit_delta": b_res["total_hit_delta"],
                "mean_monthly_hit_delta": b_res["mean_monthly_hit_delta"],
                "bootstrap_p10": b_res["bootstrap_p10"],
                "bootstrap_p50": b_res["bootstrap_p50"],
                "bootstrap_p90": b_res["bootstrap_p90"],
                "monthly_records": orig_records,
            }

    # 6. Gate evaluation
    val_b = matched_coverage_summaries.get("validation", {})
    conf_b = matched_coverage_summaries.get("confirmation", {})
    gate_result = evaluate_chronos_gate(
        validation_matched_bootstrap=val_b,
        validation_metrics=window_metrics.get("validation", {}),
        confirmation_matched_bootstrap=conf_b,
        locked_evaluation_read=False,
        max_origin=int(calibrated["origin_position"].max()),
    )

    combined_matched_df = (
        pd.concat(all_matched_comparisons, ignore_index=True)
        if all_matched_comparisons
        else pd.DataFrame()
    )

    summary = {
        "experiment_id": "chronos_zero_shot",
        "experiment_name": "Chronos-2 Zero-Shot Directional Forecasting",
        "package_pin": (config or {}).get("package_pin", "chronos-forecasting==2.3.2"),
        "model_id": (config or {}).get("model_id", "amazon/chronos-2"),
        "model_revision": (config or {}).get("model_revision", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
        "seed": seed,
        "selected_active_weight": active_weight,
        "frozen_policy_fit_through_origin": FROZEN_TUNING_FIT_THROUGH,
        "active_weight_grid_search": grid_searches,
        "windows": window_metrics,
        "diversity": diversity_metrics,
        "matched_coverage": matched_coverage_summaries,
        "gate": gate_result,
        "nonlocked_gate_passed": gate_result["nonlocked_gate_passed"],
        "promotion_eligible": False,
        "promotion_requires_locked_evaluation": True,
        "allow_promotion": False,
        "active_model_changed": False,
        "locked_evaluation_read": False,
        "maximum_origin_position": int(calibrated["origin_position"].max()),
        "contamination_caveat": (
            "Pre-trained foundation models may have been exposed to public macroeconomic series "
            "during pre-training. Historical backtest performance may reflect training data contamination. "
            "Out-of-sample validity requires forward live or post-release paper trading."
        ),
    }

    return calibrated, combined_matched_df, summary
