"""Metrics, overlap analysis and block-bootstrap comparisons for the
Local / Interaction Logistic Up Selector study.

Monthly aggregation is used everywhere that uncertainty is reported: the 50
indicator rows inside one origin share a common regime and are not treated as
independent observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


PERIODS = ("tuning", "validation", "confirmation")


def period_origins(settings: dict) -> dict[str, tuple[int, int]]:
    return {
        "tuning": tuple(int(v) for v in settings["tuning_origins"]),
        "validation": tuple(int(v) for v in settings["validation_origins"]),
        "confirmation": tuple(int(v) for v in settings["confirmation_origins"]),
    }


def origin_mask(frame: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    start, end = window
    return frame["origin_position"].between(int(start), int(end))


# ---------------------------------------------------------------------------
# Selector (top-15) metrics
# ---------------------------------------------------------------------------


def selected_calls(result: pd.DataFrame) -> pd.DataFrame:
    accepted = result[
        result["accepted"].fillna(False).astype(bool) & result["y_true"].notna()
    ].copy()
    direction = accepted["predicted_direction"].eq("Up").astype(int)
    accepted["correct"] = direction.eq(accepted["y_true"].astype(int)).astype(int)
    return accepted


def selector_metrics(result: pd.DataFrame, window: tuple[int, int]) -> dict:
    accepted = selected_calls(result)
    accepted = accepted[origin_mask(accepted, window)]
    if accepted.empty:
        return {
            "months": 0, "calls": 0, "hits": 0, "accuracy": float("nan"),
            "monthly_mean": float("nan"), "monthly_median": float("nan"),
            "monthly_std": float("nan"), "up_calls": 0, "down_calls": 0,
        }
    monthly = accepted.groupby("origin_position")["correct"].mean()
    return {
        "months": int(accepted["origin_position"].nunique()),
        "calls": int(len(accepted)),
        "hits": int(accepted["correct"].sum()),
        "accuracy": float(accepted["correct"].mean()),
        "monthly_mean": float(monthly.mean()),
        "monthly_median": float(monthly.median()),
        "monthly_std": float(monthly.std(ddof=1)) if len(monthly) > 1 else 0.0,
        "up_calls": int(accepted["predicted_direction"].eq("Up").sum()),
        "down_calls": int(accepted["predicted_direction"].eq("Down").sum()),
    }


# ---------------------------------------------------------------------------
# Raw directional model metrics (before graph / prior blending)
# ---------------------------------------------------------------------------


def raw_metrics(
    signals: pd.DataFrame,
    probability_column: str,
    window: tuple[int, int],
) -> dict:
    rows = signals[origin_mask(signals, window) & signals["y_true"].notna()]
    if rows.empty:
        return {}
    y = rows["y_true"].astype(int).to_numpy()
    p = np.clip(
        pd.to_numeric(rows[probability_column], errors="coerce").to_numpy(dtype=float),
        1e-6,
        1.0 - 1e-6,
    )
    monthly = (
        pd.DataFrame({"origin_position": rows["origin_position"].to_numpy(),
                      "hit": (p >= 0.5).astype(int) == y})
        .groupby("origin_position")["hit"].mean()
    )
    return {
        "rows": int(len(rows)),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(p.mean()),
        "base_rate": float(y.mean()),
        "monthly_mean": float(monthly.mean()),
        "monthly_std": float(monthly.std(ddof=1)) if len(monthly) > 1 else 0.0,
    }


def calibration_table(
    signals: pd.DataFrame,
    probability_column: str,
    window: tuple[int, int],
    bins: int = 5,
) -> pd.DataFrame:
    rows = signals[origin_mask(signals, window) & signals["y_true"].notna()].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["p"] = pd.to_numeric(rows[probability_column], errors="coerce")
    rows["bucket"] = pd.qcut(rows["p"], bins, duplicates="drop")
    grouped = rows.groupby("bucket", observed=True).agg(
        rows=("p", "size"),
        mean_probability=("p", "mean"),
        observed_rate=("y_true", "mean"),
    ).reset_index()
    grouped["bucket"] = grouped["bucket"].astype(str)
    return grouped


# ---------------------------------------------------------------------------
# Paired, time-aware comparison
# ---------------------------------------------------------------------------


def monthly_accuracy(result: pd.DataFrame, window: tuple[int, int]) -> pd.Series:
    accepted = selected_calls(result)
    accepted = accepted[origin_mask(accepted, window)]
    return accepted.groupby("origin_position")["correct"].mean()


def block_bootstrap_interval(
    values: np.ndarray,
    block_months: int,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_months), len(values)))
    samples = []
    for _ in range(max(50, int(replicates))):
        starts = rng.integers(
            0, len(values), size=max(1, int(np.ceil(len(values) / block)))
        )
        sample = np.concatenate([
            np.take(values, np.arange(start, start + block) % len(values))
            for start in starts
        ])[:len(values)]
        samples.append(float(sample.mean()))
    return float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))


def paired_comparison(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    window: tuple[int, int],
    block_months: int,
    replicates: int,
    seed: int,
) -> dict:
    """Paired monthly accuracy difference with a circular block bootstrap."""
    base = monthly_accuracy(baseline, window)
    cand = monthly_accuracy(candidate, window)
    joined = pd.concat([base.rename("base"), cand.rename("cand")], axis=1).dropna()
    if joined.empty:
        return {}
    difference = (joined["cand"] - joined["base"]).to_numpy(dtype=float)
    low, high = block_bootstrap_interval(difference, block_months, replicates, seed)
    return {
        "months": int(len(difference)),
        "mean_monthly_difference": float(difference.mean()),
        "median_monthly_difference": float(np.median(difference)),
        "months_better": int((difference > 0).sum()),
        "months_worse": int((difference < 0).sum()),
        "months_equal": int((difference == 0).sum()),
        "block_bootstrap_p05": low,
        "block_bootstrap_p95": high,
        "interval_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


# ---------------------------------------------------------------------------
# Selection overlap
# ---------------------------------------------------------------------------


def selection_overlap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    window: tuple[int, int],
) -> tuple[pd.DataFrame, dict]:
    """Per-origin overlap plus the correctness of added and removed calls."""
    base = selected_calls(baseline)
    base = base[origin_mask(base, window)]
    cand = selected_calls(candidate)
    cand = cand[origin_mask(cand, window)]
    truth = pd.concat([
        base[["origin_position", "indicator_id", "y_true"]],
        cand[["origin_position", "indicator_id", "y_true"]],
    ]).drop_duplicates(["origin_position", "indicator_id"])
    truth = truth.set_index(["origin_position", "indicator_id"])["y_true"]

    records = []
    added_correct = added_total = removed_correct = removed_total = 0
    for origin in sorted(set(base["origin_position"]) | set(cand["origin_position"])):
        base_set = set(
            base.loc[base["origin_position"].eq(origin), "indicator_id"]
        )
        cand_set = set(
            cand.loc[cand["origin_position"].eq(origin), "indicator_id"]
        )
        shared = base_set & cand_set
        added = sorted(cand_set - base_set)
        removed = sorted(base_set - cand_set)
        for indicator in added:
            label = truth.get((origin, indicator))
            if pd.notna(label):
                added_total += 1
                added_correct += int(int(label) == 1)
        for indicator in removed:
            label = truth.get((origin, indicator))
            if pd.notna(label):
                removed_total += 1
                removed_correct += int(int(label) == 1)
        records.append({
            "origin_position": int(origin),
            "baseline_calls": len(base_set),
            "candidate_calls": len(cand_set),
            "overlap_count": len(shared),
            "overlap_pct": (
                len(shared) / len(base_set) if base_set else float("nan")
            ),
            "added": ",".join(added),
            "removed": ",".join(removed),
        })
    frame = pd.DataFrame(records)
    summary = {
        "months": int(len(frame)),
        "mean_overlap_count": float(frame["overlap_count"].mean()) if len(frame) else float("nan"),
        "mean_overlap_pct": float(frame["overlap_pct"].mean()) if len(frame) else float("nan"),
        "added_calls": int(added_total),
        "added_correct": int(added_correct),
        "added_accuracy": (
            float(added_correct / added_total) if added_total else float("nan")
        ),
        "removed_calls": int(removed_total),
        "removed_correct": int(removed_correct),
        "removed_accuracy": (
            float(removed_correct / removed_total) if removed_total else float("nan")
        ),
        "net_hit_change": int(added_correct - removed_correct),
    }
    return frame, summary


# ---------------------------------------------------------------------------
# Per-indicator analysis
# ---------------------------------------------------------------------------


def per_indicator_table(
    signals: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    window: tuple[int, int],
    probability_columns: dict[str, str],
) -> pd.DataFrame:
    rows = signals[origin_mask(signals, window) & signals["y_true"].notna()].copy()
    table = rows.groupby("indicator_id").agg(
        eligible_rows=("y_true", "size"),
        up_rate=("y_true", "mean"),
        mean_local_training_rows=("local_training_rows", "mean"),
    ).reset_index()
    y = rows["y_true"].astype(int)
    for name, column in probability_columns.items():
        hit = (
            pd.to_numeric(rows[column], errors="coerce") >= 0.5
        ).astype(int).eq(y).astype(int)
        table = table.merge(
            rows.assign(hit=hit).groupby("indicator_id")["hit"].mean().rename(
                f"raw_accuracy_{name}"
            ).reset_index(),
            on="indicator_id",
            how="left",
        )
    for name, result in results.items():
        accepted = selected_calls(result)
        accepted = accepted[origin_mask(accepted, window)]
        stats = accepted.groupby("indicator_id")["correct"].agg(["size", "sum"])
        stats.columns = [f"selected_{name}", f"selected_hits_{name}"]
        stats[f"selected_accuracy_{name}"] = (
            stats[f"selected_hits_{name}"] / stats[f"selected_{name}"]
        )
        table = table.merge(stats.reset_index(), on="indicator_id", how="left")
    order = table["indicator_id"].str.removeprefix("X").astype(int)
    return table.assign(_order=order).sort_values("_order").drop(columns="_order")
