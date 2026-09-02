from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .correctness_calibration import (
    apply_correctness_semantics,
    decision_correctness,
    wilson_lower_bound,
)
from .io import atomic_write_json, atomic_write_parquet
from .regime_adaptive_pipeline import regime_adaptive_predictions_artifact
from .uptrend_pipeline import ROOT, active_model_artifact


def correctness_calibration_root(root: Path = ROOT) -> Path:
    return root / "research/correctness_calibration_audit"


def correctness_calibration_summary_path(root: Path = ROOT) -> Path:
    return correctness_calibration_root(root) / "metrics/summary.json"


def _prepare(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    upgraded = apply_correctness_semantics(predictions)
    selected = upgraded[
        upgraded["accepted"].fillna(False).astype(bool)
        & upgraded["y_true"].notna()
    ].copy()
    selected["correct"] = decision_correctness(selected).astype(int)
    selected["model"] = model
    selected["rank_group"] = pd.cut(
        selected["selection_rank"],
        bins=[0, 3, 5, 10, 15, np.inf],
        labels=["1_3", "4_5", "6_10", "11_15", "16_plus"],
    ).astype(str)
    selected["legacy_probability"] = pd.to_numeric(
        selected["legacy_correctness_probability"], errors="coerce"
    )
    return selected.sort_values(
        ["origin_position", "indicator_id"]
    ).reset_index(drop=True)


def _ece(y_true: np.ndarray, probability: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.digitize(probability, edges[1:-1]), 0, 9)
    return float(sum(
        np.mean(bins == index)
        * abs(
            float(np.mean(probability[bins == index]))
            - float(np.mean(y_true[bins == index]))
        )
        for index in range(10)
        if np.any(bins == index)
    ))


def _wilson_interval(
    hits: int,
    calls: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    if calls <= 0:
        return np.nan, np.nan
    rate = hits / calls
    denominator = 1.0 + z_value**2 / calls
    centre = (rate + z_value**2 / (2.0 * calls)) / denominator
    radius = (
        z_value
        * np.sqrt(
            rate * (1.0 - rate) / calls
            + z_value**2 / (4.0 * calls**2)
        )
        / denominator
    )
    return float(centre - radius), float(centre + radius)


def _probability_metrics(
    y_true: pd.Series,
    probability: pd.Series,
) -> dict[str, float | int]:
    valid = probability.notna()
    y = y_true[valid].to_numpy(dtype=int)
    p = np.clip(
        probability[valid].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    if not len(y):
        return {"calls": 0}
    intercept = np.nan
    slope = np.nan
    if len(np.unique(y)) == 2 and float(np.std(p)) > 1e-8:
        calibration = LogisticRegression(C=1e6, solver="lbfgs")
        calibration.fit(
            np.log(p / (1.0 - p)).reshape(-1, 1), y
        )
        intercept = float(calibration.intercept_[0])
        slope = float(calibration.coef_[0, 0])
    return {
        "calls": int(len(y)),
        "accuracy": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "auc": (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": _ece(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "spearman": float(
            pd.Series(p).corr(pd.Series(y), method="spearman")
        ),
    }


def _causal_group_probability(
    frame: pd.DataFrame,
    groups: list[str],
    shrinkage: float = 50.0,
    minimum_history_months: int = 12,
) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for origin, positions in frame.groupby(
        "origin_position", sort=True
    ).groups.items():
        history = frame[frame["origin_position"].le(int(origin) - 2)]
        if history["origin_position"].nunique() < minimum_history_months:
            continue
        global_mean = (1.0 + history["correct"].sum()) / (2.0 + len(history))
        current = frame.loc[list(positions)]
        if not groups:
            output.loc[current.index] = global_mean
            continue
        grouped = history.groupby(groups, observed=True)["correct"].agg(
            ["sum", "count"]
        )
        for index, row in current.iterrows():
            key = tuple(row[column] for column in groups)
            key = key[0] if len(groups) == 1 else key
            if key in grouped.index:
                hits = float(grouped.loc[key, "sum"])
                calls = float(grouped.loc[key, "count"])
            else:
                hits = 0.0
                calls = 0.0
            output.at[index] = (
                hits + shrinkage * global_mean
            ) / (calls + shrinkage)
    return output


def _causal_platt_or_isotonic(
    frame: pd.DataFrame,
    method: str,
    minimum_history_months: int = 12,
) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    score = pd.to_numeric(frame["legacy_probability"], errors="coerce")
    rank = (pd.to_numeric(frame["selection_rank"], errors="coerce") - 1) / 14
    for origin, positions in frame.groupby(
        "origin_position", sort=True
    ).groups.items():
        history_mask = frame["origin_position"].le(int(origin) - 2)
        if frame.loc[history_mask, "origin_position"].nunique() < minimum_history_months:
            continue
        history_valid = history_mask & score.notna() & rank.notna()
        if frame.loc[history_valid, "correct"].nunique() < 2:
            continue
        current = frame.loc[list(positions)]
        current_score = score.loc[current.index]
        current_rank = rank.loc[current.index]
        valid = current_score.notna() & current_rank.notna()
        if method == "platt":
            features = pd.DataFrame({
                "score": score[history_valid],
                "rank": rank[history_valid],
            })
            current_features = pd.DataFrame({
                "score": current_score[valid],
                "rank": current_rank[valid],
            })
            model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
            model.fit(features, frame.loc[history_valid, "correct"])
            output.loc[current.index[valid]] = model.predict_proba(
                current_features
            )[:, 1]
        elif method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(
                score[history_valid], frame.loc[history_valid, "correct"]
            )
            output.loc[current.index[valid]] = model.predict(
                current_score[valid]
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    return output


def _reliability_rows(
    frame: pd.DataFrame,
    method: str,
    probability: pd.Series,
) -> list[dict]:
    edges = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
    labels = ["<55%", "55-60%", "60-65%", "65-70%", "70-75%", ">=75%"]
    bucket = pd.cut(
        probability,
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows = []
    for label in labels:
        current = frame[bucket.eq(label) & probability.notna()]
        if current.empty:
            continue
        calls = len(current)
        hits = int(current["correct"].sum())
        interval_low, interval_high = _wilson_interval(hits, calls)
        rows.append({
            "model": str(frame["model"].iloc[0]),
            "method": method,
            "bucket": label,
            "calls": calls,
            "average_probability": float(probability.loc[current.index].mean()),
            "observed_accuracy": hits / calls,
            "observed_wilson_ci95_low": interval_low,
            "observed_wilson_ci95_high": interval_high,
            "observed_wilson_lcb_95_one_sided": wilson_lower_bound(
                hits, calls
            ),
        })
    return rows


def _slice_rows(
    frame: pd.DataFrame,
    probability: pd.Series,
) -> list[dict]:
    slices: list[tuple[str, pd.Series]] = []
    for column in ["regime_label", "rank_group", "indicator_id", "predicted_direction"]:
        if column not in frame:
            continue
        for value in frame[column].dropna().astype(str).unique():
            slices.append((f"{column}:{value}", frame[column].astype(str).eq(value)))
    windows = {
        "tuning_early": (120, 149),
        "tuning_late": (150, 179),
        "validation": (180, 219),
        "confirmation": (220, 266),
    }
    for name, (start, end) in windows.items():
        slices.append((f"window:{name}", frame["origin_position"].between(start, end)))
    rows = []
    for label, mask in slices:
        metric = _probability_metrics(frame.loc[mask, "correct"], probability[mask])
        rows.append({
            "model": str(frame["model"].iloc[0]),
            "slice": label,
            **metric,
        })
    return rows


def _coverage_rows(
    frame: pd.DataFrame,
    method: str,
    probability: pd.Series,
) -> list[dict]:
    rows = []
    windows = {
        "all": pd.Series(True, index=frame.index),
        "tuning_early": frame["origin_position"].between(120, 149),
        "tuning_late": frame["origin_position"].between(150, 179),
        "validation": frame["origin_position"].between(180, 219),
        "confirmation": frame["origin_position"].between(220, 266),
    }
    for window, window_mask in windows.items():
        available = window_mask & probability.notna()
        denominator = int(available.sum())
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            retained = available & probability.ge(threshold)
            current = frame[retained]
            rows.append({
                "model": str(frame["model"].iloc[0]),
                "method": method,
                "window": window,
                "threshold": threshold,
                "calls": int(len(current)),
                "coverage": (
                    float(len(current) / denominator)
                    if denominator
                    else np.nan
                ),
                "accuracy": (
                    float(current["correct"].mean())
                    if len(current)
                    else np.nan
                ),
            })
    return rows


def _bootstrap_metrics(
    frame: pd.DataFrame,
    probability: pd.Series,
    seed: int,
    replicates: int = 500,
    block_months: int = 6,
) -> dict[str, float]:
    valid = probability.notna()
    current = frame[valid].copy()
    current["_probability"] = probability[valid]
    origins = np.sort(current["origin_position"].unique())
    if not len(origins):
        return {
            f"{name}_{quantile}": np.nan
            for name in ["auc", "brier", "ece"]
            for quantile in ["p05", "p95"]
        }
    rng = np.random.default_rng(seed)
    samples = {"auc": [], "brier": [], "ece": []}
    blocks = max(1, int(np.ceil(len(origins) / block_months)))
    monthly_arrays = {
        origin: (
            group["correct"].to_numpy(dtype=int),
            group["_probability"].to_numpy(dtype=float),
        )
        for origin, group in current.groupby("origin_position", sort=False)
    }
    for _ in range(replicates):
        starts = rng.integers(0, len(origins), size=blocks)
        sampled_origins = np.concatenate([
            np.take(origins, np.arange(start, start + block_months) % len(origins))
            for start in starts
        ])[:len(origins)]
        y = np.concatenate([
            monthly_arrays[origin][0] for origin in sampled_origins
        ])
        p = np.concatenate([
            monthly_arrays[origin][1] for origin in sampled_origins
        ])
        samples["auc"].append(
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else np.nan
        )
        samples["brier"].append(float(brier_score_loss(y, p)))
        samples["ece"].append(_ece(y, p))
    return {
        f"{name}_p05": float(np.nanquantile(values, 0.05))
        for name, values in samples.items()
    } | {
        f"{name}_p95": float(np.nanquantile(values, 0.95))
        for name, values in samples.items()
    }


def build_correctness_calibration_audit(root: Path = ROOT) -> Path:
    input_paths = {
        "active": active_model_artifact(root),
        "adaptive": regime_adaptive_predictions_artifact(root),
    }
    if not all(path.exists() for path in input_paths.values()):
        raise FileNotFoundError("Active and adaptive prediction artifacts are required")
    raw = {
        model: pd.read_parquet(path)
        for model, path in input_paths.items()
    }
    adaptive_regime = raw["adaptive"][[
        "origin_position", "regime_label", "regime_stress"
    ]].drop_duplicates("origin_position")
    if "regime_label" not in raw["active"]:
        raw["active"] = raw["active"].merge(
            adaptive_regime,
            on="origin_position",
            how="left",
            validate="many_to_one",
        )
    frames = {
        model: _prepare(predictions, model)
        for model, predictions in raw.items()
    }
    comparison_rows = []
    reliability_rows = []
    slice_rows = []
    coverage_rows = []
    bootstrap_rows = []
    upgraded_paths = {}
    for model, frame in frames.items():
        causal_global = _causal_group_probability(frame, [])
        candidates = {
            "legacy_directional_score": frame["legacy_probability"],
            "legacy_directional_score_causal_window": (
                frame["legacy_probability"].where(causal_global.notna())
            ),
            "causal_global_cohort": causal_global,
            "causal_rank_beta": _causal_group_probability(
                frame, ["rank_group"]
            ),
            "causal_regime_beta": _causal_group_probability(
                frame, ["regime_label"]
            ),
            "causal_rank_regime_beta": _causal_group_probability(
                frame, ["rank_group", "regime_label"]
            ),
            "causal_platt_score_rank": _causal_platt_or_isotonic(
                frame, "platt"
            ),
            "causal_isotonic_score": _causal_platt_or_isotonic(
                frame, "isotonic"
            ),
        }
        for method, probability in candidates.items():
            comparison_rows.append({
                "model": model,
                "method": method,
                **_probability_metrics(frame["correct"], probability),
            })
            reliability_rows.extend(
                _reliability_rows(frame, method, probability)
            )
            coverage_rows.extend(_coverage_rows(frame, method, probability))
            bootstrap_rows.append({
                "model": model,
                "method": method,
                **_bootstrap_metrics(
                    frame,
                    probability,
                    seed=20260727,
                ),
            })
        slice_rows.extend(
            _slice_rows(frame, frame["legacy_probability"])
        )
        upgraded = apply_correctness_semantics(raw[model])
        experiment_root = correctness_calibration_root(root)
        artifact_path = experiment_root / f"artifacts/{model}_scored.parquet"
        atomic_write_parquet(upgraded, artifact_path)
        upgraded_paths[model] = artifact_path.relative_to(root).as_posix()

    experiment_root = correctness_calibration_root(root)
    metrics_root = experiment_root / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(comparison_rows)
    reliability = pd.DataFrame(reliability_rows)
    slices = pd.DataFrame(slice_rows)
    coverage = pd.DataFrame(coverage_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    comparison.to_csv(metrics_root / "model_comparison.csv", index=False)
    reliability.to_csv(metrics_root / "reliability_buckets.csv", index=False)
    slices.to_csv(metrics_root / "slice_metrics.csv", index=False)
    coverage.to_csv(metrics_root / "accuracy_coverage.csv", index=False)
    bootstrap.to_csv(metrics_root / "bootstrap_intervals.csv", index=False)

    legacy = comparison[
        comparison["method"].eq("legacy_directional_score")
    ].set_index("model")
    individual_causal = comparison[
        comparison["method"].isin([
            "causal_rank_beta",
            "causal_regime_beta",
            "causal_rank_regime_beta",
            "causal_platt_score_rank",
            "causal_isotonic_score",
        ])
    ]
    global_brier = comparison[
        comparison["method"].eq("causal_global_cohort")
    ].set_index("model")["brier"]
    best_individual_brier = individual_causal.groupby("model")["brier"].min()
    brier_improvement = global_brier - best_individual_brier
    summary = {
        "experiment_id": "correctness_calibration_audit",
        "locked_evaluation_read": False,
        "locked_evaluation_path_read": False,
        "decision": "no_individual_correctness_probability",
        "correctness_probability_status": (
            "unavailable_no_valid_oof_individual_calibrator"
        ),
        "reason": (
            "No causal candidate demonstrated useful stable discrimination; "
            "the cohort baseline is monitoring-only."
        ),
        "active_legacy_auc": float(legacy.loc["active", "auc"]),
        "adaptive_legacy_auc": float(legacy.loc["adaptive", "auc"]),
        "active_legacy_brier": float(legacy.loc["active", "brier"]),
        "adaptive_legacy_brier": float(legacy.loc["adaptive", "brier"]),
        "best_causal_auc": float(individual_causal["auc"].max()),
        "active_best_brier_improvement_vs_causal_constant": float(
            brier_improvement.loc["active"]
        ),
        "adaptive_best_brier_improvement_vs_causal_constant": float(
            brier_improvement.loc["adaptive"]
        ),
        "legacy_score_ranges": {
            model: {
                "minimum": float(frame["legacy_probability"].min()),
                "maximum": float(frame["legacy_probability"].max()),
            }
            for model, frame in frames.items()
        },
        "bootstrap": {
            "replicates": 500,
            "block_months": 6,
            "interval_quantiles": [0.05, 0.95],
            "seed": 20260727,
        },
        "individual_probability_release_gate": {
            "minimum_auc": 0.55,
            "minimum_brier_improvement_vs_constant": 0.002,
            "requires_stable_accuracy_coverage_by_time_window": True,
            "passed": False,
        },
        "upgraded_artifacts": upgraded_paths,
        "active_model_changed": False,
        "adaptive_selection_changed": False,
    }
    atomic_write_json(summary, correctness_calibration_summary_path(root))
    (experiment_root / "README.md").write_text(
        "\n".join([
            "# Correctness calibration audit",
            "",
            "This audit separates direction scores, ranking utility, and final-decision correctness.",
            "",
            f"- Active legacy correctness AUC: `{summary['active_legacy_auc']:.4f}`.",
            f"- Adaptive legacy correctness AUC: `{summary['adaptive_legacy_auc']:.4f}`.",
            f"- Best causal candidate AUC: `{summary['best_causal_auc']:.4f}`.",
            "- No evaluated causal calibrator produced stable useful individual-call discrimination.",
            "- `correctness_probability` and `correctness_lcb` are therefore unavailable rather than copied from directional score.",
            "- A causal marginal cohort rate and Wilson lower bound are retained for monitoring only.",
            "- The locked evaluation artifact was not read.",
            "",
            "## Field contract",
            "",
            "- `p_up`: estimated Up-direction score; not proven calibrated.",
            "- `selection_score`: ranking utility; not a correctness probability.",
            "- `directional_score`: uncalibrated strength of the chosen direction.",
            "- `correctness_probability`: null because no individual calibrator passed the release gate.",
            "- `cohort_correctness_probability`: causal Laplace-smoothed historical selected-call rate, for monitoring only.",
            "- `cohort_correctness_lcb`: one-sided 95% Wilson lower bound for the same cohort rate.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python -m forecast_select build-correctness-audit",
            "python -m forecast_select show-correctness-audit",
            "```",
            "",
            "The `metrics` directory contains candidate comparison, reliability, slice, accuracy-coverage, and temporal block-bootstrap tables.",
        ]) + "\n",
        encoding="utf-8",
    )
    return correctness_calibration_summary_path(root)


def correctness_calibration_status(root: Path = ROOT) -> dict:
    path = correctness_calibration_summary_path(root)
    if not path.exists():
        raise FileNotFoundError("Correctness calibration audit has not been built")
    return json.loads(path.read_text(encoding="utf-8"))
