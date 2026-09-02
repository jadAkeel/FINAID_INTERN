from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .correctness_calibration import apply_correctness_semantics
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .directional_downside import (
    apply_bidirectional_selector,
    build_directional_downside_features,
    fit_directional_downside_model,
    predict_directional_downside,
    summarize_bidirectional_predictions,
)
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .schemas import validate_oof_columns
from .uptrend_pipeline import (
    ROOT,
    _prepare,
    active_model_artifact,
    build_active_model,
    build_uptrend_predictions,
)
from .validation import (
    assert_target_history_available,
    causal_training_rows,
    latest_available_target_origin,
)


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def directional_downside_experiment_root(root: Path = ROOT) -> Path:
    return root / "research/directional_downside_selector"


def directional_downside_probabilities_artifact(root: Path = ROOT) -> Path:
    return directional_downside_experiment_root(root) / (
        "artifacts/downside_probabilities.parquet"
    )


def directional_downside_predictions_artifact(root: Path = ROOT) -> Path:
    return directional_downside_experiment_root(root) / (
        "artifacts/predictions.parquet"
    )


def directional_downside_summary_path(root: Path = ROOT) -> Path:
    return directional_downside_experiment_root(root) / "metrics/summary.json"


def _configuration_hash(root: Path) -> str:
    payload = {
        "project": _read_yaml(root / "configs/config.yaml"),
        "uptrend_model": _read_yaml(root / "configs/uptrend_model.yaml"),
        "directional_downside": _read_yaml(
            root / "configs/directional_downside_model.yaml"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_panel(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    frame, _, base_panel, config, _ = _prepare(root)
    settings = _read_yaml(root / "configs/directional_downside_model.yaml")
    feature_settings = settings["features"]
    features = build_directional_downside_features(
        frame,
        availability_lag=int(settings["availability_lag_months"]),
        lead_correlation_window=int(
            feature_settings["lead_correlation_window"]
        ),
        lead_minimum_pairs=int(feature_settings["lead_minimum_pairs"]),
        lead_top_k=int(feature_settings["lead_top_k"]),
    )
    panel = features.merge(
        base_panel[[
            "origin_position",
            "indicator_id",
            "origin_date",
            "target_date",
            "y_true",
            "eligible",
            "data_quality_ok",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    panel["down_target"] = 1.0 - panel["y_true"]
    panel["down_eligible"] = (
        panel["eligible"].fillna(False).astype(bool)
        & panel["down_target"].notna()
    )
    return panel, config, settings


def build_directional_downside_probabilities(
    root: Path = ROOT,
) -> pd.DataFrame:
    """Generate global, local, and pattern Down probabilities walk-forward."""
    panel, config, settings = _build_panel(root)
    tuning_start, _ = (int(value) for value in settings["tuning_origins"])
    _, confirmation_end = (
        int(value) for value in settings["confirmation_origins"]
    )
    lag = int(settings["availability_lag_months"])
    model_settings = settings["model"]
    eligible = panel[panel["down_eligible"]].copy()
    data_hash = sha256_file(root / config["data_path"])
    configuration_hash = _configuration_hash(root)
    pieces = []
    for origin in range(tuning_start, confirmation_end + 1):
        train = causal_training_rows(
            eligible,
            origin,
            availability_lag=lag,
        )
        test = eligible[eligible["origin_position"].eq(origin)].copy()
        assert_target_history_available(
            train,
            origin,
            availability_lag=lag,
        )
        started = time.perf_counter()
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
            trailing_prior_window=int(
                model_settings["trailing_prior_window"]
            ),
            minimum_pattern_rows=int(model_settings["minimum_pattern_rows"]),
        )
        current = test[[
            "origin_position",
            "origin_date",
            "target_date",
            "indicator_id",
            "y_true",
            "down_target",
            "down_exhaustion_flag",
            "down_lead_peer_score",
        ]].merge(
            probabilities,
            on=["origin_position", "indicator_id"],
            how="left",
            validate="one_to_one",
        )
        current["down_fit_through_origin"] = latest_available_target_origin(
            origin,
            lag,
        )
        current["run_id"] = "directional_downside_probability_research"
        current["model_id"] = settings["experiment_id"]
        current["model_version"] = settings["experiment_release"]
        current["data_hash"] = data_hash
        current["config_hash"] = configuration_hash
        current["locked_evaluation_read"] = False
        current["runtime_seconds"] = time.perf_counter() - started
        pieces.append(current)
    result = pd.concat(pieces, ignore_index=True)
    if (
        result["down_fit_through_origin"]
        > result["origin_position"] - lag - 1
    ).any():
        raise AssertionError("Directional downside model used unavailable targets")
    probability_columns = [
        "p_down_global",
        "p_down_local",
        "p_down_pattern",
        "p_down_indicator_prior",
    ]
    if not result[probability_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        raise AssertionError("Directional downside probabilities are invalid")
    return result


def _load_or_build_downside_probabilities(root: Path) -> pd.DataFrame:
    """Reuse probabilities only when their data and configuration match."""
    path = directional_downside_probabilities_artifact(root)
    config = _read_yaml(root / "configs/config.yaml")
    settings = _read_yaml(root / "configs/directional_downside_model.yaml")
    expected_data_hash = sha256_file(root / config["data_path"])
    expected_config_hash = _configuration_hash(root)
    tuning_start, _ = (int(value) for value in settings["tuning_origins"])
    _, confirmation_end = (
        int(value) for value in settings["confirmation_origins"]
    )
    if path.exists():
        try:
            cached = pd.read_parquet(path)
            required = {
                "data_hash",
                "config_hash",
                "locked_evaluation_read",
                "down_fit_through_origin",
            }
            valid = (
                required.issubset(cached.columns)
                and set(cached["data_hash"].dropna().astype(str).unique())
                == {expected_data_hash}
                and set(cached["config_hash"].dropna().astype(str).unique())
                == {expected_config_hash}
                and int(cached["origin_position"].min()) == tuning_start
                and int(cached["origin_position"].max()) == confirmation_end
                and not cached["locked_evaluation_read"].fillna(True).any()
            )
            if valid:
                return cached
        except (KeyError, OSError, ValueError):
            pass
    probabilities = build_directional_downside_probabilities(root)
    atomic_write_parquet(probabilities, path)
    return probabilities


def _base_predictions(root: Path, settings: dict) -> pd.DataFrame:
    active = active_model_artifact(root)
    if not active.exists():
        build_active_model(root)
    development = pd.read_parquet(active)
    confirmation_start, confirmation_end = (
        int(value) for value in settings["confirmation_origins"]
    )
    confirmation = build_uptrend_predictions(
        root,
        origin_range=(confirmation_start, confirmation_end),
    )
    result = pd.concat([development, confirmation], ignore_index=True)
    validate_oof_columns(result.columns.tolist())
    return result


def _window_summary(
    predictions: pd.DataFrame,
    bounds: list[int],
) -> dict[str, float | int]:
    start, end = (int(value) for value in bounds)
    return summarize_bidirectional_predictions(
        predictions[predictions["origin_position"].between(start, end)]
    )


def _probability_metrics(
    predictions: pd.DataFrame,
    bounds: list[int],
    probability_column: str,
) -> dict[str, float | int]:
    start, end = (int(value) for value in bounds)
    current = predictions[
        predictions["origin_position"].between(start, end)
        & predictions["down_target"].notna()
        & predictions[probability_column].notna()
    ]
    target = current["down_target"].astype(int)
    probability = current[probability_column].astype(float)
    return {
        "rows": int(len(current)),
        "down_events": int(target.sum()),
        "prevalence": float(target.mean()),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(
            average_precision_score(target, probability)
        ),
        "brier": float(brier_score_loss(target, probability)),
    }


def _paired_block_delta(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    bounds: list[int],
    block_months: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    start, end = (int(value) for value in bounds)

    def monthly_accuracy(frame: pd.DataFrame) -> pd.Series:
        selected = frame[
            frame["origin_position"].between(start, end)
            & frame["accepted"].fillna(False).astype(bool)
            & frame["y_true"].notna()
        ].copy()
        selected["correct"] = selected["predicted_direction"].eq("Up").astype(
            int
        ).eq(selected["y_true"].astype(int))
        return selected.groupby("origin_position")["correct"].mean()

    base_monthly = monthly_accuracy(base)
    candidate_monthly = monthly_accuracy(candidate).reindex(base_monthly.index)
    difference = (candidate_monthly - base_monthly).to_numpy(dtype=float)
    block = max(1, min(block_months, len(difference)))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(100, replicates)):
        starts = rng.integers(
            0,
            len(difference),
            size=max(1, int(np.ceil(len(difference) / block))),
        )
        sample = np.concatenate([
            np.take(
                difference,
                np.arange(start_index, start_index + block) % len(difference),
            )
            for start_index in starts
        ])[:len(difference)]
        samples.append(float(sample.mean()))
    return {
        "delta_bootstrap_p10": float(np.quantile(samples, 0.10)),
        "delta_bootstrap_median": float(np.quantile(samples, 0.50)),
        "delta_bootstrap_p90": float(np.quantile(samples, 0.90)),
    }


def _write_report(
    summary: dict,
    candidate_search: pd.DataFrame,
    root: Path,
) -> None:
    experiment_root = directional_downside_experiment_root(root)
    metrics = experiment_root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    candidate_search.to_csv(metrics / "candidate_search.csv", index=False)
    atomic_write_json(summary, directional_downside_summary_path(root))
    selected = summary["selected_parameters"]
    lines = [
        "# Directional Downside Selector",
        "",
        "This experiment learns actual Down directions and can include them in the monthly top 15.",
        "",
        "## Frozen design",
        "",
        "- Global regularized logistic model across all indicators.",
        "- Local regularized logistic model per indicator when history is sufficient.",
        "- Indicator-specific rise-then-stall pattern prior.",
        "- Rolling learned lead-lag peer features; no indicator meanings are assumed.",
        "- Candidate selection uses Tuning origins 120-179 only.",
        "- Validation is 180-219 and Confirmation is 220-266.",
        "- Historical locked evidence 268-315 was not read.",
        "",
        "## Selected parameters",
        "",
        f"- Local weight: `{selected['local_weight']}`",
        f"- Pattern weight: `{selected['pattern_weight']}`",
        f"- Down threshold: `{selected['down_threshold']}`",
        f"- Down margin: `{selected['down_margin']}`",
        "",
        "## Results",
        "",
        f"- Tuning base / candidate: `{summary['tuning_base']['accuracy']:.4%}` / `{summary['tuning_candidate']['accuracy']:.4%}`",
        f"- Validation base / candidate: `{summary['validation_base']['accuracy']:.4%}` / `{summary['validation_candidate']['accuracy']:.4%}`",
        f"- Confirmation base / candidate: `{summary['confirmation_base']['accuracy']:.4%}` / `{summary['confirmation_candidate']['accuracy']:.4%}`",
        f"- Confirmation Down calls / hits: `{summary['confirmation_candidate']['down_calls']} / {summary['confirmation_candidate']['down_hits']}`",
        f"- Promotion eligible: `{summary['promotion_eligible']}`",
        "",
        "This experiment never changes the active model automatically.",
    ]
    (experiment_root / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_directional_downside_selector(root: Path = ROOT) -> Path:
    """Fit, tune, and evaluate the bidirectional top-15 experiment."""
    settings = _read_yaml(root / "configs/directional_downside_model.yaml")
    config = _read_yaml(root / "configs/config.yaml")
    probabilities = _load_or_build_downside_probabilities(root)
    base = _base_predictions(root, settings)
    tuning_start, tuning_end = (
        int(value) for value in settings["tuning_origins"]
    )
    tuning_base = base[base["origin_position"].between(
        tuning_start,
        tuning_end,
    )]
    tuning_probabilities = probabilities[
        probabilities["origin_position"].between(tuning_start, tuning_end)
    ]
    selection = settings["selection"]
    rows = []
    for blend in settings["model"]["blend_grid"]:
        for threshold in selection["down_threshold_grid"]:
            for margin in selection["down_margin_grid"]:
                candidate = apply_bidirectional_selector(
                    tuning_base,
                    tuning_probabilities,
                    local_weight=float(blend["local_weight"]),
                    pattern_weight=float(blend["pattern_weight"]),
                    down_threshold=float(threshold),
                    down_margin=float(margin),
                    cap=int(selection["monthly_selection_count"]),
                )
                result = summarize_bidirectional_predictions(candidate)
                row_id = len(rows)
                rows.append({
                    "candidate_id": row_id,
                    "local_weight": float(blend["local_weight"]),
                    "pattern_weight": float(blend["pattern_weight"]),
                    "down_threshold": float(threshold),
                    "down_margin": float(margin),
                    "minimum_down_calls_met": (
                        int(result["down_calls"])
                        >= int(selection["minimum_tuning_down_calls"])
                    ),
                    **result,
                })
    search = pd.DataFrame(rows)
    eligible = search[search["minimum_down_calls_met"]].copy()
    if eligible.empty:
        raise RuntimeError("No tuning candidate made enough Down calls")
    eligible = eligible.sort_values(
        ["accuracy", "down_accuracy", "down_calls", "local_weight"],
        ascending=[False, False, False, True],
    )
    selected_row = eligible.iloc[0]
    selected_parameters = {
        "local_weight": float(selected_row["local_weight"]),
        "pattern_weight": float(selected_row["pattern_weight"]),
        "down_threshold": float(selected_row["down_threshold"]),
        "down_margin": float(selected_row["down_margin"]),
    }
    final = apply_bidirectional_selector(
        base,
        probabilities,
        **selected_parameters,
        cap=int(selection["monthly_selection_count"]),
    )
    final["run_id"] = "directional_downside_selector_research"
    final["model_id"] = settings["experiment_id"]
    final["model_version"] = settings["experiment_release"]
    final["parameters_selected_on"] = "tuning_120_179"
    final["locked_evaluation_read"] = False
    final = apply_correctness_semantics(final)
    validate_oof_columns(final.columns.tolist())
    selected_counts = final[final["accepted"]].groupby(
        "origin_position"
    )["indicator_id"].agg(["count", "nunique"])
    cap = int(selection["monthly_selection_count"])
    if not selected_counts.eq(cap).all().all():
        raise AssertionError("Directional selector must choose 15 unique indicators")

    probability_local_weight = selected_parameters["local_weight"]
    probability_pattern_weight = selected_parameters["pattern_weight"]
    probabilities["p_down_selected"] = (
        (1.0 - probability_local_weight - probability_pattern_weight)
        * probabilities["p_down_global"]
        + probability_local_weight * probabilities["p_down_local"]
        + probability_pattern_weight * probabilities["p_down_pattern"]
    )
    tuning_base_summary = _window_summary(base, settings["tuning_origins"])
    tuning_summary = _window_summary(final, settings["tuning_origins"])
    validation_base_summary = _window_summary(
        base,
        settings["validation_origins"],
    )
    validation_summary = _window_summary(
        final,
        settings["validation_origins"],
    )
    confirmation_base_summary = _window_summary(
        base,
        settings["confirmation_origins"],
    )
    confirmation_summary = _window_summary(
        final,
        settings["confirmation_origins"],
    )
    bootstrap = _paired_block_delta(
        base,
        final,
        settings["confirmation_origins"],
        block_months=int(config["bootstrap_blocks"]),
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    )
    promotion = settings["promotion"]
    validation_delta = (
        float(validation_summary["accuracy"])
        - float(validation_base_summary["accuracy"])
    )
    confirmation_delta = (
        float(confirmation_summary["accuracy"])
        - float(confirmation_base_summary["accuracy"])
    )
    validation_ok = (
        validation_delta > 0
        if promotion["require_positive_validation_delta"]
        else True
    )
    confirmation_ok = (
        confirmation_delta > 0
        if promotion["require_positive_confirmation_delta"]
        else True
    )
    bootstrap_ok = (
        float(bootstrap["delta_bootstrap_p10"]) > 0
        if promotion["require_positive_bootstrap_p10"]
        else True
    )
    promotion_eligible = bool(
        validation_ok
        and confirmation_ok
        and int(confirmation_summary["down_calls"]) > 0
        and float(confirmation_summary["down_accuracy"])
        >= float(promotion["minimum_down_accuracy"])
        and bootstrap_ok
    )
    summary = {
        "experiment_id": settings["experiment_id"],
        "experiment_name": settings["experiment_name"],
        "experiment_release": settings["experiment_release"],
        "selected_parameters": selected_parameters,
        "parameters_selected_on": "tuning_120_179",
        "tuning_base": tuning_base_summary,
        "tuning_candidate": tuning_summary,
        "validation_base": validation_base_summary,
        "validation_candidate": validation_summary,
        "validation_accuracy_delta": validation_delta,
        "confirmation_base": confirmation_base_summary,
        "confirmation_candidate": confirmation_summary,
        "confirmation_accuracy_delta": confirmation_delta,
        "tuning_probability_metrics": _probability_metrics(
            probabilities,
            settings["tuning_origins"],
            "p_down_selected",
        ),
        "validation_probability_metrics": _probability_metrics(
            probabilities,
            settings["validation_origins"],
            "p_down_selected",
        ),
        "confirmation_probability_metrics": _probability_metrics(
            probabilities,
            settings["confirmation_origins"],
            "p_down_selected",
        ),
        **bootstrap,
        "promotion_eligible": promotion_eligible,
        "active_model_changed": False,
        "confirmation_read": True,
        "locked_evaluation_read": False,
        "locked_origins": list(settings["locked_origins"]),
        "config_hash": _configuration_hash(root),
        "data_hash": sha256_file(root / config["data_path"]),
    }
    atomic_write_parquet(
        final,
        directional_downside_predictions_artifact(root),
    )
    search = search.sort_values(
        ["minimum_down_calls_met", "accuracy", "down_accuracy"],
        ascending=[False, False, False],
    )
    _write_report(summary, search, root)
    return directional_downside_summary_path(root)


def directional_downside_status(root: Path = ROOT) -> dict:
    path = directional_downside_summary_path(root)
    if not path.exists():
        return {
            "ready": False,
            "summary": path.relative_to(root).as_posix(),
        }
    with path.open(encoding="utf-8") as handle:
        return {"ready": True, **json.load(handle)}
