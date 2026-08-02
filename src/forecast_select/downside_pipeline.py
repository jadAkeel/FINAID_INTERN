from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .downside_risk import (
    add_known_shock_features,
    apply_downside_risk_gate,
    build_downside_feature_panel,
    build_sudden_drop_labels,
    fit_downside_risk_model,
    predict_downside_probability,
    summarize_gate_predictions,
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


def _risk_configuration_hash(root: Path) -> str:
    payload = {
        "project": _read_yaml(root / "configs/config.yaml"),
        "uptrend_model": _read_yaml(root / "configs/uptrend_model.yaml"),
        "downside_risk_gate": _read_yaml(
            root / "configs/downside_risk_gate.yaml"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def downside_experiment_root(root: Path = ROOT) -> Path:
    return root / "research/downside_risk_gate"


def downside_risk_artifact(root: Path = ROOT) -> Path:
    return downside_experiment_root(root) / "artifacts/risk_predictions.parquet"


def gated_predictions_artifact(root: Path = ROOT) -> Path:
    return downside_experiment_root(root) / "artifacts/gated_predictions.parquet"


def downside_summary_path(root: Path = ROOT) -> Path:
    return downside_experiment_root(root) / "metrics/summary.json"


def _build_risk_panel(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    frame, targets, base_panel, config, _ = _prepare(root)
    settings = _read_yaml(root / "configs/downside_risk_gate.yaml")
    shock = settings["shock_definition"]
    labels = build_sudden_drop_labels(
        targets,
        trailing_window=int(shock["trailing_window"]),
        minimum_history=int(shock["minimum_history"]),
        lower_quantile=float(shock["lower_quantile"]),
        robust_z=float(shock["robust_z"]),
    )
    features = build_downside_feature_panel(frame, settings)
    panel = add_known_shock_features(
        features,
        labels,
        settings,
    ).merge(
        labels[[
            "origin_position",
            "indicator_id",
            "target_date",
            "target_return",
            "shock_lower_tail",
            "shock_robust_threshold",
            "shock_label_valid",
            "sudden_drop",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        base_panel[[
            "origin_position",
            "indicator_id",
            "origin_date",
            "y_true",
            "eligible",
            "data_quality_ok",
        ]],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    excluded = {str(value) for value in settings["excluded_indicators"]}
    panel["risk_eligible"] = (
        panel["eligible"].fillna(False).astype(bool)
        & ~panel["indicator_id"].isin(excluded)
    )
    return panel, config, settings


def build_downside_risk_predictions(root: Path = ROOT) -> pd.DataFrame:
    """Create walk-forward shock probabilities through Confirmation only."""
    panel, config, settings = _build_risk_panel(root)
    discovery_start, _ = (
        int(value) for value in settings["discovery_origins"]
    )
    _, confirmation_end = (
        int(value) for value in settings["confirmation_origins"]
    )
    lag = int(settings["availability_lag_months"])
    model_settings = settings["model"]
    training_rows = panel[
        panel["risk_eligible"]
        & panel["shock_label_valid"].fillna(False).astype(bool)
    ].copy()
    prediction_rows = panel[panel["risk_eligible"]].copy()
    data_hash = sha256_file(root / config["data_path"])
    configuration_hash = _risk_configuration_hash(root)
    parts = []
    for origin in range(discovery_start, confirmation_end + 1):
        train = causal_training_rows(
            training_rows,
            origin,
            availability_lag=lag,
        )
        test = prediction_rows[
            prediction_rows["origin_position"].eq(origin)
        ].copy()
        assert_target_history_available(
            train,
            origin,
            availability_lag=lag,
        )
        started = time.perf_counter()
        model = fit_downside_risk_model(
            train,
            seed=int(config["seed"]),
            logistic_c=float(model_settings["logistic_c"]),
            max_iter=int(model_settings["logistic_max_iter"]),
        )
        probability = predict_downside_probability(model, test)
        current = test[[
            "origin_position",
            "origin_date",
            "target_date",
            "indicator_id",
            "indicator_group",
            "y_true",
            "target_return",
            "shock_label_valid",
            "sudden_drop",
            "risk_eligible",
        ]].copy()
        current["run_id"] = "downside_risk_gate_research"
        current["model_id"] = settings["experiment_id"]
        current["model_version"] = settings["experiment_release"]
        current["p_sudden_drop"] = probability
        current["risk_fit_through_origin"] = (
            latest_available_target_origin(origin, lag)
        )
        current["fit_window"] = (
            f"<=position_{latest_available_target_origin(origin, lag)}"
        )
        current["data_hash"] = data_hash
        current["config_hash"] = configuration_hash
        current["seed"] = int(config["seed"])
        current["runtime_seconds"] = time.perf_counter() - started
        parts.append(current)
    result = pd.concat(parts, ignore_index=True)
    if (
        result["risk_fit_through_origin"]
        > result["origin_position"] - lag - 1
    ).any():
        raise AssertionError("Downside model used unavailable shock labels")
    if result["indicator_id"].isin(settings["excluded_indicators"]).any():
        raise AssertionError("Excluded indicators entered downside predictions")
    if not result["p_sudden_drop"].between(0, 1).all():
        raise AssertionError("Downside probabilities are invalid")
    return result


def _selection_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    selected["correct"] = selected["predicted_direction"].eq("Up").astype(
        int
    ).eq(selected["y_true"].astype(int))
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(selected["predicted_direction"].eq("Down").sum()),
    }


def _paired_block_delta(
    base_predictions: pd.DataFrame,
    gated_predictions: pd.DataFrame,
    block_months: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    def monthly_accuracy(frame: pd.DataFrame) -> pd.Series:
        selected = frame[
            frame["accepted"].fillna(False).astype(bool)
            & frame["y_true"].notna()
        ].copy()
        selected["correct"] = selected["predicted_direction"].eq("Up").astype(
            int
        ).eq(selected["y_true"].astype(int))
        return selected.groupby("origin_position")["correct"].mean()

    base = monthly_accuracy(base_predictions)
    gated = monthly_accuracy(gated_predictions)
    difference = gated.reindex(base.index) - base
    values = difference.to_numpy(dtype=float)
    block = max(1, min(block_months, len(values)))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(max(100, replicates)):
        starts = rng.integers(
            0,
            len(values),
            size=max(1, int(np.ceil(len(values) / block))),
        )
        sample = np.concatenate([
            np.take(
                values,
                np.arange(start, start + block) % len(values),
            )
            for start in starts
        ])[:len(values)]
        samples.append(float(sample.mean()))
    return {
        "delta_bootstrap_p10": float(np.quantile(samples, 0.10)),
        "delta_bootstrap_median": float(np.quantile(samples, 0.50)),
        "delta_bootstrap_p90": float(np.quantile(samples, 0.90)),
    }


def _shock_metrics(
    risk_predictions: pd.DataFrame,
    start: int,
    end: int,
) -> dict[str, float | int]:
    evaluation = risk_predictions[
        risk_predictions["origin_position"].between(start, end)
        & risk_predictions["shock_label_valid"].fillna(False).astype(bool)
    ].copy()
    target = evaluation["sudden_drop"].astype(int)
    probability = evaluation["p_sudden_drop"].astype(float)
    return {
        "rows": int(len(evaluation)),
        "events": int(target.sum()),
        "prevalence": float(target.mean()),
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(
            average_precision_score(target, probability)
        ),
        "brier": float(brier_score_loss(target, probability)),
    }


def _write_experiment_report(
    summary: dict,
    penalty_search: pd.DataFrame,
    root: Path,
) -> None:
    experiment_root = downside_experiment_root(root)
    metrics_dir = experiment_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    penalty_search.to_csv(
        metrics_dir / "penalty_search.csv",
        index=False,
    )
    atomic_write_json(summary, downside_summary_path(root))
    lines = [
        "# Downside Risk Gate",
        "",
        "This is an experimental risk filter. It does not replace the active Uptrend Selector.",
        "",
        "## Frozen design",
        "",
        f"- Selected penalty: `{summary['selected_penalty']}`",
        "- Penalty selection window: Discovery origins 120-219.",
        "- Evaluation window: Confirmation origins 220-266.",
        "- Historical locked evidence 268-315 was not read.",
        "- The gate reranks Up candidates; it does not flip predictions to Down.",
        "",
        "## Results",
        "",
        f"- Discovery base accuracy: `{summary['discovery_base']['accuracy']:.4%}`",
        f"- Discovery gated accuracy: `{summary['discovery_gated']['accuracy']:.4%}`",
        f"- Confirmation base accuracy: `{summary['confirmation_base']['accuracy']:.4%}`",
        f"- Confirmation gated accuracy: `{summary['confirmation_gated']['accuracy']:.4%}`",
        f"- Confirmation accuracy delta: `{summary['confirmation_accuracy_delta']:+.4%}`",
        f"- Confirmation changed selections: `{summary['confirmation_gated']['changed_calls']}`",
        f"- Shock ROC AUC on Confirmation: `{summary['confirmation_shock_metrics']['roc_auc']:.4f}`",
        "",
        "The experiment is not promoted automatically. A positive point estimate is insufficient when the paired date-block interval includes zero.",
    ]
    (experiment_root / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_downside_risk_gate(root: Path = ROOT) -> Path:
    """Build and evaluate the experimental risk gate without touching locked evidence."""
    settings = _read_yaml(root / "configs/downside_risk_gate.yaml")
    discovery_start, discovery_end = (
        int(value) for value in settings["discovery_origins"]
    )
    confirmation_start, confirmation_end = (
        int(value) for value in settings["confirmation_origins"]
    )
    cap = int(settings["gate"]["monthly_selection_count"])

    active_path = active_model_artifact(root)
    if not active_path.exists():
        build_active_model(root)
    discovery_base = pd.read_parquet(active_path)
    confirmation_base = build_uptrend_predictions(
        root,
        origin_range=(confirmation_start, confirmation_end),
    )
    validate_oof_columns(discovery_base.columns.tolist())
    validate_oof_columns(confirmation_base.columns.tolist())
    base_predictions = pd.concat(
        [discovery_base, confirmation_base],
        ignore_index=True,
    )

    risk_predictions = build_downside_risk_predictions(root)
    search_rows = []
    discovery_risk = risk_predictions[
        risk_predictions["origin_position"].between(
            discovery_start,
            discovery_end,
        )
    ]
    for penalty in settings["gate"]["penalty_grid"]:
        gated = apply_downside_risk_gate(
            discovery_base,
            discovery_risk,
            penalty=float(penalty),
            cap=cap,
        )
        result = summarize_gate_predictions(gated)
        search_rows.append({
            "penalty": float(penalty),
            **result,
        })
    penalty_search = pd.DataFrame(search_rows).sort_values(
        ["accuracy", "penalty"],
        ascending=[False, True],
    ).reset_index(drop=True)
    selected_penalty = float(penalty_search.iloc[0]["penalty"])
    gated_predictions = apply_downside_risk_gate(
        base_predictions,
        risk_predictions,
        penalty=selected_penalty,
        cap=cap,
    )
    gated_predictions["model_id"] = settings["experiment_id"]
    gated_predictions["model_version"] = settings["experiment_release"]
    gated_predictions["risk_gate_selected_on"] = "discovery_120_219"
    gated_predictions["locked_evaluation_read"] = False
    validate_oof_columns(gated_predictions.columns.tolist())

    confirmation_base = base_predictions[
        base_predictions["origin_position"].between(
            confirmation_start,
            confirmation_end,
        )
    ].copy()
    confirmation_gated = gated_predictions[
        gated_predictions["origin_position"].between(
            confirmation_start,
            confirmation_end,
        )
    ].copy()
    discovery_gated = gated_predictions[
        gated_predictions["origin_position"].between(
            discovery_start,
            discovery_end,
        )
    ].copy()
    discovery_base_summary = _selection_summary(discovery_base)
    discovery_gated_summary = summarize_gate_predictions(discovery_gated)
    confirmation_base_summary = _selection_summary(confirmation_base)
    confirmation_gated_summary = summarize_gate_predictions(
        confirmation_gated
    )
    delta = (
        float(confirmation_gated_summary["accuracy"])
        - float(confirmation_base_summary["accuracy"])
    )
    config = _read_yaml(root / "configs/config.yaml")
    summary = {
        "experiment_id": settings["experiment_id"],
        "experiment_name": settings["experiment_name"],
        "experiment_release": settings["experiment_release"],
        "selected_penalty": selected_penalty,
        "penalty_selected_on": "discovery_120_219",
        "discovery_base": discovery_base_summary,
        "discovery_gated": discovery_gated_summary,
        "confirmation_base": confirmation_base_summary,
        "confirmation_gated": confirmation_gated_summary,
        "confirmation_accuracy_delta": delta,
        "confirmation_shock_metrics": _shock_metrics(
            risk_predictions,
            confirmation_start,
            confirmation_end,
        ),
        "confirmation_read": True,
        "locked_evaluation_read": False,
        "locked_origins": [268, 315],
        "active_model_changed": False,
        "config_hash": _risk_configuration_hash(root),
        "data_hash": sha256_file(root / config["data_path"]),
    }
    summary.update(_paired_block_delta(
        confirmation_base,
        confirmation_gated,
        block_months=int(config["bootstrap_blocks"]),
        replicates=int(config["bootstrap_replicates"]),
        seed=int(config["seed"]),
    ))

    atomic_write_parquet(
        risk_predictions,
        downside_risk_artifact(root),
    )
    atomic_write_parquet(
        gated_predictions,
        gated_predictions_artifact(root),
    )
    _write_experiment_report(summary, penalty_search, root)
    return downside_summary_path(root)


def downside_risk_gate_status(root: Path = ROOT) -> dict:
    path = downside_summary_path(root)
    if not path.exists():
        return {
            "ready": False,
            "summary": path.relative_to(root).as_posix(),
        }
    with path.open(encoding="utf-8") as handle:
        return {"ready": True, **json.load(handle)}
