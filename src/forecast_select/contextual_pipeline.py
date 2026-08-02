from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .contextual_defensive import (
    apply_contextual_defensive_selector,
    build_causal_market_regime,
    contextual_selection_summary,
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


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _context_configuration_hash(root: Path) -> str:
    payload = {
        "project": _read_yaml(root / "configs/config.yaml"),
        "uptrend_model": _read_yaml(root / "configs/uptrend_model.yaml"),
        "contextual_defensive_selector": _read_yaml(
            root / "configs/contextual_defensive_selector.yaml"
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def contextual_experiment_root(root: Path = ROOT) -> Path:
    return root / "research/contextual_defensive_selector"


def contextual_predictions_artifact(root: Path = ROOT) -> Path:
    return contextual_experiment_root(root) / "artifacts/predictions.parquet"


def contextual_summary_path(root: Path = ROOT) -> Path:
    return contextual_experiment_root(root) / "metrics/summary.json"


def _selection_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["y_true"].notna()
    ].copy()
    direction = selected["predicted_direction"].eq("Up").astype(int)
    selected["correct"] = direction.eq(selected["y_true"].astype(int))
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(selected["correct"].sum()),
        "accuracy": float(selected["correct"].mean()),
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(selected["predicted_direction"].eq("Down").sum()),
    }


def _window_summary(
    predictions: pd.DataFrame,
    bounds: list[int],
    contextual: bool,
) -> dict[str, float | int]:
    start, end = (int(value) for value in bounds)
    window = predictions[
        predictions["origin_position"].between(start, end)
    ]
    if contextual:
        return contextual_selection_summary(window)
    return _selection_summary(window)


def _paired_block_delta(
    base_predictions: pd.DataFrame,
    contextual_predictions: pd.DataFrame,
    block_months: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    def monthly_accuracy(frame: pd.DataFrame) -> pd.Series:
        selected = frame[
            frame["accepted"].fillna(False).astype(bool)
            & frame["y_true"].notna()
        ].copy()
        selected["correct"] = selected[
            "predicted_direction"
        ].eq("Up").astype(int).eq(selected["y_true"].astype(int))
        return selected.groupby("origin_position")["correct"].mean()

    base = monthly_accuracy(base_predictions)
    contextual = monthly_accuracy(contextual_predictions)
    difference = contextual.reindex(base.index) - base
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


def _candidate_search(
    discovery_base: pd.DataFrame,
    regime: pd.DataFrame,
    settings: dict,
) -> tuple[pd.DataFrame, float, list[str]]:
    cap = int(settings["selection"]["monthly_selection_count"])
    rows = []
    for threshold in settings["regime"]["stress_threshold_grid"]:
        for roles in settings["role_sets"]:
            contextual = apply_contextual_defensive_selector(
                discovery_base,
                regime,
                stress_threshold=float(threshold),
                role_indicators=[str(role) for role in roles],
                cap=cap,
            )
            tuning = _window_summary(
                contextual,
                settings["internal_tuning_origins"],
                contextual=True,
            )
            validation = _window_summary(
                contextual,
                settings["internal_validation_origins"],
                contextual=True,
            )
            discovery = contextual_selection_summary(contextual)
            rows.append({
                "stress_threshold": float(threshold),
                "role_indicators": ",".join(str(role) for role in roles),
                "role_count": int(len(roles)),
                "tuning_hits": int(tuning["hits"]),
                "tuning_calls": int(tuning["calls"]),
                "tuning_accuracy": float(tuning["accuracy"]),
                "validation_hits": int(validation["hits"]),
                "validation_calls": int(validation["calls"]),
                "validation_accuracy": float(validation["accuracy"]),
                "discovery_hits": int(discovery["hits"]),
                "discovery_calls": int(discovery["calls"]),
                "discovery_accuracy": float(discovery["accuracy"]),
                "changed_calls": int(discovery["changed_calls"]),
            })
    search = pd.DataFrame(rows).sort_values(
        [
            "tuning_accuracy",
            "validation_accuracy",
            "role_count",
            "stress_threshold",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    selected = search.iloc[0]
    selected_roles = str(selected["role_indicators"]).split(",")
    return search, float(selected["stress_threshold"]), selected_roles


def _write_report(
    summary: dict,
    candidate_search: pd.DataFrame,
    root: Path,
) -> None:
    experiment_root = contextual_experiment_root(root)
    metrics_dir = experiment_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    candidate_search.to_csv(
        metrics_dir / "candidate_search.csv",
        index=False,
    )
    atomic_write_json(summary, contextual_summary_path(root))
    lines = [
        "# Contextual Defensive Selector",
        "",
        "This experiment forces neutral indicator roles Up only when a past-only market-breadth signal indicates stress.",
        "",
        "## Frozen design",
        "",
        f"- Selected stress threshold: `{summary['selected_stress_threshold']}`",
        f"- Selected role indicators: `{', '.join(summary['selected_role_indicators'])}`",
        "- Candidate selection used origins 120-179, with origins 180-219 as internal validation.",
        "- Confirmation evaluation used origins 220-266 once after the rule was selected.",
        "- Historical locked evidence 268-315 was not read.",
        "- Indicator identities remain unknown; role labels describe behavior only.",
        "",
        "## Results",
        "",
        f"- Discovery base accuracy: `{summary['discovery_base']['accuracy']:.4%}`",
        f"- Discovery contextual accuracy: `{summary['discovery_contextual']['accuracy']:.4%}`",
        f"- Confirmation base accuracy: `{summary['confirmation_base']['accuracy']:.4%}`",
        f"- Confirmation contextual accuracy: `{summary['confirmation_contextual']['accuracy']:.4%}`",
        f"- Confirmation accuracy delta: `{summary['confirmation_accuracy_delta']:+.4%}`",
        "",
        f"Promotion eligible: `{summary['promotion_eligible']}`. A positive point estimate alone is not enough; the paired block lower bound must also be positive.",
    ]
    (experiment_root / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_contextual_defensive_selector(root: Path = ROOT) -> Path:
    """Build a Discovery-selected contextual role experiment without locked data."""
    settings = _read_yaml(
        root / "configs/contextual_defensive_selector.yaml"
    )
    project_config = _read_yaml(root / "configs/config.yaml")
    frame, _, _, _, _ = _prepare(root)
    regime = build_causal_market_regime(
        frame,
        availability_lag=int(settings["availability_lag_months"]),
    )

    active_path = active_model_artifact(root)
    if not active_path.exists():
        build_active_model(root)
    discovery_base = pd.read_parquet(active_path)
    validate_oof_columns(discovery_base.columns.tolist())
    candidate_search, selected_threshold, selected_roles = _candidate_search(
        discovery_base,
        regime,
        settings,
    )

    confirmation_start, confirmation_end = (
        int(value) for value in settings["confirmation_origins"]
    )
    confirmation_base = build_uptrend_predictions(
        root,
        origin_range=(confirmation_start, confirmation_end),
    )
    base_predictions = pd.concat(
        [discovery_base, confirmation_base],
        ignore_index=True,
    )
    contextual = apply_contextual_defensive_selector(
        base_predictions,
        regime,
        stress_threshold=selected_threshold,
        role_indicators=selected_roles,
        cap=int(settings["selection"]["monthly_selection_count"]),
    )
    contextual["model_id"] = settings["experiment_id"]
    contextual["model_version"] = settings["experiment_release"]
    contextual["context_selected_on"] = "discovery_120_219"
    contextual["locked_evaluation_read"] = False
    validate_oof_columns(contextual.columns.tolist())

    discovery_bounds = settings["discovery_origins"]
    confirmation_bounds = settings["confirmation_origins"]
    discovery_base_summary = _window_summary(
        base_predictions,
        discovery_bounds,
        contextual=False,
    )
    discovery_contextual_summary = _window_summary(
        contextual,
        discovery_bounds,
        contextual=True,
    )
    confirmation_base_summary = _window_summary(
        base_predictions,
        confirmation_bounds,
        contextual=False,
    )
    confirmation_contextual_summary = _window_summary(
        contextual,
        confirmation_bounds,
        contextual=True,
    )
    confirmation_delta = (
        float(confirmation_contextual_summary["accuracy"])
        - float(confirmation_base_summary["accuracy"])
    )
    confirmation_base_rows = base_predictions[
        base_predictions["origin_position"].between(
            confirmation_start,
            confirmation_end,
        )
    ]
    confirmation_contextual_rows = contextual[
        contextual["origin_position"].between(
            confirmation_start,
            confirmation_end,
        )
    ]
    uncertainty = _paired_block_delta(
        confirmation_base_rows,
        confirmation_contextual_rows,
        block_months=int(project_config["bootstrap_blocks"]),
        replicates=int(project_config["bootstrap_replicates"]),
        seed=int(project_config["seed"]),
    )
    summary = {
        "experiment_id": settings["experiment_id"],
        "experiment_name": settings["experiment_name"],
        "experiment_release": settings["experiment_release"],
        "selected_stress_threshold": selected_threshold,
        "selected_role_indicators": selected_roles,
        "selected_on": "internal_tuning_120_179_then_validation_180_219",
        "discovery_base": discovery_base_summary,
        "discovery_contextual": discovery_contextual_summary,
        "confirmation_base": confirmation_base_summary,
        "confirmation_contextual": confirmation_contextual_summary,
        "confirmation_accuracy_delta": confirmation_delta,
        "confirmation_read": True,
        "locked_evaluation_read": False,
        "locked_origins": [268, 315],
        "active_model_changed": False,
        "config_hash": _context_configuration_hash(root),
        "data_hash": sha256_file(root / project_config["data_path"]),
        **uncertainty,
    }
    summary["promotion_eligible"] = bool(
        confirmation_delta > 0
        and float(summary["delta_bootstrap_p10"]) > 0
    )

    atomic_write_parquet(
        contextual,
        contextual_predictions_artifact(root),
    )
    _write_report(summary, candidate_search, root)
    return contextual_summary_path(root)


def contextual_defensive_status(root: Path = ROOT) -> dict:
    path = contextual_summary_path(root)
    if not path.exists():
        return {
            "ready": False,
            "artifact": str(path),
        }
    with path.open(encoding="utf-8") as handle:
        return {"ready": True, **json.load(handle)}
