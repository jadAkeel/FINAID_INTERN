from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .directional_downside import summarize_bidirectional_predictions
from .directional_downside_pipeline import (
    build_directional_downside_selector,
    directional_downside_predictions_artifact,
)
from .downside_pipeline import (
    build_downside_risk_gate,
    gated_predictions_artifact,
)
from .contextual_pipeline import (
    build_contextual_defensive_selector,
    contextual_predictions_artifact,
)
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .schemas import validate_oof_columns
from .unified_controller import (
    apply_unified_controller,
    summarize_unified_predictions,
)
from .uptrend_pipeline import ROOT


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def unified_experiment_root(root: Path = ROOT) -> Path:
    return root / "research/unified_forecast_controller"


def unified_predictions_artifact(root: Path = ROOT) -> Path:
    return unified_experiment_root(root) / "artifacts/predictions.parquet"


def unified_summary_path(root: Path = ROOT) -> Path:
    return unified_experiment_root(root) / "metrics/summary.json"


def unified_search_path(root: Path = ROOT) -> Path:
    return unified_experiment_root(root) / "metrics/candidate_search.csv"


def _ensure_inputs(root: Path) -> None:
    if not gated_predictions_artifact(root).exists():
        build_downside_risk_gate(root)
    if not contextual_predictions_artifact(root).exists():
        build_contextual_defensive_selector(root)
    if not directional_downside_predictions_artifact(root).exists():
        build_directional_downside_selector(root)


def _build_panel(root: Path) -> pd.DataFrame:
    _ensure_inputs(root)
    directional = pd.read_parquet(directional_downside_predictions_artifact(root))
    risk = pd.read_parquet(gated_predictions_artifact(root))
    context = pd.read_parquet(contextual_predictions_artifact(root))
    validate_oof_columns(directional.columns.tolist())
    if int(directional["origin_position"].max()) >= 268:
        raise AssertionError("Unified controller must stop before locked origins")

    risk_columns = [
        "origin_position",
        "indicator_id",
        "p_sudden_drop",
        "risk_percentile",
        "risk_gate_changed",
    ]
    context_columns = [
        "origin_position",
        "indicator_id",
        "breadth_mean_3",
        "context_stress",
        "context_role_indicators",
        "context_selection_changed",
        "context_forced_role",
    ]
    panel = directional.merge(
        risk[risk_columns],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        context[context_columns],
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    if panel["context_role_indicators"].isna().any():
        raise AssertionError("Contextual artifact did not cover controller rows")
    panel["risk_percentile"] = pd.to_numeric(
        panel["risk_percentile"], errors="coerce"
    )
    return panel


def _window(frame: pd.DataFrame, bounds: list[int]) -> pd.DataFrame:
    return frame[frame["origin_position"].between(int(bounds[0]), int(bounds[1]))]


def _paired_block_delta(
    base: pd.DataFrame,
    candidate: pd.DataFrame,
    bounds: list[int],
    block_months: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    base = _window(base, bounds)
    candidate = _window(candidate, bounds)
    base_selected = base[base["accepted"]].copy()
    candidate_selected = candidate[candidate["accepted"]].copy()
    base_selected["correct"] = base_selected["predicted_direction"].eq(
        base_selected["y_true"].astype(int).map({1: "Up", 0: "Down"})
    )
    candidate_selected["correct"] = candidate_selected[
        "predicted_direction"
    ].eq(candidate_selected["y_true"].astype(int).map({1: "Up", 0: "Down"}))
    paired = base_selected[["origin_position", "indicator_id", "correct"]].merge(
        candidate_selected[
            ["origin_position", "indicator_id", "correct"]
        ],
        on=["origin_position", "indicator_id"],
        suffixes=("_base", "_candidate"),
        validate="one_to_one",
    )
    monthly = paired.assign(
        delta=paired["correct_candidate"].astype(float)
        - paired["correct_base"].astype(float)
    ).groupby("origin_position")["delta"].mean().to_numpy()
    observed = float(monthly.mean()) if len(monthly) else 0.0
    rng = np.random.default_rng(seed)
    block = max(1, min(int(block_months), len(monthly)))
    samples = []
    for _ in range(max(50, int(replicates))):
        starts = rng.integers(0, len(monthly), size=max(1, int(np.ceil(len(monthly) / block))))
        sample = np.concatenate([
            np.take(monthly, np.arange(start, start + block) % len(monthly))
            for start in starts
        ])[:len(monthly)]
        samples.append(float(sample.mean()))
    return {
        "confirmation_delta_bootstrap_p10": float(np.quantile(samples, 0.10)),
        "confirmation_delta_bootstrap_median": float(np.quantile(samples, 0.50)),
        "confirmation_delta_bootstrap_p90": float(np.quantile(samples, 0.90)),
        "confirmation_delta_observed": observed,
    }


def build_unified_controller(root: Path = ROOT) -> Path:
    settings = _read_yaml(root / "configs/unified_controller.yaml")
    project_config = _read_yaml(root / "configs/config.yaml")
    panel = _build_panel(root)
    controller = settings["controller"]
    tuning = _window(panel, settings["tuning_origins"])
    rows = []
    for risk_penalty, down_bonus, context_bonus in itertools.product(
        controller["risk_penalty_grid"],
        controller["down_risk_bonus_grid"],
        controller["context_role_bonus_grid"],
    ):
        candidate = apply_unified_controller(
            tuning,
            risk_penalty=float(risk_penalty),
            down_risk_bonus=float(down_bonus),
            context_role_bonus=float(context_bonus),
            cap=int(controller["monthly_selection_count"]),
            require_risk_data=bool(controller["require_risk_data"]),
        )
        summary = summarize_unified_predictions(candidate)
        rows.append({
            "risk_penalty": float(risk_penalty),
            "down_risk_bonus": float(down_bonus),
            "context_role_bonus": float(context_bonus),
            "minimum_down_calls_met": (
                int(summary["down_calls"])
                >= int(controller["minimum_tuning_down_calls"])
            ),
            **summary,
        })
    search = pd.DataFrame(rows)
    eligible = search[search["minimum_down_calls_met"]].copy()
    if eligible.empty:
        raise RuntimeError("No unified-controller candidate made enough Down calls")
    eligible = eligible.sort_values(
        ["accuracy", "down_accuracy", "down_calls", "risk_penalty"],
        ascending=[False, False, False, True],
    )
    selected = eligible.iloc[0]
    selected_parameters = {
        "risk_penalty": float(selected["risk_penalty"]),
        "down_risk_bonus": float(selected["down_risk_bonus"]),
        "context_role_bonus": float(selected["context_role_bonus"]),
    }
    final = apply_unified_controller(
        panel,
        **selected_parameters,
        cap=int(controller["monthly_selection_count"]),
        require_risk_data=bool(controller["require_risk_data"]),
    )
    final["run_id"] = settings["experiment_id"]
    final["model_id"] = settings["experiment_id"]
    final["model_version"] = settings["experiment_release"]
    final["parameters_selected_on"] = "tuning_120_179"
    final["locked_evaluation_read"] = False
    validate_oof_columns(final.columns.tolist())

    tuning_base = summarize_bidirectional_predictions(_window(panel, settings["tuning_origins"]))
    validation_base = summarize_bidirectional_predictions(_window(panel, settings["validation_origins"]))
    confirmation_base = summarize_bidirectional_predictions(_window(panel, settings["confirmation_origins"]))
    tuning_candidate = summarize_unified_predictions(_window(final, settings["tuning_origins"]))
    validation_candidate = summarize_unified_predictions(_window(final, settings["validation_origins"]))
    confirmation_candidate = summarize_unified_predictions(_window(final, settings["confirmation_origins"]))
    validation_delta = float(validation_candidate["accuracy"] - validation_base["accuracy"])
    confirmation_delta = float(confirmation_candidate["accuracy"] - confirmation_base["accuracy"])
    uncertainty = _paired_block_delta(
        panel,
        final,
        settings["confirmation_origins"],
        block_months=int(project_config["bootstrap_blocks"]),
        replicates=int(project_config["bootstrap_replicates"]),
        seed=int(project_config["seed"]),
    )
    promotion = settings["promotion"]
    promotion_eligible = bool(
        (validation_delta > 0 if promotion["require_positive_validation_delta"] else True)
        and (confirmation_delta > 0 if promotion["require_positive_confirmation_delta"] else True)
        and (
            uncertainty["confirmation_delta_bootstrap_p10"] > 0
            if promotion["require_positive_bootstrap_p10"]
            else True
        )
    )
    summary = {
        "experiment_id": settings["experiment_id"],
        "experiment_name": settings["experiment_name"],
        "experiment_release": settings["experiment_release"],
        "selected_parameters": selected_parameters,
        "parameters_selected_on": "tuning_120_179",
        "tuning_base": tuning_base,
        "tuning_candidate": tuning_candidate,
        "validation_base": validation_base,
        "validation_candidate": validation_candidate,
        "validation_accuracy_delta": validation_delta,
        "confirmation_base": confirmation_base,
        "confirmation_candidate": confirmation_candidate,
        "confirmation_accuracy_delta": confirmation_delta,
        **uncertainty,
        "promotion_eligible": promotion_eligible,
        "active_model_changed": False,
        "confirmation_read": True,
        "locked_evaluation_read": False,
        "locked_origins": list(settings["locked_origins"]),
        "data_hash": sha256_file(root / project_config["data_path"]),
    }
    atomic_write_parquet(final, unified_predictions_artifact(root))
    atomic_write_json(summary, unified_summary_path(root))
    search.sort_values(
        ["minimum_down_calls_met", "accuracy", "down_accuracy"],
        ascending=[False, False, False],
    ).to_csv(unified_search_path(root), index=False)
    return unified_summary_path(root)


def unified_controller_status(root: Path = ROOT) -> dict:
    path = unified_summary_path(root)
    if not path.exists():
        return {"ready": False, "artifact": str(path)}
    with path.open(encoding="utf-8") as handle:
        return {"ready": True, **json.load(handle)}

