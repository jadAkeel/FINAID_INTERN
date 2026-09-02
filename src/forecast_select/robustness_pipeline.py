from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .downside_risk import build_sudden_drop_labels
from .io import atomic_write_json, load_workbook
from .regime_adaptive_pipeline import (
    _apply,
    _build_inputs,
    _read_yaml,
    regime_adaptive_summary_path,
)
from .targets import build_targets
from .uptrend_pipeline import ROOT


def robustness_experiment_root(root: Path = ROOT) -> Path:
    return root / "research/regime_adaptive_robustness"


def robustness_summary_path(root: Path = ROOT) -> Path:
    return robustness_experiment_root(root) / "metrics/summary.json"


def robustness_metrics_path(root: Path = ROOT) -> Path:
    return robustness_experiment_root(root) / "metrics/scenarios.csv"


def _outcome_panel(root: Path, settings: dict) -> pd.DataFrame:
    project = _read_yaml(root / "configs/config.yaml")
    frame = load_workbook(root / project["data_path"])
    targets = build_targets(frame)
    shock = _read_yaml(root / "configs/downside_risk_gate.yaml")["shock_definition"]
    labels = build_sudden_drop_labels(
        targets,
        trailing_window=int(shock["trailing_window"]),
        minimum_history=int(shock["minimum_history"]),
        lower_quantile=float(shock["lower_quantile"]),
        robust_z=float(shock["robust_z"]),
    )
    return labels[[
        "origin_position",
        "indicator_id",
        "shock_label_valid",
        "sudden_drop",
    ]].rename(columns={"sudden_drop": "actual_sudden_drop"})


def _scenario_metrics(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    start: int,
    end: int,
    regime: str = "all",
) -> dict[str, float | int | str]:
    frame = predictions[
        predictions["origin_position"].between(start, end)
    ].merge(
        outcomes,
        on=["origin_position", "indicator_id"],
        how="left",
        validate="one_to_one",
    )
    if regime != "all":
        frame = frame[frame["regime_label"].eq(regime)]
    selected = frame[
        frame["accepted"].fillna(False).astype(bool)
        & frame["y_true"].notna()
    ]
    down = selected[selected["predicted_direction"].eq("Down")]
    actual_down = frame[frame["y_true"].eq(0.0)]
    valid_shocks = frame[frame["shock_label_valid"].fillna(False).astype(bool)]
    sudden = valid_shocks[valid_shocks["actual_sudden_drop"].eq(True)]
    sudden_down = down[down["actual_sudden_drop"].eq(True)]
    correct = np.where(
        selected["predicted_direction"].eq("Up"),
        selected["y_true"].eq(1.0),
        selected["y_true"].eq(0.0),
    )
    return {
        "window": f"{start}_{end}",
        "regime": regime,
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(np.sum(correct)),
        "accuracy": float(np.mean(correct)) if len(correct) else np.nan,
        "down_calls": int(len(down)),
        "down_hits": int(down["y_true"].eq(0.0).sum()),
        "down_precision": float(down["y_true"].eq(0.0).mean()) if len(down) else np.nan,
        "normal_down_recall": float(down["y_true"].eq(0.0).sum() / len(actual_down)) if len(actual_down) else np.nan,
        "sudden_drop_events": int(len(sudden)),
        "sudden_down_calls": int(len(sudden_down)),
        "sudden_down_precision": float(sudden_down["y_true"].eq(0.0).mean()) if len(sudden_down) else np.nan,
        "sudden_drop_recall": float(len(sudden_down) / len(sudden)) if len(sudden) else np.nan,
        "replacement_calls": int(selected["regime_replacement"].fillna(False).sum()),
        "abstained_down_candidates": int(
            frame["down_candidate"].fillna(False).astype(bool)
            .astype(int).sub(frame["accepted"].fillna(False).astype(bool).astype(int))
            .clip(lower=0).sum()
        ),
    }


def build_regime_adaptive_robustness(root: Path = ROOT) -> Path:
    current_settings = _read_yaml(root / "configs/regime_adaptive_selector.yaml")
    study_settings = _read_yaml(root / "configs/regime_adaptive_robustness.yaml")
    current_summary = json.loads(regime_adaptive_summary_path(root).read_text(encoding="utf-8"))
    selected = current_summary["selected_parameters"]
    effective_cap = (
        None
        if bool(current_settings["selection"]["dynamic_cap_enabled"])
        else int(current_settings["selection"]["monthly_selection_count"])
    )
    inputs = _build_inputs(
        root,
        current_settings,
        int(current_settings["selection"]["maximum_selection_count"]),
    )
    outcomes = _outcome_panel(root, current_settings)
    rows: list[dict] = []
    windows = {
        "tuning": current_settings["tuning_origins"],
        "validation": current_settings["validation_origins"],
        "confirmation": current_settings["confirmation_origins"],
    }
    for policy in study_settings["abstention_policies"]:
        for replacement_cap in study_settings["replacement_caps"]:
            params = {
                key: selected[key]
                for key in [
                    "stress_trigger",
                    "maximum_down_share",
                    "regime_down_bonus",
                    "shock_down_bonus",
                    "replacement_margin",
                ]
            }
            params.update({
                "down_threshold": float(policy["down_threshold"]),
                "down_margin": float(policy["down_margin"]),
                "maximum_replacements": int(replacement_cap),
            })
            predictions = _apply(inputs, current_settings, params, effective_cap)
            for window, bounds in windows.items():
                for regime in ["all", "calm", "mixed", "stressed"]:
                    row = _scenario_metrics(
                        predictions,
                        outcomes,
                        int(bounds[0]),
                        int(bounds[1]),
                        regime,
                    )
                    row.update({
                        "policy": policy["id"],
                        "down_threshold": params["down_threshold"],
                        "down_margin": params["down_margin"],
                        "maximum_replacements": int(replacement_cap),
                    })
                    rows.append(row)
    metrics = pd.DataFrame(rows)
    output_root = robustness_experiment_root(root)
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    metrics.to_csv(robustness_metrics_path(root), index=False)
    summary = {
        "experiment_id": "regime_adaptive_robustness",
        "base_experiment_id": current_summary["experiment_id"],
        "locked_evaluation_read": False,
        "locked_origins": current_settings["locked_origins"],
        "replacement_caps": study_settings["replacement_caps"],
        "abstention_policies": study_settings["abstention_policies"],
        "best_validation": metrics[
            (metrics["window"] == "180_219") & (metrics["regime"] == "all")
        ].sort_values("accuracy", ascending=False).head(1).to_dict("records"),
        "confirmation_descriptive_current_policy": metrics[
            (metrics["window"] == "220_266")
            & (metrics["regime"] == "all")
            & metrics["down_threshold"].eq(float(selected["down_threshold"]))
            & metrics["down_margin"].eq(float(selected["down_margin"]))
            & metrics["maximum_replacements"].eq(
                int(selected["maximum_replacements"])
            )
        ].to_dict("records"),
        "confirmation_used_for_selection": False,
        "current_selector_parameters": selected,
    }
    atomic_write_json(summary, robustness_summary_path(root))
    (output_root / "README.md").write_text(
        "\n".join([
            "# Regime Adaptive Robustness Study",
            "",
            "This study evaluates replacement caps 0-3 and conservative Down abstention policies without reading locked origins.",
            "",
            "- Tuning: origins 120-179.",
            "- Validation: origins 180-219.",
            "- Confirmation: origins 220-266.",
            "- Locked origins 268-315 were not read.",
            "- Metrics separate all regimes, calm/mixed/stressed regimes, normal Down, and sudden-drop outcomes.",
            "",
            "The results are descriptive robustness evidence. They do not change the active model.",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    return robustness_metrics_path(root)


def regime_adaptive_robustness_status(root: Path = ROOT) -> dict:
    return json.loads(robustness_summary_path(root).read_text(encoding="utf-8"))
