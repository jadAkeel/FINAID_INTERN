"""Orchestration for the extreme-down sensing and guarded replacement study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .down_sensing import (
    DEFAULT_POLICY,
    evaluate_window,
    extreme_down_labels,
    build_extreme_down_features,
    paired_block_bootstrap_delta,
    score_extreme_down_walk_forward,
    select_variant,
)
from .downside_risk import build_sudden_drop_labels
from .io import atomic_write_json, load_workbook
from .targets import build_targets

LOCKED_ORIGINS = (268, 315)
TUNING_ORIGINS = (120, 179)
TUNING_SUBWINDOWS = [(120, 149), (150, 179)]
VALIDATION_ORIGINS = (180, 219)
CONFIRMATION_ORIGINS = (220, 266)


def _policy_grid(grid_settings: dict) -> list[dict]:
    variants = []
    for breadth_gate in grid_settings["breadth_gates"]:
        for risk_quantile in grid_settings["risk_quantiles"]:
            for max_replacements in grid_settings["max_replacements"]:
                for ceiling in grid_settings.get(
                    "conviction_ceilings", [DEFAULT_POLICY["conviction_ceiling"]]
                ):
                    policy = dict(DEFAULT_POLICY)
                    policy.update({
                        "breadth_gate": float(breadth_gate),
                        "risk_quantile": float(risk_quantile),
                        "max_replacements": int(max_replacements),
                        "conviction_ceiling": float(ceiling),
                    })
                    name = (
                        f"breadth<{breadth_gate:g}_q{risk_quantile:g}"
                        f"_cap{max_replacements}_ceil{ceiling:g}"
                    )
                    variants.append((name, policy))
    return variants


def run_down_sensing_study(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    settings: dict,
) -> tuple[dict, pd.DataFrame]:
    """Pure in-memory study used by the pipeline and tests."""
    maximum_origin = int(settings["evaluation_origins"][1])
    frame = frame[frame["position"] <= maximum_origin].copy()
    targets = build_targets(frame)

    shock_labels = build_sudden_drop_labels(
        targets,
        trailing_window=int(settings["shock_definition"]["trailing_window"]),
        minimum_history=int(settings["shock_definition"]["minimum_history"]),
        lower_quantile=float(settings["shock_definition"]["lower_quantile"]),
        robust_z=float(settings["shock_definition"]["robust_z"]),
    )
    labels = extreme_down_labels(shock_labels)

    features = build_extreme_down_features(frame)
    features = features.merge(labels, on=["origin_position", "indicator_id"], how="left")

    scores = score_extreme_down_walk_forward(
        features,
        settings["model"],
        train_lag=int(settings.get("train_lag_months", 2)),
        start_origin=int(settings["evaluation_origins"][0]),
        end_origin=maximum_origin,
    )

    work = panel[
        panel["origin_position"].between(*settings["evaluation_origins"])
    ].copy()

    variants = _policy_grid(settings["policy_grid"])
    tuning_results = {}
    subwindow_results = {}
    validation_results = {}
    confirmation_results = {}
    for name, policy in variants:
        tuning_results[name] = evaluate_window(
            work[work["origin_position"].between(*TUNING_ORIGINS)], scores, policy
        )
        subwindow_results[name] = [
            {
                "window": f"{lo}_{hi}",
                **evaluate_window(
                    work[work["origin_position"].between(lo, hi)], scores, policy
                ),
            }
            for lo, hi in TUNING_SUBWINDOWS
        ]
        validation_results[name] = evaluate_window(
            work[work["origin_position"].between(*VALIDATION_ORIGINS)],
            scores,
            policy,
        )
        confirmation_results[name] = evaluate_window(
            work[work["origin_position"].between(*CONFIRMATION_ORIGINS)],
            scores,
            policy,
        )

    selected_name, selection_mode = select_variant(tuning_results, subwindow_results)
    summary = {
        "experiment_id": settings["experiment_id"],
        "experiment_name": settings["experiment_name"],
        "experiment_release": settings.get("experiment_release", "initial_experiment"),
        "active_model_changed": False,
        "locked_evaluation_read": False,
        "locked_origins": list(LOCKED_ORIGINS),
        "confirmation_read": True,
        "selection_mode": selection_mode,
        "selected_policy": selected_name,
        "selected_parameters": (
            dict(variants[[n for n, _ in variants].index(selected_name)][1])
            if selected_name else None
        ),
        "tuning": tuning_results.get(selected_name),
        "validation": validation_results.get(selected_name),
        "confirmation_descriptive": confirmation_results.get(selected_name),
        "all_tuning_variants": tuning_results,
        "all_validation_variants": validation_results,
    }

    if selected_name:
        selected_policy = summary["selected_parameters"]
        summary["validation_bootstrap"] = paired_block_bootstrap_delta(
            work[work["origin_position"].between(*VALIDATION_ORIGINS)],
            scores,
            selected_policy,
        )
        validation = summary["validation"]
        bootstrap = summary["validation_bootstrap"]
        down_calls = int(validation["down_calls"])
        prevalence = float(work["y_true"].eq(0).mean())
        summary["promotion_eligible"] = bool(
            validation["delta_hits"] >= 0
            and bootstrap["bootstrap_p10"] >= -0.02
            and down_calls >= 10
            and (validation["down_hits"] / down_calls) > prevalence
        )

    scored = scores[scores["origin_position"] <= maximum_origin]
    diagnostics = []
    for window_lo, window_hi in [
        TUNING_ORIGINS,
        VALIDATION_ORIGINS,
        CONFIRMATION_ORIGINS,
    ]:
        mask = scored["origin_position"].between(window_lo, window_hi)
        joined = scored[mask].merge(
            labels, on=["origin_position", "indicator_id"], how="left"
        ).dropna(subset=["extreme_down_next", "p_extreme_down"])
        if joined.empty or joined["extreme_down_next"].nunique() < 2:
            continue
        order = np.argsort(joined["p_extreme_down"].to_numpy())
        ranks = pd.Series(
            joined["p_extreme_down"].to_numpy()[order]
        ).rank(method="average").to_numpy()
        y = joined["extreme_down_next"].to_numpy()[order]
        positives = y.sum()
        negatives = len(y) - positives
        auc = float(
            (ranks[y == 1].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
        diagnostics.append({
            "window": f"{window_lo}_{window_hi}",
            "auc": round(auc, 4),
            "prevalence": round(float(y.mean()), 4),
            "rows": int(len(y)),
        })
    summary["score_diagnostics"] = diagnostics
    summary["data_hash_note"] = "frozen_regime_panel_reused"
    return summary, scores


def build_down_sensing_gate(root: Path) -> str:
    root = Path(root)
    config_path = root / "configs" / "down_sensing_gate.yaml"
    with config_path.open(encoding="utf-8") as handle:
        import yaml

        settings = yaml.safe_load(handle)

    frame = load_workbook(
        root / "data" / "monthly_indicators.xlsx",
        maximum_position=LOCKED_ORIGINS[0],
    )
    panel_path = (
        root
        / "research"
        / "regime_adaptive_selector"
        / "artifacts"
        / "predictions.parquet"
    )
    if not panel_path.exists():
        raise FileNotFoundError(
            "Run build-regime-adaptive first; the frozen selector panel is required"
        )
    panel = pd.read_parquet(panel_path)
    panel = panel[panel["locked_evaluation_read"] == False]  # noqa: E712

    summary, scores = run_down_sensing_study(frame, panel, settings)

    base_dir = root / "research" / settings["experiment_id"]
    (base_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (base_dir / "metrics").mkdir(parents=True, exist_ok=True)
    scores.to_parquet(base_dir / "artifacts" / "extreme_scores.parquet", index=False)
    atomic_write_json(summary, base_dir / "metrics" / "summary.json")
    readme = base_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Down Sensing Gate\n\n"
            "Extreme-down sensing with guarded Down replacement on the frozen\n"
            "Regime Adaptive panel. See metrics/summary.json.\n",
            encoding="utf-8",
        )
    return (
        f"built {settings['experiment_id']}: "
        f"selected={summary['selected_policy']} "
        f"promotion_eligible={summary.get('promotion_eligible')}"
    )


def down_sensing_status(root: Path) -> dict:
    path = Path(root) / "research" / "down_sensing_gate" / "metrics" / "summary.json"
    if not path.exists():
        return {"status": "not_built"}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
