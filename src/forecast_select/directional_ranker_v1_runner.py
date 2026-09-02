from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .directional_ranker_v1 import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, directional_metrics,
    fit_directional_ranker, score_directional_candidates,
    select_with_existing_caps, selection_metrics,
)
from .io import atomic_write_json, atomic_write_parquet
from .uptrend_pipeline import ROOT


def build_directional_ranker_v1_audit(root: Path = ROOT) -> Path:
    source = pd.read_parquet(root / "artifacts/active/regime_adaptive_predictions.parquet")
    if int(source["origin_position"].max()) >= 268:
        raise ValueError("Directional-ranker source includes locked origins")
    fitted = fit_directional_ranker(source)
    scored = select_with_existing_caps(score_directional_candidates(source, fitted))
    windows = {
        "tuning": (120, 179), "validation": (180, 219),
        "confirmation_descriptive": (220, 266),
    }
    evidence = {}
    for name, bounds in windows.items():
        current = scored[scored["origin_position"].between(*bounds)]
        evidence[name] = {
            "baseline_direction": directional_metrics(current, "p_up_base"),
            "directional_ranker_v1": directional_metrics(current, "p_up_directional_v1"),
            "baseline_selection": selection_metrics(current, "accepted", "predicted_direction"),
            "directional_selection_v1": selection_metrics(
                current, "accepted_directional_v1", "predicted_direction_v1"
            ),
        }
    validation = evidence["validation"]
    accepted = bool(
        validation["directional_ranker_v1"]["auc"] > validation["baseline_direction"]["auc"]
        and validation["directional_ranker_v1"]["auc"] >= 0.55
        and validation["directional_selection_v1"]["accuracy"]
        > validation["baseline_selection"]["accuracy"]
    )
    payload = {
        "experiment_id": "directional_ranker_v1",
        "decision": "candidate_passed_validation" if accepted else "candidate_rejected",
        "accepted": accepted,
        "selected_regularization_c": fitted.regularization_c,
        "late_tuning_auc_used_for_c_selection": fitted.late_tuning_auc,
        "features": [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES],
        "training_origins": [120, 179],
        "locked_evaluation_read": False,
        "windows": evidence,
    }
    output_root = root / "reports/directional_ranker_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(scored, output_root / "scored_candidates.parquet")
    report = output_root / "summary.json"
    atomic_write_json(payload, report)
    return report


def directional_ranker_v1_status(root: Path = ROOT) -> dict:
    path = root / "reports/directional_ranker_v1/summary.json"
    if not path.exists():
        raise FileNotFoundError("Run build-directional-ranker-v1 first")
    return json.loads(path.read_text(encoding="utf-8"))
