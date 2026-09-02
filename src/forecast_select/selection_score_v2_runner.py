from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io import atomic_write_json, atomic_write_parquet
from .selection_score_v2 import (
    FEATURE_COLUMNS, fit_selection_score_v2, score_metrics,
    score_selection_candidates, select_with_existing_caps, selection_metrics,
)
from .uptrend_pipeline import ROOT


def build_selection_score_v2_audit(root: Path = ROOT) -> Path:
    source = pd.read_parquet(root / "artifacts/active/regime_adaptive_predictions.parquet")
    if int(source["origin_position"].max()) >= 268:
        raise ValueError("Selection-score source includes locked origins")
    fitted = fit_selection_score_v2(source)
    scored = select_with_existing_caps(score_selection_candidates(source, fitted))
    windows = {
        "tuning": (120, 179), "validation": (180, 219),
        "confirmation_descriptive": (220, 266),
    }
    evidence = {}
    for name, bounds in windows.items():
        current = scored[scored["origin_position"].between(*bounds)]
        evidence[name] = {
            "baseline_score": score_metrics(current, "selection_score"),
            "selection_score_v2": score_metrics(current, "selection_score_v2"),
            "baseline_selection": selection_metrics(current, "accepted"),
            "selection_v2": selection_metrics(current, "accepted_v2"),
        }
    validation = evidence["validation"]
    accepted = bool(
        validation["selection_score_v2"]["auc"] > validation["baseline_score"]["auc"]
        and validation["selection_score_v2"]["auc"] >= 0.55
        and validation["selection_v2"]["accuracy"] > validation["baseline_selection"]["accuracy"]
    )
    payload = {
        "experiment_id": "selection_score_v2",
        "decision": "candidate_passed_validation" if accepted else "candidate_rejected",
        "accepted": accepted,
        "selected_regularization_c": fitted.regularization_c,
        "late_tuning_auc_used_for_c_selection": fitted.tuning_auc,
        "features": FEATURE_COLUMNS,
        "training_origins": [120, 179],
        "locked_evaluation_read": False,
        "windows": evidence,
    }
    output_root = root / "reports/selection_score_v2"
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(scored, output_root / "scored_candidates.parquet")
    report = output_root / "summary.json"
    atomic_write_json(payload, report)
    return report


def selection_score_v2_status(root: Path = ROOT) -> dict:
    path = root / "reports/selection_score_v2/summary.json"
    if not path.exists():
        raise FileNotFoundError("Run build-selection-score-v2 first")
    return json.loads(path.read_text(encoding="utf-8"))
