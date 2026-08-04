from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .contextual_pipeline import (
    contextual_defensive_status,
    contextual_predictions_artifact,
)
from .downside_pipeline import (
    downside_risk_artifact,
    downside_risk_gate_status,
    gated_predictions_artifact,
)
from .directional_downside_pipeline import (
    directional_downside_predictions_artifact,
    directional_downside_status,
)
from .uptrend_pipeline import active_model_artifact, active_model_status
from .unified_pipeline import (
    unified_controller_status,
    unified_predictions_artifact,
)
from .io import atomic_write_json, load_workbook, sha256_file
from .targets import build_targets
from .validation import make_layout


ROOT = Path(__file__).resolve().parents[2]
LOCKED_EVALUATION_SHA256 = (
    "04ebedf9455051b189486f61deba949299a499915aa33f11b7126efa5a035b39"
)


def read_config(root: Path = ROOT) -> dict:
    with (root / "configs/config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def audit_data(root: Path = ROOT) -> Path:
    config = read_config(root)
    with (root / "configs/uptrend_model.yaml").open(encoding="utf-8") as handle:
        model_settings = yaml.safe_load(handle)
    source = root / config["data_path"]
    frame = load_workbook(source)
    targets = build_targets(frame)
    layout = make_layout(len(frame), int(config["audit_origins"]))
    indicators = [column for column in frame.columns if column.startswith("X")]
    profile: dict[str, Any] = {
        "source_sha256": sha256_file(source),
        "worksheet": "Sheet1",
        "n_rows": int(len(frame)),
        "n_indicators": int(len(indicators)),
        "date_min": str(frame["Dates"].min().date()),
        "date_max": str(frame["Dates"].max().date()),
        "target_rows": int(targets["target_available"].sum()),
        "selection_origins": list(model_settings["selection_origins"]),
        "locked_evaluation_origins": [
            int(layout.audit_origins[0]),
            int(layout.audit_origins[-1]),
        ],
        "production_origin": int(layout.production_origin),
        "series": {},
    }
    for indicator in indicators:
        values = frame[indicator]
        valid = values.dropna()
        profile["series"][indicator] = {
            "observations": int(valid.size),
            "leading_missing": int(values.isna().cumprod().sum()),
            "internal_missing": int(
                values.iloc[int(values.isna().cumprod().sum()):].isna().sum()
            ),
            "zero_changes": int(valid.diff().eq(0).sum()),
        }
    output = root / "reports/data_profile.json"
    atomic_write_json(profile, output)
    lines = [
        "# Data audit",
        "",
        f"- Source SHA-256: `{profile['source_sha256']}`",
        f"- Shape: `{profile['n_rows']}` months × `{profile['n_indicators']}` indicators",
        f"- Date range: `{profile['date_min']}` to `{profile['date_max']}`",
        "- Leading missing history is preserved; no interpolation or backfill is used.",
        "- The source workbook is treated as immutable.",
    ]
    (root / "reports/data_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return output


def check_project(root: Path = ROOT) -> Path:
    status = active_model_status(root)
    locked_evaluation = root / "artifacts/audit/locked_evaluation.parquet"
    locked_hash = sha256_file(locked_evaluation) if locked_evaluation.exists() else None
    checks = {
        "active_model": "uptrend_selector",
        "active_model_artifact_exists": active_model_artifact(root).exists(),
        "active_model_ready": bool(status.get("ready")),
        "registered_result_matches": bool(
            status.get("registered_result_matches", False)
        ),
        "locked_evaluation_sha256": locked_hash,
        "locked_evaluation_preserved": locked_hash == LOCKED_EVALUATION_SHA256,
        "locked_evaluation_used_by_active_model": False,
        "claim": "artifact_integrity_only",
    }
    risk_status = downside_risk_gate_status(root)
    checks.update({
        "downside_risk_gate_ready": bool(risk_status.get("ready")),
        "downside_risk_artifact_exists": downside_risk_artifact(root).exists(),
        "gated_predictions_artifact_exists": gated_predictions_artifact(
            root
        ).exists(),
        "downside_risk_gate_promoted": False,
        "downside_risk_gate_locked_evaluation_read": bool(
            risk_status.get("locked_evaluation_read", False)
        ),
    })
    directional_status = directional_downside_status(root)
    checks.update({
        "directional_downside_selector_ready": bool(
            directional_status.get("ready")
        ),
        "directional_downside_predictions_exist": (
            directional_downside_predictions_artifact(root).exists()
        ),
        "directional_downside_promotion_eligible": bool(
            directional_status.get("promotion_eligible", False)
        ),
        "directional_downside_promoted": False,
        "directional_downside_locked_evaluation_read": bool(
            directional_status.get("locked_evaluation_read", False)
        ),
    })
    context_status = contextual_defensive_status(root)
    checks.update({
        "contextual_defensive_selector_ready": bool(
            context_status.get("ready")
        ),
        "contextual_predictions_artifact_exists": (
            contextual_predictions_artifact(root).exists()
        ),
        "contextual_defensive_selector_promotion_eligible": bool(
            context_status.get("promotion_eligible", False)
        ),
        "contextual_defensive_selector_promoted": False,
        "contextual_defensive_selector_locked_evaluation_read": bool(
            context_status.get("locked_evaluation_read", False)
        ),
    })
    unified_status = unified_controller_status(root)
    checks.update({
        "unified_controller_ready": bool(unified_status.get("ready")),
        "unified_controller_artifact_exists": unified_predictions_artifact(root).exists(),
        "unified_controller_promotion_eligible": bool(
            unified_status.get("promotion_eligible", False)
        ),
        "unified_controller_promoted": False,
        "unified_controller_locked_evaluation_read": bool(
            unified_status.get("locked_evaluation_read", False)
        ),
    })
    output = root / "reports/project_status.json"
    atomic_write_json(checks, output)
    return output
