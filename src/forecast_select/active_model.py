from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .regime_adaptive_pipeline import (
    ROOT,
    build_regime_adaptive_selector,
    regime_adaptive_predictions_artifact,
    regime_adaptive_status,
)
from .schemas import validate_oof_columns


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def active_model_artifact(root: Path = ROOT) -> Path:
    settings = _read_yaml(root / "configs/active_model.yaml")
    return root / settings["active_artifact"]


def _selected_summary(
    predictions: pd.DataFrame,
    bounds: list[int],
) -> dict[str, float | int | None]:
    selected = predictions[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["origin_position"].between(int(bounds[0]), int(bounds[1]))
        & predictions["y_true"].notna()
    ].copy()
    correct = selected["predicted_direction"].eq(
        selected["y_true"].astype(int).map({1: "Up", 0: "Down"})
    )
    down = selected[selected["predicted_direction"].eq("Down")]
    down_hits = int(down["y_true"].eq(0).sum())
    return {
        "months": int(selected["origin_position"].nunique()),
        "calls": int(len(selected)),
        "hits": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(selected) else None,
        "up_calls": int(selected["predicted_direction"].eq("Up").sum()),
        "down_calls": int(len(down)),
        "down_hits": down_hits,
        "down_precision": float(down_hits / len(down)) if len(down) else None,
    }


def _validate_active_predictions(
    predictions: pd.DataFrame,
    settings: dict,
    root: Path,
) -> None:
    validate_oof_columns(predictions.columns.tolist())
    if predictions.empty:
        raise AssertionError("Active model artifact is empty")
    if predictions["locked_evaluation_read"].fillna(False).astype(bool).any():
        raise AssertionError("Active model must not read the locked evaluation")
    if int(predictions["origin_position"].max()) >= int(settings["locked_origins"][0]):
        raise AssertionError("Active model artifact crosses the locked boundary")
    if set(predictions["model_id"].dropna().astype(str).unique()) != {
        settings["model_id"]
    }:
        raise AssertionError("Active model id does not match active_model.yaml")
    if not predictions["active_model"].fillna(False).astype(bool).all():
        raise AssertionError("Active model artifact is missing its activation marker")
    if not predictions["activation_status"].eq(
        settings["activation_status"]
    ).all():
        raise AssertionError("Active model activation status is inconsistent")
    if not predictions["predicted_direction"].dropna().isin({"Up", "Down"}).all():
        raise AssertionError("Active model contains an unsupported direction")

    selected = predictions[predictions["accepted"].fillna(False).astype(bool)]
    monthly = selected.groupby("origin_position")["indicator_id"].agg(
        ["count", "nunique"]
    )
    if monthly.empty or not monthly["count"].eq(monthly["nunique"]).all():
        raise AssertionError("Active model must select unique indicators each month")
    if not monthly["count"].between(15, 20).all():
        raise AssertionError("Active model must select between 15 and 20 indicators")
    if not set(monthly["count"].unique()).issubset({15, 16, 17, 18, 19, 20}):
        raise AssertionError("Active model cap must be in 15..20")

    project = _read_yaml(root / "configs/config.yaml")
    expected_data_hash = sha256_file(root / project["data_path"])
    if set(predictions["data_hash"].dropna().astype(str).unique()) != {
        expected_data_hash
    }:
        raise AssertionError("Active model data hash does not match the input workbook")


def _validate_source_status(
    predictions: pd.DataFrame,
    source_status: dict,
) -> None:
    expected = source_status.get("config_hash")
    actual = set(predictions["config_hash"].dropna().astype(str).unique())
    if not expected or actual != {str(expected)}:
        raise AssertionError(
            "Active model artifact is stale relative to the regime research config"
        )


def _write_active_report(
    predictions: pd.DataFrame,
    source_status: dict,
    settings: dict,
    root: Path,
) -> dict:
    windows = {
        name: _selected_summary(predictions, bounds)
        for name, bounds in settings["evaluation_windows"].items()
    }
    overall_bounds = [
        settings["evaluation_windows"]["tuning"][0],
        settings["evaluation_windows"]["confirmation"][1],
    ]
    summary = {
        "ready": True,
        "model_id": settings["model_id"],
        "model_name": settings["model_name"],
        "model_release": settings["model_release"],
        "activation_status": settings["activation_status"],
        "activation_basis": settings["activation_basis"],
        "baseline_model_id": settings["baseline_model_id"],
        "research_gate_passed": bool(settings["research_gate_passed"]),
        "research_promotion_eligible": bool(
            source_status.get("promotion_eligible", False)
        ),
        "owner_promoted": settings["activation_status"] == "owner_promoted",
        "active_model_changed": True,
        "locked_evaluation_read": False,
        "locked_origins": list(settings["locked_origins"]),
        "artifact": active_model_artifact(root).relative_to(root).as_posix(),
        "source_artifact": settings["source_artifact"],
        "source_config_hash": source_status.get("config_hash"),
        "overall_nonlocked": _selected_summary(predictions, overall_bounds),
        "windows": windows,
    }
    atomic_write_json(summary, root / "reports/model_performance.json")

    overall = summary["overall_nonlocked"]
    lines = [
        "# Regime Adaptive Bidirectional Selector performance",
        "",
        "This is the owner-promoted active model. The research promotion gate did not pass; activation is an explicit product decision to support both Up and Down directions.",
        "",
        f"- Non-locked hits / calls: `{overall['hits']} / {overall['calls']}`",
        f"- Non-locked accuracy: `{overall['accuracy']:.4%}`",
        f"- Up / Down calls: `{overall['up_calls']} / {overall['down_calls']}`",
        f"- Research promotion eligible: `{summary['research_promotion_eligible']}`",
        "- Locked evaluation read: `False`",
        "",
        "The Uptrend Selector remains the reproducible baseline used by the research pipeline.",
    ]
    (root / "reports/model_performance.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return summary


def build_active_model(root: Path = ROOT) -> Path:
    settings = _read_yaml(root / "configs/active_model.yaml")
    source = regime_adaptive_predictions_artifact(root)
    try:
        source_status = regime_adaptive_status(root)
    except FileNotFoundError:
        build_regime_adaptive_selector(root)
        source_status = regime_adaptive_status(root)
    if not source.exists():
        build_regime_adaptive_selector(root)
        source_status = regime_adaptive_status(root)

    predictions = pd.read_parquet(source)
    predictions = predictions.copy()
    predictions["active_model"] = True
    predictions["activation_status"] = settings["activation_status"]
    predictions["activation_basis"] = settings["activation_basis"]
    predictions["research_promotion_eligible"] = bool(
        source_status.get("promotion_eligible", False)
    )
    predictions["source_experiment_artifact"] = settings["source_artifact"]
    _validate_active_predictions(predictions, settings, root)
    _validate_source_status(predictions, source_status)
    atomic_write_parquet(predictions, active_model_artifact(root))
    _write_active_report(predictions, source_status, settings, root)
    return active_model_artifact(root)


def active_model_status(root: Path = ROOT) -> dict:
    settings = _read_yaml(root / "configs/active_model.yaml")
    output = active_model_artifact(root)
    if not output.exists():
        return {
            "ready": False,
            "model_id": settings["model_id"],
            "artifact": str(output),
        }
    predictions = pd.read_parquet(output)
    _validate_active_predictions(predictions, settings, root)
    source_status = regime_adaptive_status(root)
    _validate_source_status(predictions, source_status)
    return _write_active_report(predictions, source_status, settings, root)
