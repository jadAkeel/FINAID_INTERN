from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .chronos_cache import (
    CACHE_VERSION,
    ChronosCacheKey,
    build_chronos_cache,
    load_chronos_cache,
    load_chronos_config,
)
from .chronos_evaluation import (
    MAX_EVALUATION_ORIGIN,
    evaluate_chronos_zero_shot,
)
from .chronos_protocol import (
    CHRONOS_MODEL_ID,
    CHRONOS_MODEL_REVISION,
    CHRONOS_PACKAGE_PIN,
    DEFAULT_QUANTILES,
    ChronosProvider,
)
from .io import atomic_write_json, atomic_write_parquet, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_chronos_zero_shot_config(
    root: Path = ROOT,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate Chronos-2 zero-shot configuration."""
    if path is not None:
        target = Path(path)
    elif (root / "configs/chronos_zero_shot.yaml").exists():
        target = root / "configs/chronos_zero_shot.yaml"
    else:
        target = ROOT / "configs/chronos_zero_shot.yaml"
    return load_chronos_config(target)


def read_active_predictions(
    root: Path = ROOT,
    active_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read active model predictions.
    Enforces that parquet must be read with filter origin_position <= 266 and then asserted.
    Never reads audit artifacts or locked evaluation rows.
    """
    target = (
        Path(active_path)
        if active_path is not None
        else root / "artifacts/active/regime_adaptive_predictions.parquet"
    )
    if not target.exists():
        raise FileNotFoundError(f"Active predictions artifact not found at {target}")

    # Read with pyarrow filter origin_position <= 266
    frame = pd.read_parquet(target, filters=[("origin_position", "<=", MAX_EVALUATION_ORIGIN)])

    if frame.empty:
        raise ValueError(f"Active predictions from {target} is empty")

    max_orig = int(frame["origin_position"].max())
    if max_orig > MAX_EVALUATION_ORIGIN:
        raise AssertionError(
            f"Active predictions origin {max_orig} exceeds maximum allowed {MAX_EVALUATION_ORIGIN}; "
            f"origins >= 267 must never be read."
        )

    if "locked_evaluation_read" in frame.columns:
        if frame["locked_evaluation_read"].fillna(False).astype(bool).any():
            raise AssertionError("Active predictions contains locked_evaluation_read=True")

    return frame


def build_chronos_zero_shot(
    root: Path = ROOT,
    provider: ChronosProvider | None = None,
    active_predictions: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    inputs: pd.DataFrame | None = None,
    outcomes: pd.DataFrame | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Build and evaluate the Chronos-2 zero-shot research pipeline:
    - Reuses or builds Chronos cache
    - Reads active model predictions filtered <= 266
    - Runs pure evaluator returning predictions, selected comparison, and JSON summary
    - Writes artifacts and metrics atomically under research/chronos_zero_shot/
    """
    cfg = config if config is not None else load_chronos_zero_shot_config(root)

    # 1. Resolve inputs and outcomes from cache or direct injection
    if inputs is None or outcomes is None:
        data_path = root / cfg.get("data_path", "data/monthly_indicators.xlsx")
        data_hash = sha256_file(data_path) if data_path.exists() else "synthetic"
        config_hash = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        key = ChronosCacheKey(
            source_data_hash=data_hash,
            config_hash=config_hash,
            package_pin=str(cfg.get("package_pin", CHRONOS_PACKAGE_PIN)),
            model_id=str(cfg.get("model_id", CHRONOS_MODEL_ID)),
            model_revision=str(cfg.get("model_revision", CHRONOS_MODEL_REVISION)),
            cache_version=CACHE_VERSION,
            seed=int(cfg.get("seed", 20260727)),
            origin_start=int(cfg.get("tuning_origins", [120, 179])[0]),
            origin_end=int(cfg.get("confirmation_origins", [220, 266])[1]),
            prediction_length=int(cfg.get("prediction_length", 2)),
            availability_lag=int(cfg.get("availability_lag", 1)),
            quantiles=tuple(float(q) for q in cfg.get("quantiles", DEFAULT_QUANTILES)),
        )

        try:
            inputs, outcomes = load_chronos_cache(root, key)
        except Exception:
            # Lazy ChronosRealProvider only when live cache build needed
            if provider is None:
                from .chronos_protocol import ChronosRealProvider
                provider = ChronosRealProvider()
            build_chronos_cache(root, cfg, provider, data_path=data_path)
            inputs, outcomes = load_chronos_cache(root, key)

    # 2. Read active predictions
    if active_predictions is None:
        active_predictions = read_active_predictions(root)

    # 3. Run pure evaluation
    predictions_table, matched_comparison, summary = evaluate_chronos_zero_shot(
        chronos_inputs=inputs,
        chronos_outcomes=outcomes,
        active_predictions=active_predictions,
        config=cfg,
        min_history_origins=int(cfg.get("minimum_history_origins", 12)),
        bootstrap_block_size=int(cfg.get("bootstrap_block_size", 6)),
        bootstrap_replicates=int(cfg.get("bootstrap_replicates", 500)),
        seed=int(cfg.get("seed", 20260727)),
    )

    # 4. Atomic writes
    base_out = output_dir if output_dir is not None else root / "research/chronos_zero_shot"
    artifacts_dir = base_out / "artifacts"
    metrics_dir = base_out / "metrics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_parquet(predictions_table, artifacts_dir / "predictions.parquet")
    atomic_write_parquet(matched_comparison, artifacts_dir / "matched_coverage.parquet")
    atomic_write_json(summary, metrics_dir / "summary.json")

    # CSV metrics summaries
    window_rows = []
    for win, data in summary.get("windows", {}).items():
        for mod in ["chronos_raw", "chronos_calibrated", "active", "blend"]:
            m = data.get(mod, {})
            window_rows.append({
                "window": win,
                "model": mod,
                "months": data.get("months"),
                "common_eligible_rows": data.get("common_eligible_rows"),
                "calls": m.get("n"),
                "hits": m.get("hits"),
                "accuracy": m.get("accuracy"),
                "brier": m.get("brier"),
                "log_loss": m.get("log_loss"),
                "roc_auc": m.get("roc_auc"),
                "ece": m.get("ece"),
            })
    if window_rows:
        atomic_write_csv(pd.DataFrame(window_rows), metrics_dir / "window_metrics.csv")

    diversity_rows = []
    for win, d in summary.get("diversity", {}).items():
        diversity_rows.append({
            "window": win,
            "n": d.get("n"),
            "pearson": d.get("pearson"),
            "spearman": d.get("spearman"),
            "disagreement_count": d.get("disagreement_count"),
            "disagreement_rate": d.get("disagreement_rate"),
            "chronos_accuracy_on_disagreements": d.get("chronos_accuracy_on_disagreements"),
            "active_accuracy_on_disagreements": d.get("active_accuracy_on_disagreements"),
            "active_wrong_chronos_correct_count": d.get("active_wrong_chronos_correct_count"),
            "double_fault_count": d.get("double_fault_count"),
            "double_fault_rate": d.get("double_fault_rate"),
        })
    if diversity_rows:
        atomic_write_csv(pd.DataFrame(diversity_rows), metrics_dir / "diversity_metrics.csv")

    matched_rows = []
    for win, m in summary.get("matched_coverage", {}).items():
        matched_rows.append({
            "window": win,
            "origin_count": m.get("origin_count"),
            "total_calls": m.get("total_calls"),
            "candidate_hits": m.get("candidate_hits"),
            "active_hits": m.get("active_hits"),
            "total_hit_delta": m.get("total_hit_delta"),
            "mean_monthly_hit_delta": m.get("mean_monthly_hit_delta"),
            "bootstrap_p10": m.get("bootstrap_p10"),
            "bootstrap_p50": m.get("bootstrap_p50"),
            "bootstrap_p90": m.get("bootstrap_p90"),
        })
    if matched_rows:
        atomic_write_csv(pd.DataFrame(matched_rows), metrics_dir / "matched_coverage.csv")

    return summary


def chronos_zero_shot_status(
    root: Path = ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Read and validate Chronos-2 zero-shot status:
    - Validates locked_evaluation_read is False
    - Validates max origin <= 266
    - Validates promotion_eligible is False and allow_promotion is False
    """
    base_out = output_dir if output_dir is not None else root / "research/chronos_zero_shot"
    summary_path = base_out / "metrics/summary.json"
    pred_path = base_out / "artifacts/predictions.parquet"

    if not summary_path.exists():
        return {
            "ready": False,
            "experiment_id": "chronos_zero_shot",
            "summary_path": str(summary_path),
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Strict validations
    if summary.get("locked_evaluation_read") is not False:
        raise AssertionError("Summary reports locked_evaluation_read is not False")
    if int(summary.get("maximum_origin_position", 0)) > MAX_EVALUATION_ORIGIN:
        raise AssertionError(
            f"Summary maximum_origin_position {summary.get('maximum_origin_position')} exceeds {MAX_EVALUATION_ORIGIN}"
        )
    if summary.get("allow_promotion") is not False:
        raise AssertionError("Summary reports allow_promotion is not False")
    if summary.get("promotion_eligible") is not False:
        raise AssertionError("Summary reports promotion_eligible is not False")

    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        if int(preds["origin_position"].max()) > MAX_EVALUATION_ORIGIN:
            raise AssertionError("Predictions artifact contains origins > 266")
        if "locked_evaluation_read" in preds.columns:
            if preds["locked_evaluation_read"].fillna(False).astype(bool).any():
                raise AssertionError("Predictions artifact contains locked_evaluation_read=True")

    summary["ready"] = True
    return summary
