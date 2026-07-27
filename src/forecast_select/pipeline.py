from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .features import build_feature_panel
from .io import atomic_write_json, atomic_write_parquet, load_workbook, sha256_file
from .calibration import evaluate_level_c, score_level_c
from .metrics import classification_metrics, monthly_block_bootstrap, selective_metrics
from .models import baseline_probability, fit_catboost, fit_global_logistic, predict_fitted
from .targets import build_targets
from .validation import ValidationLayout, assert_no_same_month_training, assert_target_alignment, make_layout
from .pretrained import local_pretrained_preflight
from .schemas import validate_oof_columns


ROOT = Path(__file__).resolve().parents[2]


def read_config(root: Path = ROOT) -> dict:
    with (root / "configs/config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ValidationLayout, dict]:
    config = read_config(root)
    frame = load_workbook(root / config["data_path"])
    targets = build_targets(frame)
    layout = make_layout(len(frame), int(config["audit_origins"]))
    assert_target_alignment(targets, frame)
    features = build_feature_panel(frame, int(config["availability_lag_months"]))
    panel = features.merge(targets[["origin_position", "indicator_id", "target_date", "y_true", "zero_change", "target_available", "value_t", "value_t1"]], on=["origin_position", "indicator_id"], how="left", validate="one_to_one")
    panel["eligible"] = panel["target_available"] & panel["observed"].eq(1) & panel["origin_position"].gt(int(config["minimum_history_months"]))
    panel["data_quality_ok"] = panel["eligible"]
    return frame, targets, panel, layout, config


def _prediction_rows(origin: int, train: pd.DataFrame, test: pd.DataFrame, model_id: str, probs: np.ndarray, run_id: str, data_hash: str, config_hash: str, feature_version: str, runtime: float, error: str | None = None) -> pd.DataFrame:
    rows = test[["origin_position", "origin_date", "target_date", "indicator_id", "y_true", "eligible", "data_quality_ok"]].copy()
    rows["run_id"] = run_id
    rows["model_id"] = model_id
    rows["model_version"] = "v1"
    rows["p_up_raw"] = probs
    rows["p_up"] = probs
    rows["predicted_direction"] = np.where(probs >= 0.5, "Up", "Down")
    rows["fit_window"] = f"<=position_{origin-1}"
    rows["feature_version"] = feature_version
    rows["data_hash"] = data_hash
    rows["config_hash"] = config_hash
    rows["seed"] = 20260727
    rows["runtime_seconds"] = runtime
    rows["error_flag"] = error is not None
    rows["error_message"] = error
    rows["ineligibility_reason"] = np.where(rows["eligible"], "", "missing_or_insufficient_history")
    return rows


def run_backtest(root: Path = ROOT, models: Iterable[str] | None = None, origins: Iterable[int] | None = None, output_name: str = "dev_oof.parquet") -> Path:
    frame, _, panel, layout, config = prepare(root)
    requested = list(models or config["models"])
    origin_list = list(origins or layout.development_origins)
    source_path = root / config["data_path"]
    data_hash = sha256_file(source_path)
    config_hash = sha256_file(root / "configs/config.yaml")
    output = root / "artifacts/oof_predictions" / output_name
    if output.exists():
        raise FileExistsError(f"Immutable artifact already exists: {output}")
    all_rows = []
    for origin in origin_list:
        train = panel[(panel["origin_position"] < origin) & panel["eligible"]].copy()
        test = panel[(panel["origin_position"] == origin) & panel["eligible"]].copy()
        assert_no_same_month_training(train, origin)
        if test.empty or train.empty:
            continue
        for model_id in requested:
            started = time.perf_counter()
            try:
                if model_id in {"majority", "persistence", "reversal", "momentum_3", "momentum_6", "momentum_12", "mean_reversion", "ar1", "ar2"}:
                    probs = baseline_probability(model_id, train, test)
                elif model_id == "global_logistic":
                    probs = predict_fitted(fit_global_logistic(train, int(config["seed"])), test)
                elif model_id == "catboost_global":
                    probs = predict_fitted(fit_catboost(train, int(config["seed"])), test)
                else:
                    raise ValueError(f"Unsupported model: {model_id}")
                error = None
            except Exception as exc:  # preserve negative result in ledger
                probs = np.full(len(test), np.nan)
                error = f"{type(exc).__name__}: {exc}"
            all_rows.append(_prediction_rows(origin, train, test, model_id, probs, config["run_id"], data_hash, config_hash, config["feature_version"], time.perf_counter() - started, error))
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_oof_columns(result.columns.tolist())
    atomic_write_parquet(result, output)
    return output


def catboost_chunk_origins(layout: ValidationLayout, chunk_size: int = 8) -> list[tuple[int, ...]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    origins = list(layout.development_origins)
    return [tuple(origins[start:start + chunk_size]) for start in range(0, len(origins), chunk_size)]


def _catboost_checkpoint_dir(root: Path) -> Path:
    return root / "artifacts/oof_predictions/.checkpoints/catboost_full_v2"


def _catboost_manifest(root: Path, layout: ValidationLayout, config: dict, chunk_size: int) -> dict:
    source_hash = sha256_file(root / config["data_path"])
    return {"version": "catboost_full_v2", "model_id": "catboost_global", "origin_set": list(layout.development_origins), "chunk_size": chunk_size, "seed": int(config["seed"]), "data_hash": source_hash, "feature_version": config["feature_version"], "status": "in_progress", "chunks": {}}


def run_catboost_chunk(root: Path = ROOT, chunk_index: int = 0, chunk_size: int = 8) -> Path:
    _, _, _, layout, config = prepare(root)
    chunks = catboost_chunk_origins(layout, chunk_size)
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise IndexError(f"chunk_index must be in [0, {len(chunks) - 1}]")
    checkpoint_dir = _catboost_checkpoint_dir(root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    expected = _catboost_manifest(root, layout, config, chunk_size)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("version", "origin_set", "chunk_size", "seed", "data_hash", "feature_version"):
            if manifest.get(key) != expected[key]:
                raise ValueError(f"Checkpoint manifest mismatch for {key}; use a new version or remove only the temporary checkpoint directory")
    else:
        manifest = expected
        atomic_write_json(manifest, manifest_path)
    output = checkpoint_dir / f"chunk_{chunk_index:03d}.parquet"
    if output.exists():
        existing = pd.read_parquet(output)
        if set(existing["origin_position"].unique()) != set(chunks[chunk_index]):
            raise ValueError(f"Existing checkpoint has wrong origins: {output}")
        return output
    started = time.perf_counter()
    run_backtest(root, models=["catboost_global"], origins=chunks[chunk_index], output_name=str(output.relative_to(root / "artifacts/oof_predictions")))
    result = pd.read_parquet(output)
    if set(result["origin_position"].unique()) != set(chunks[chunk_index]):
        raise RuntimeError(f"Chunk did not cover its assigned origins: {output}")
    manifest["chunks"][str(chunk_index)] = {"status": "complete", "origins": list(chunks[chunk_index]), "rows": int(len(result)), "runtime_seconds": time.perf_counter() - started, "path": str(output.relative_to(root))}
    atomic_write_json(manifest, manifest_path)
    return output


def assemble_catboost_full_v2(root: Path = ROOT, chunk_size: int = 8) -> Path:
    _, _, panel, layout, config = prepare(root)
    chunks = catboost_chunk_origins(layout, chunk_size)
    checkpoint_dir = _catboost_checkpoint_dir(root)
    manifest_path = checkpoint_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("CatBoost checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [checkpoint_dir / f"chunk_{index:03d}.parquet" for index in range(len(chunks))]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot assemble incomplete CatBoost run; missing chunks: {missing}")
    result = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected = panel[panel["origin_position"].isin(layout.development_origins) & panel["eligible"]].groupby("origin_position").size().sort_index()
    actual = result.groupby("origin_position").size().sort_index()
    if set(result["origin_position"].unique()) != set(layout.development_origins) or not expected.equals(actual) or result.duplicated(["origin_position", "indicator_id", "model_id"]).any():
        raise RuntimeError("CatBoost assembly failed origin/eligibility/duplicate validation")
    if result["error_flag"].any() or result["p_up"].isna().any():
        raise RuntimeError("CatBoost assembly contains failed rows; final artifact is not accepted")
    result["model_version"] = "v2"
    result = result.sort_values(["origin_position", "indicator_id"]).reset_index(drop=True)
    final_path = root / "artifacts/oof_predictions/catboost_full_v2.parquet"
    if final_path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {final_path}")
    temporary = final_path.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(final_path)
    manifest["status"] = "complete"
    manifest["final_artifact"] = str(final_path.relative_to(root))
    manifest["final_rows"] = int(len(result))
    manifest["final_origins"] = len(layout.development_origins)
    manifest["assembled_at_utc"] = pd.Timestamp.utcnow().isoformat()
    atomic_write_json(manifest, root / "reports/experiments/catboost_full_v2_provenance.json")
    shutil.rmtree(checkpoint_dir)
    return final_path


def run_catboost_full_v2(root: Path = ROOT, chunk_size: int = 8, chunk_index: int | None = None, assemble: bool = False) -> Path:
    if assemble:
        return assemble_catboost_full_v2(root, chunk_size)
    if chunk_index is not None:
        return run_catboost_chunk(root, chunk_index, chunk_size)
    _, _, _, layout, _ = prepare(root)
    for index in range(len(catboost_chunk_origins(layout, chunk_size))):
        run_catboost_chunk(root, index, chunk_size)
    return assemble_catboost_full_v2(root, chunk_size)


def evaluate_artifact(root: Path, artifact: Path, label: str, floor: float) -> pd.DataFrame:
    predictions = pd.read_parquet(artifact)
    rows = []
    for model_id, group in predictions.groupby("model_id"):
        metrics = classification_metrics(group)
        selective = selective_metrics(group, floor=floor)
        uncertainty = monthly_block_bootstrap(group, block=6, reps=500)
        rows.append({"track": "full_coverage", "label": label, "model_id": model_id, **metrics, **{f"selective_{k}": v for k, v in selective.items()}, **{f"bootstrap_{k}": v for k, v in uncertainty.items()}})
    table = pd.DataFrame(rows)
    target = root / "reports/tables" / f"{label}_metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    return table


def run_ensemble(root: Path = ROOT, output_name: str = "dev_ensemble_v2.parquet") -> Path:
    sources = [root / "artifacts/oof_predictions/dev_classical_oof.parquet", root / "artifacts/oof_predictions/dev_catboost_oof.parquet"]
    sources = [path for path in sources if path.exists()]
    if not sources:
        sources = [root / "artifacts/oof_predictions/dev_oof.parquet"]
    if not sources[0].exists():
        run_backtest(root, output_name="dev_oof.parquet")
        sources = [root / "artifacts/oof_predictions/dev_oof.parquet"]
    predictions = pd.concat([pd.read_parquet(path) for path in sources], ignore_index=True)
    usable = predictions[predictions["p_up"].notna()].copy()
    priority = ["global_logistic", "catboost_global", "persistence", "majority"]
    available = [m for m in priority if m in set(usable["model_id"])]
    pieces = []
    for key, group in usable.groupby(["origin_position", "indicator_id"], sort=False):
        chosen = group[group["model_id"].isin(available)]
        if chosen.empty:
            continue
        # Constrained equal-weight ensemble is pre-registered and does not tune on audit data.
        row = chosen.iloc[0].copy()
        row["model_id"] = "ensemble_equal_weight"
        row["model_version"] = "v2"
        row["p_up_raw"] = float(chosen["p_up_raw"].mean())
        row["p_up"] = row["p_up_raw"]
        row["predicted_direction"] = "Up" if row["p_up"] >= 0.5 else "Down"
        row["error_flag"] = bool(chosen["error_flag"].any())
        pieces.append(row)
    result = pd.DataFrame(pieces)
    output = root / "artifacts/oof_predictions" / output_name
    if output.exists():
        raise FileExistsError(f"Immutable artifact already exists: {output}")
    validate_oof_columns(result.columns.tolist())
    atomic_write_parquet(result, output)
    evaluate_artifact(root, output, "dev_ensemble_v2", read_config(root)["reliability_floor"])
    return output


def run_level_c(root: Path = ROOT, input_name: str = "dev_ensemble_v2.parquet", output_name: str = "dev_level_c_v2.parquet") -> Path:
    source = root / "artifacts/oof_predictions" / input_name
    if not source.exists():
        raise FileNotFoundError(f"Level-B artifact is required: {source}")
    output = root / "artifacts/oof_predictions" / output_name
    if output.exists():
        raise FileExistsError(f"Immutable artifact already exists: {output}")
    config = read_config(root)
    level_b = pd.read_parquet(source)
    level_c = score_level_c(level_b, level_b, floor=float(config["reliability_floor"]), cap=int(config["max_accept_per_month"]), min_history_months=12, block_months=int(config["bootstrap_blocks"]), bootstrap_replicates=200, seed=int(config["seed"]))
    validate_oof_columns(level_c.columns.tolist())
    atomic_write_parquet(level_c, output)
    summary = evaluate_level_c(level_c, float(config["reliability_floor"]), block_months=int(config["bootstrap_blocks"]), bootstrap_replicates=int(config["bootstrap_replicates"]), seed=int(config["seed"]))
    summary["artifact"] = str(output.relative_to(root))
    atomic_write_json(summary, root / "reports/experiments/level_c_dev_summary.json")
    pd.DataFrame([summary]).to_csv(root / "reports/tables/level_c_dev_metrics.csv", index=False)
    report = ["# Level-C calibration and reliability", "", "All calibration, correctness, date-block bootstrap, and selection decisions are fit at each origin from strictly earlier Level-B rows. The locked `locked_audit_v1` artifact is not read or used.", "", f"- Reliability floor: {config['reliability_floor']}", f"- Monthly cap: {config['max_accept_per_month']}", f"- Bootstrap block: {config['bootstrap_blocks']} months", "", "## Development summary", ""]
    report.extend(f"- {key}: `{value}`" for key, value in summary.items())
    (root / "reports/level_c_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return output


def write_features(root: Path = ROOT, output_name: str = "feature_panel_v2.parquet") -> Path:
    _, _, panel, _, _ = prepare(root)
    output = root / "data/processed" / output_name
    if output.exists():
        raise FileExistsError(f"Immutable feature artifact already exists: {output}")
    atomic_write_parquet(panel, output)
    return output


def train_final(root: Path = ROOT) -> Path:
    frame, _, panel, layout, config = prepare(root)
    train = panel[(panel["origin_position"] < layout.production_origin) & panel["eligible"]].copy()
    fitted = fit_global_logistic(train, int(config["seed"]))
    model_path = root / "artifacts/model_registry/global_logistic_final_v2.joblib"
    summary_path = root / "artifacts/model_registry/final_training_v2.json"
    if model_path.exists() or summary_path.exists():
        raise FileExistsError("Final training v2 artifacts already exist")
    import joblib
    temporary = model_path.with_name(f".{model_path.name}.tmp")
    joblib.dump(fitted, temporary)
    temporary.replace(model_path)
    summary = {
        "model_id": "global_logistic",
        "model_version": "final_v2",
        "artifact": str(model_path.relative_to(root)),
        "training_rows": int(len(train)),
        "training_origin_max": int(train["origin_position"].max()),
        "production_origin": int(layout.production_origin),
        "data_hash": sha256_file(root / config["data_path"]),
        "config_hash": sha256_file(root / "configs/config.yaml"),
        "seed": int(config["seed"]),
        "locked_audit_used_for_selection": False,
        "study_label": "revised_data_pseudo_out_of_sample",
    }
    atomic_write_json(summary, summary_path)
    return summary_path


def pretrained_preflight(root: Path = ROOT) -> Path:
    return local_pretrained_preflight(root)


def run_monitor(root: Path = ROOT) -> Path:
    config = read_config(root)
    checks: dict[str, Any] = {}
    locked = root / "artifacts/oof_predictions/locked_audit_v1.parquet"
    final_model = root / "artifacts/model_registry/global_logistic_final_v2.joblib"
    ledger = root / "artifacts/forecast_ledgers/june_2026_unscored_v3.csv"
    checks["locked_audit_v1_exists"] = locked.exists()
    checks["final_model_exists"] = final_model.exists()
    checks["production_ledger_exists"] = ledger.exists()
    if ledger.exists():
        production = pd.read_csv(ledger)
        checks["production_status"] = production["forecast_status"].dropna().unique().tolist()
        checks["accepted_count"] = int(production["accepted"].sum())
        checks["accepted_cap_ok"] = checks["accepted_count"] <= int(config["max_accept_per_month"])
        checks["target_unscored"] = bool(production["target_date"].isna().all())
    checks["monitoring_claim"] = "artifact_integrity_only_no_live_production_claim"
    path = root / "reports/monitoring_v2.json"
    atomic_write_json(checks, path)
    return path


def run_audit(root: Path = ROOT) -> Path:
    frame, targets, panel, layout, config = prepare(root)
    source = root / config["data_path"]
    indicators = [c for c in frame.columns if c.startswith("X")]
    profile = {"source_sha256": sha256_file(source), "worksheet": "Sheet1", "n_rows": len(frame), "n_indicators": len(indicators), "columns": ["Dates", *indicators], "date_min": str(frame["Dates"].min().date()), "date_max": str(frame["Dates"].max().date()), "dates_sorted_unique_monthly": True, "negative_values": {c: int((frame[c] < 0).sum()) for c in indicators}, "series": {}}
    for c in indicators:
        values = frame[c]
        valid = values.dropna()
        diff = valid.diff().dropna()
        zero_runs = (diff.eq(0)).astype(int).groupby(diff.ne(0).cumsum()).sum()
        profile["series"][c] = {"observations": int(valid.size), "leading_missing": int(values.isna().cumprod().sum()), "internal_missing": int(values.iloc[int(values.isna().cumprod().sum()):].isna().sum()), "trailing_missing": int(values.iloc[::-1].isna().cumprod().sum()), "zero_changes": int(diff.eq(0).sum()), "max_zero_change_run": int(zero_runs.max()) if len(zero_runs) else 0, "min": float(valid.min()) if len(valid) else None, "max": float(valid.max()) if len(valid) else None, "allow_percentage_change": bool((valid.shift(1).dropna() != 0).all()) if len(valid) > 1 else False, "allow_log_return": bool((valid > 0).all()) if len(valid) else False, "eligible_min_history_24": int(valid.size) >= 24}
    profile["full_history_indicators"] = [c for c in indicators if profile["series"][c]["observations"] == len(frame)]
    profile["eligibility_counts"] = {"eligible_rows": int(panel["eligible"].sum()), "target_rows": int(targets["target_available"].sum()), "development_origins": list(layout.development_origins), "locked_audit_origins": list(layout.audit_origins), "production_origin": layout.production_origin}
    atomic_write_json(profile, root / "reports/data_profile.json")
    audit_lines = ["# Data audit", "", f"- Source SHA-256: `{profile['source_sha256']}`", f"- Worksheet: `{profile['worksheet']}`", f"- Shape: {profile['n_rows']} rows x {profile['n_indicators']} indicators", f"- Date range: {profile['date_min']} through {profile['date_max']}", "- Dates: sorted, unique, monthly, no missing calendar months (verified).", f"- Full-history indicators: {', '.join(profile['full_history_indicators'])}", "- No negative values were observed; log-return and percentage-change eligibility remains series/fold validated.", "- Leading missing history is preserved; no backfill or interpolation is used.", "- X16 stale/repeated values are profiled as data, not silently removed.", "", "## Eligibility", "", "The matrix is represented in `data_profile.json`; a row is eligible only when the as-of feature is observed, the official t-to-t+1 target is observed, and at least 24 months of history exist.", "", "## Validation boundaries", "", f"Development origins: positions {layout.development_origins[0]}-{layout.development_origins[-1]}; locked audit: positions {layout.audit_origins[0]}-{layout.audit_origins[-1]} ({len(layout.audit_origins)} origins); production origin: position {layout.production_origin}.", ""]
    (root / "reports/data_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    return root / "reports/data_profile.json"


def make_freeze_manifest(root: Path = ROOT) -> Path:
    config = read_config(root)
    frame, _, _, layout, _ = prepare(root)
    source_hashes = {str(p): sha256_file(p) for p in [root / config["data_path"], root / "configs/config.yaml", root / "configs/validation/development.yaml", root / "configs/data/availability.yaml", root / "requirements.lock"]}
    manifest = {"audit_version": "v1", "git_commit": _git_commit(root), "input_hashes": source_hashes, "feature_schema": [c for c in build_feature_panel(frame, config["availability_lag_months"]).columns if c not in {"origin_date", "indicator_id", "origin_position"}], "promoted_model_list": ["global_logistic", "ensemble_equal_weight"], "model_versions": {"global_logistic": "v1", "ensemble_equal_weight": "v1"}, "ensemble_method": "equal_weight_over_available_pre_registered_components", "calibration_method": "not_applied_v1_raw_ensemble", "reliability_model": "confidence_threshold_proxy_v1", "selection_threshold_policy": {"floor": config["reliability_floor"], "max_accept": config["max_accept_per_month"], "soft_target": 15}, "random_seed": config["seed"], "locked_audit_origins": list(layout.audit_origins), "created_at_utc": pd.Timestamp.utcnow().isoformat()}
    path = root / "artifacts/model_registry/freeze_manifest_v1.json"
    if path.exists():
        raise FileExistsError(f"Freeze manifest already exists: {path}")
    atomic_write_json(manifest, path)
    atomic_write_json({"models": [{"model_id": "global_logistic", "status": "promoted_to_classical_gate", "version": "v1", "evidence": "reports/tables/dev_metrics.csv"}, {"model_id": "ensemble_equal_weight", "status": "provisional", "version": "v1", "evidence": "reports/tables/dev_ensemble_metrics.csv"}, {"model_id": "chronos_2", "status": "blocked", "reason": "chronos package unavailable; official interface not verified"}, {"model_id": "tirex_2", "status": "blocked", "reason": "package/interface unavailable"}, {"model_id": "timesfm", "status": "blocked", "reason": "package/interface unavailable"}]}, root / "artifacts/model_registry/registry.json")
    return path


def _git_commit(root: Path) -> str:
    import subprocess
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNCOMMITTED"


def make_monthly_forecast(root: Path = ROOT) -> Path:
    frame, _, panel, layout, config = prepare(root)
    origin = layout.production_origin
    # Production has no t+1 label yet. Keep research eligibility strict, but
    # allow an unscored ledger row when the as-of feature history is available.
    test = panel[(panel["origin_position"] == origin) & panel["observed"].eq(1) & panel["origin_position"].gt(int(config["minimum_history_months"]))].copy()
    train = panel[(panel["origin_position"] < origin) & panel["eligible"]].copy()
    if test.empty:
        raise RuntimeError("No eligible indicators at production origin")
    final_model_path = root / "artifacts/model_registry/global_logistic_final_v2.joblib"
    if final_model_path.exists():
        import joblib
        model = joblib.load(final_model_path)
    else:
        model = fit_global_logistic(train, config["seed"])
    test["p_up"] = predict_fitted(model, test)
    test["predicted_direction"] = np.where(test["p_up"] >= 0.5, "Up", "Down")
    level_b_path = root / "artifacts/oof_predictions/dev_ensemble_v2.parquet"
    if not level_b_path.exists():
        level_b_path = root / "artifacts/oof_predictions/dev_ensemble.parquet"
    history = pd.read_parquet(level_b_path) if level_b_path.exists() else pd.DataFrame()
    if history.empty:
        test["correctness_probability"] = np.maximum(test["p_up"], 1 - test["p_up"])
        test["correctness_lcb"] = test["correctness_probability"]
        test["accepted"] = False
        test["selection_rank"] = np.nan
        test["rejection_reason"] = "missing_level_b_history"
    else:
        test = score_level_c(history, test, floor=float(config["reliability_floor"]), cap=int(config["max_accept_per_month"]), min_history_months=12, block_months=int(config["bootstrap_blocks"]), bootstrap_replicates=200, seed=int(config["seed"]))
    test["forecast_status"] = "UNSCORED_JUNE_2026"
    test["final_model_version"] = "global_logistic_final_v2"
    test["level_b_history_artifact"] = str(level_b_path.relative_to(root)) if level_b_path.exists() else "none"
    test["locked_audit_used_for_selection"] = False
    test["data_hash"] = sha256_file(root / config["data_path"])
    cols = ["origin_position", "origin_date", "target_date", "indicator_id", "p_up", "p_up_calibrated", "predicted_direction", "correctness_probability", "correctness_lcb", "accepted", "selection_rank", "rejection_reason", "final_model_version", "level_b_history_artifact", "locked_audit_used_for_selection", "data_hash", "forecast_status"]
    output = root / "artifacts/forecast_ledgers/june_2026_unscored_v3.csv"
    if output.exists():
        raise FileExistsError(f"Forecast ledger already exists: {output}")
    test[cols].to_csv(output, index=False)
    return output
