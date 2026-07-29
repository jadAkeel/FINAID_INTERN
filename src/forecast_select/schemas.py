from __future__ import annotations

REQUIRED_OOF_COLUMNS = {
    "run_id", "model_id", "model_version", "origin_position", "origin_date", "target_date", "indicator_id",
    "y_true", "p_up_raw", "predicted_direction", "eligible", "ineligibility_reason", "data_quality_ok",
    "fit_window", "feature_version", "data_hash", "config_hash", "seed", "runtime_seconds", "error_flag", "error_message",
}


def validate_oof_columns(columns: list[str]) -> None:
    missing = sorted(REQUIRED_OOF_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"OOF contract missing columns: {missing}")
