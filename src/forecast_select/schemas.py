from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class OOFContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    model_id: str
    model_version: str
    origin_position: int
    origin_date: datetime
    indicator_id: str
    p_up: float | None
    eligible: bool
    data_quality_ok: bool
    feature_version: str
    data_hash: str
    config_hash: str
    seed: int
    error_flag: bool


REQUIRED_OOF_COLUMNS = {
    "run_id", "model_id", "model_version", "origin_position", "origin_date", "target_date", "indicator_id",
    "y_true", "p_up_raw", "predicted_direction", "eligible", "ineligibility_reason", "data_quality_ok",
    "fit_window", "feature_version", "data_hash", "config_hash", "seed", "runtime_seconds", "error_flag", "error_message",
}


def validate_oof_columns(columns: list[str]) -> None:
    missing = sorted(REQUIRED_OOF_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"OOF contract missing columns: {missing}")
