from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .chronos_distribution import analyze_step_distribution
from .chronos_protocol import (
    CHRONOS_MODEL_ID,
    CHRONOS_MODEL_REVISION,
    CHRONOS_PACKAGE_PIN,
    DEFAULT_QUANTILES,
    ChronosForecastResult,
    ChronosProvider,
)
from .io import atomic_write_json, atomic_write_parquet, load_workbook, sha256_file

DEFAULT_CHRONOS_CONFIG_PATH = Path("configs/chronos_zero_shot.yaml")
CACHE_VERSION = "chronos_zero_shot_cache_v1"
FEATURE_VERSION = "chronos_zero_shot_v1"

OUTCOME_FORBIDDEN_COLUMNS = {
    "y_true",
    "value_t",
    "value_t1",
    "change_t_to_t1",
    "zero_change",
    "target_available",
    "target_date",
}


def load_chronos_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and strictly validate the Chronos-2 zero-shot configuration."""
    target = Path(path) if path is not None else DEFAULT_CHRONOS_CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"Chronos config file not found: {target}")
    with target.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_chronos_config(config)
    return config


def validate_chronos_config(config: dict[str, Any]) -> None:
    """Strictly validate Chronos configuration fields against research constraints."""
    if not config.get("research_only", False):
        raise ValueError("Chronos config must have research_only=True")
    if config.get("allow_promotion", False):
        raise ValueError("Chronos config must have allow_promotion=False")
    if config.get("locked_evaluation_read_allowed", True):
        raise ValueError("Chronos config must have locked_evaluation_read_allowed=False")

    pkg = config.get("package_pin") or config.get("package")
    if pkg != "chronos-forecasting==2.3.2":
        raise ValueError(f"Expected package pin chronos-forecasting==2.3.2, got {pkg}")

    model = config.get("model_id") or config.get("model")
    if model != "amazon/chronos-2":
        raise ValueError(f"Expected model amazon/chronos-2, got {model}")

    rev = config.get("model_revision")
    if rev != "29ec3766d36d6f73f0696f85560a422f50e8498c":
        raise ValueError(
            f"Expected model revision 29ec3766d36d6f73f0696f85560a422f50e8498c, got {rev}"
        )

    quantiles = config.get("quantiles", [])
    expected_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if [round(float(q), 4) for q in quantiles] != expected_quantiles:
        raise ValueError(f"Expected quantiles {expected_quantiles}, got {quantiles}")

    if config.get("prediction_length") != 2:
        raise ValueError(f"prediction_length must be 2, got {config.get('prediction_length')}")

    lag = config.get("availability_lag", config.get("availability_lag_months"))
    if lag != 1:
        raise ValueError(f"availability_lag must be 1, got {lag}")

    windows = config.get("evaluation_windows", {})
    tuning = config.get("tuning_origins", windows.get("tuning"))
    val = config.get("validation_origins", windows.get("validation"))
    conf = config.get("confirmation_origins", windows.get("confirmation"))
    locked = config.get("locked_origins", windows.get("locked"))

    if list(tuning or []) != [120, 179]:
        raise ValueError(f"Expected tuning [120, 179], got {tuning}")
    if list(val or []) != [180, 219]:
        raise ValueError(f"Expected validation [180, 219], got {val}")
    if list(conf or []) != [220, 266]:
        raise ValueError(f"Expected confirmation [220, 266], got {conf}")
    if list(locked or []) != [268, 315]:
        raise ValueError(f"Expected locked [268, 315], got {locked}")

    max_pos = config.get("maximum_workbook_position")
    if max_pos is None:
        data_sec = config.get("data", {})
        max_pos = data_sec.get("maximum_workbook_position")
    if max_pos != 267:
        raise ValueError(f"Expected maximum_workbook_position 267, got {max_pos}")

    if config.get("seed") != 20260727:
        raise ValueError(f"Expected seed 20260727, got {config.get('seed')}")

    blocks = config.get("bootstrap_blocks", config.get("bootstrap_block_size"))
    reps = config.get("bootstrap_replicates")
    if blocks != 6:
        raise ValueError(f"Expected bootstrap blocks 6, got {blocks}")
    if reps != 500:
        raise ValueError(f"Expected bootstrap replicates 500, got {reps}")


def validate_origin(origin: int) -> int:
    """Validate requested forecast origin. Rejects any < 120, 267, and >= 268."""
    o = int(origin)
    if o < 120 or o > 266:
        raise ValueError(
            f"Origin {o} is invalid. Requested origin must be in [120, 266]. "
            f"Origin 267 is excluded and origins >= 268 are locked."
        )
    return o


def validate_origins(origins: Sequence[int]) -> list[int]:
    """Validate a sequence of forecast origins."""
    cleaned = [validate_origin(o) for o in origins]
    if not cleaned:
        raise ValueError("Origins sequence cannot be empty")
    return sorted(set(cleaned))


def build_chronos_context(
    series: pd.Series | np.ndarray,
    origin_position: int,
    availability_lag: int = 1,
) -> np.ndarray:
    """
    Pure context builder:
    1. Truncates raw values strictly to position <= origin_position - availability_lag (t - 1).
    2. Computes first differences purely inside this truncated context.
    At origin 266 with lag 1, cutoff is position 265.
    """
    validate_origin(origin_position)
    cutoff = origin_position - availability_lag
    arr = np.asarray(series, dtype=float)
    if len(arr) < cutoff:
        raise ValueError(
            f"Series has length {len(arr)} < required cutoff {cutoff} for origin {origin_position}"
        )
    # Raw values only through t - 1
    truncated = arr[:cutoff]
    finite_positions = np.flatnonzero(np.isfinite(truncated))
    if not len(finite_positions):
        raise ValueError("Truncated context contains no finite observations")
    truncated = truncated[finite_positions[0]:]
    if len(truncated) < 2:
        raise ValueError(
            f"Truncated context length {len(truncated)} is too short to compute differences"
        )
    # Differences computed inside truncated context
    diff = np.diff(truncated)
    return diff


def load_chronos_workbook(
    path: str | Path = "data/monthly_indicators.xlsx",
    maximum_position: int = 267,
) -> pd.DataFrame:
    """
    Load monthly indicators workbook with maximum_position=267.
    Asserts that the loaded dataframe position does not exceed 267.
    """
    if maximum_position != 267:
        raise ValueError(f"maximum_position must be exactly 267, got {maximum_position}")
    frame = load_workbook(path, maximum_position=267)
    max_pos = int(frame["position"].max())
    if max_pos > 267:
        raise AssertionError(f"Loaded workbook position {max_pos} exceeds maximum_position 267")
    return frame


@dataclass(frozen=True)
class ChronosCacheKey:
    """Provenance and cache key for Chronos-2 zero-shot forecasts."""

    source_data_hash: str
    config_hash: str
    package_pin: str = CHRONOS_PACKAGE_PIN
    model_id: str = CHRONOS_MODEL_ID
    model_revision: str = CHRONOS_MODEL_REVISION
    cache_version: str = CACHE_VERSION
    seed: int = 20260727
    origin_start: int = 120
    origin_end: int = 266
    prediction_length: int = 2
    availability_lag: int = 1
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES

    def __post_init__(self) -> None:
        validate_origin(self.origin_start)
        validate_origin(self.origin_end)
        if self.origin_start > self.origin_end:
            raise ValueError("origin_start must not exceed origin_end")
        if self.prediction_length != 2 or self.availability_lag != 1:
            raise ValueError("Chronos cache requires prediction_length=2 and availability_lag=1")

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(
            self.payload(),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    if frame.duplicated(["origin_position", "indicator_id"]).any():
        raise ValueError("Frame has duplicate origin_position and indicator_id keys")
    return pd.MultiIndex.from_frame(
        frame[["origin_position", "indicator_id"]].sort_values(
            ["origin_position", "indicator_id"]
        )
    )


def _assert_matching_keys(inputs: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    if not _key_index(inputs).equals(_key_index(outcomes)):
        raise ValueError("Chronos inputs and outcomes keys do not match exactly")


def attach_chronos_outcomes(
    inputs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach evaluation outcomes to outcome-free inputs via exact unique keys."""
    for col in OUTCOME_FORBIDDEN_COLUMNS:
        if col in inputs.columns:
            raise ValueError(f"Outcome column {col!r} found in inputs; inputs must be outcome-free")
    _assert_matching_keys(inputs, outcomes)
    keys = ["origin_position", "indicator_id"]
    outcome_columns = keys + [
        column
        for column in outcomes.columns
        if column not in inputs.columns and column not in keys
    ]
    return inputs.merge(
        outcomes[outcome_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )


def generate_chronos_origin_forecasts(
    frame: pd.DataFrame,
    origin_position: int,
    provider: ChronosProvider,
    *,
    data_hash: str,
    config_hash: str,
    seed: int = 20260727,
    availability_lag: int = 1,
    prediction_length: int = 2,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
    model_id: str = CHRONOS_MODEL_ID,
    model_revision: str = CHRONOS_MODEL_REVISION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate forecast inputs and evaluation outcomes for one origin across all indicators X1..Xn.
    Pure context uses raw values only through t-1.
    Evaluation label (row t+1) is stored only in outcomes.
    """
    validate_origin(origin_position)
    indicators = [c for c in frame.columns if c.startswith("X")]
    levels = tuple(float(q) for q in quantile_levels)

    origin_row = frame.loc[frame["position"] == origin_position]
    origin_date = origin_row["Dates"].iloc[0] if not origin_row.empty else pd.NaT

    target_pos = origin_position + 1
    target_row = frame.loc[frame["position"] == target_pos]
    target_date = target_row["Dates"].iloc[0] if not target_row.empty else pd.NaT

    inputs_rows: list[dict[str, Any]] = []
    outcomes_rows: list[dict[str, Any]] = []

    for ind in indicators:
        series = frame[ind].to_numpy(dtype=float)
        try:
            diff_context = build_chronos_context(
                series,
                origin_position=origin_position,
                availability_lag=availability_lag,
            )
            context_len = len(diff_context)
            data_quality_ok = bool(np.all(np.isfinite(diff_context)))
        except Exception:
            diff_context = np.array([])
            context_len = 0
            data_quality_ok = False

        if not data_quality_ok or context_len < 2:
            forecast_res = ChronosForecastResult(
                quantiles=np.full((prediction_length, len(levels)), np.nan),
                quantile_levels=levels,
                error_flag=True,
                error_message="Insufficient or invalid causal context",
                model_id=model_id,
                model_revision=model_revision,
                seed=seed,
            )
        else:
            try:
                forecast_res = provider.predict(
                    diff_context,
                    prediction_length=prediction_length,
                    quantile_levels=levels,
                    seed=seed,
                )
            except Exception as exc:
                forecast_res = ChronosForecastResult(
                    quantiles=np.full((prediction_length, len(levels)), np.nan),
                    quantile_levels=levels,
                    error_flag=True,
                    error_message=f"Provider failure: {exc}",
                    model_id=model_id,
                    model_revision=model_revision,
                    seed=seed,
                )

        if not forecast_res.error_flag and forecast_res.quantiles is not None:
            try:
                step2_quantiles = forecast_res.step_quantiles(step_index=1)
                dist_summary = analyze_step_distribution(step2_quantiles, levels)
            except Exception:
                dist_summary = analyze_step_distribution(np.empty(0), levels)
        else:
            dist_summary = analyze_step_distribution(np.empty(0), levels)

        input_row = {
            "origin_position": origin_position,
            "origin_date": origin_date,
            "indicator_id": ind,
            "p_up_raw": dist_summary.p_up,
            "p_up": dist_summary.p_up,
            "predicted_direction": dist_summary.predicted_direction,
            "median": dist_summary.median,
            "width_80": dist_summary.width_80,
            "width_50": dist_summary.width_50,
            "normalized_asymmetry": dist_summary.normalized_asymmetry,
            "dispersion_proxy": dist_summary.dispersion_proxy,
            "crossing_flag": dist_summary.crossing_flag,
            "nan_flag": dist_summary.nan_flag,
            "degenerate_flag": dist_summary.degenerate_flag,
            "failure_flag": dist_summary.failure_flag or forecast_res.error_flag,
            "step_index": 1,
            "horizon": prediction_length,
            "context_length": context_len,
            "runtime_seconds": forecast_res.runtime_seconds,
            "error_flag": forecast_res.error_flag or dist_summary.failure_flag,
            "error_message": forecast_res.error_message or dist_summary.error_message,
            "model_id": model_id,
            "model_revision": model_revision,
            "seed": seed,
            "feature_version": FEATURE_VERSION,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "eligible": bool(
                data_quality_ok
                and not forecast_res.error_flag
                and not dist_summary.failure_flag
            ),
            "data_quality_ok": data_quality_ok,
            "locked_evaluation_read": False,
        }
        inputs_rows.append(input_row)

        val_t = float(origin_row[ind].iloc[0]) if not origin_row.empty else np.nan
        val_t1 = float(target_row[ind].iloc[0]) if not target_row.empty else np.nan
        change = val_t1 - val_t if np.isfinite(val_t) and np.isfinite(val_t1) else np.nan
        y_true = float(change > 0) if np.isfinite(change) else np.nan
        zero_change = bool(change == 0.0) if np.isfinite(change) else False

        outcome_row = {
            "origin_position": origin_position,
            "indicator_id": ind,
            "origin_date": origin_date,
            "target_date": target_date,
            "value_t": val_t,
            "value_t1": val_t1,
            "change_t_to_t1": change,
            "zero_change": zero_change,
            "y_true": y_true,
        }
        outcomes_rows.append(outcome_row)

    return inputs_rows, outcomes_rows


def chronos_cache_dir(root: Path, key: ChronosCacheKey) -> Path:
    return root / "research/chronos_zero_shot/cache" / key.digest()


def chronos_cache_generation(root: Path, key: ChronosCacheKey) -> str:
    pointer_path = chronos_cache_dir(root, key) / "current.json"
    if not pointer_path.exists():
        raise FileNotFoundError(f"Generation pointer current.json missing at {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    gen = str(pointer.get("generation", ""))
    if not gen or Path(gen).name != gen:
        raise ValueError(f"Invalid generation pointer: {gen}")
    return gen


def write_chronos_cache(
    root: Path,
    key: ChronosCacheKey,
    inputs: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> Path:
    """Atomically write inputs.parquet, outcomes.parquet, and manifest.json to cache."""
    # Ensure no outcome leakage in inputs
    for col in OUTCOME_FORBIDDEN_COLUMNS:
        if col in inputs.columns:
            raise ValueError(f"Outcome column {col!r} found in inputs; inputs must be outcome-free")

    if inputs.empty or outcomes.empty:
        raise ValueError("Chronos cache frames must not be empty")
    if int(inputs["origin_position"].max()) > 266:
        raise ValueError("Origins above 266 must never enter Chronos cache")
    if int(outcomes["origin_position"].max()) > 266:
        raise ValueError("Origins above 266 must never enter Chronos cache")

    # Enforce origin bounds matching key
    if int(inputs["origin_position"].min()) != key.origin_start:
        raise ValueError(
            f"Input min origin {inputs['origin_position'].min()} does not match key origin_start {key.origin_start}"
        )
    if int(inputs["origin_position"].max()) != key.origin_end:
        raise ValueError(
            f"Input max origin {inputs['origin_position'].max()} does not match key origin_end {key.origin_end}"
        )

    _assert_matching_keys(inputs, outcomes)

    target_dir = chronos_cache_dir(root, key)
    generations_dir = target_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation_id = uuid.uuid4().hex

    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations_dir))
    published = generations_dir / generation_id

    input_path = staging / "inputs.parquet"
    outcome_path = staging / "outcomes.parquet"
    manifest_path = staging / "manifest.json"

    atomic_write_parquet(inputs, input_path)
    atomic_write_parquet(outcomes, outcome_path)

    manifest = {
        "cache_key": key.payload(),
        "cache_key_digest": key.digest(),
        "generation": generation_id,
        "input_columns": inputs.columns.tolist(),
        "input_hash": sha256_file(input_path),
        "input_rows": int(len(inputs)),
        "locked_evaluation_read": False,
        "outcome_columns": outcomes.columns.tolist(),
        "outcome_hash": sha256_file(outcome_path),
        "outcome_rows": int(len(outcomes)),
        "package_pin": key.package_pin,
        "model_id": key.model_id,
        "model_revision": key.model_revision,
        "cache_version": key.cache_version,
        "seed": key.seed,
    }
    atomic_write_json(manifest, manifest_path)
    os.replace(staging, published)
    atomic_write_json({"generation": generation_id}, target_dir / "current.json")
    return published


def load_chronos_cache(
    root: Path,
    key: ChronosCacheKey,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and strictly verify inputs and outcomes from Chronos cache."""
    target_dir = chronos_cache_dir(root, key)
    gen = chronos_cache_generation(root, key)
    published = target_dir / "generations" / gen

    manifest_path = published / "manifest.json"
    input_path = published / "inputs.parquet"
    outcome_path = published / "outcomes.parquet"

    for p in (manifest_path, input_path, outcome_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required Chronos cache file: {p}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized_key_payload = json.loads(
        json.dumps(key.payload(), sort_keys=True, default=str)
    )
    if manifest.get("cache_key") != normalized_key_payload:
        raise ValueError("Chronos cache manifest key does not match key payload")
    if manifest.get("cache_key_digest") != key.digest():
        raise ValueError("Chronos cache manifest digest does not match key digest")
    if manifest.get("generation") != gen:
        raise ValueError("Chronos cache generation does not match manifest")
    if manifest.get("locked_evaluation_read") is not False:
        raise ValueError("Chronos cache manifest reports locked_evaluation_read is not False")
    if manifest.get("input_hash") != sha256_file(input_path):
        raise ValueError("inputs.parquet hash mismatch against manifest")
    if manifest.get("outcome_hash") != sha256_file(outcome_path):
        raise ValueError("outcomes.parquet hash mismatch against manifest")

    inputs = pd.read_parquet(input_path)
    outcomes = pd.read_parquet(outcome_path)

    for col in OUTCOME_FORBIDDEN_COLUMNS:
        if col in inputs.columns:
            raise ValueError(f"Outcome column {col!r} found in cached inputs")

    if inputs.empty or outcomes.empty:
        raise ValueError("Chronos cache frames must not be empty")
    if int(inputs["origin_position"].max()) > 266:
        raise ValueError("Origins above 266 found in cached inputs")
    if int(outcomes["origin_position"].max()) > 266:
        raise ValueError("Origins above 266 found in cached outcomes")

    _assert_matching_keys(inputs, outcomes)

    if len(inputs) != int(manifest["input_rows"]):
        raise ValueError("Input row count mismatch against manifest")
    if len(outcomes) != int(manifest["outcome_rows"]):
        raise ValueError("Outcome row count mismatch against manifest")

    return inputs, outcomes


def build_chronos_cache(
    root: Path,
    config: dict[str, Any],
    provider: ChronosProvider,
    data_path: str | Path | None = None,
    origins: Sequence[int] | None = None,
) -> Path:
    """Build Chronos-2 zero-shot cache calling load_workbook with maximum_position=267."""
    validate_chronos_config(config)
    d_path = Path(
        data_path
        if data_path is not None
        else config.get("data_path", "data/monthly_indicators.xlsx")
    )
    if not d_path.is_absolute():
        d_path = root / d_path

    # Must call load_workbook(... maximum_position=267) and assert <= 267
    frame = load_chronos_workbook(d_path, maximum_position=267)

    if origins is None:
        origins = list(range(120, 267))
    validated_origins = validate_origins(origins)

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    data_hash = sha256_file(d_path)

    all_inputs: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []

    for orig in validated_origins:
        in_rows, out_rows = generate_chronos_origin_forecasts(
            frame=frame,
            origin_position=orig,
            provider=provider,
            data_hash=data_hash,
            config_hash=config_hash,
            seed=int(config["seed"]),
            availability_lag=int(config.get("availability_lag", 1)),
            prediction_length=int(config.get("prediction_length", 2)),
            quantile_levels=tuple(float(q) for q in config.get("quantiles", DEFAULT_QUANTILES)),
            model_id=str(config.get("model_id", CHRONOS_MODEL_ID)),
            model_revision=str(config.get("model_revision", CHRONOS_MODEL_REVISION)),
        )
        all_inputs.extend(in_rows)
        all_outcomes.extend(out_rows)

    inputs_df = pd.DataFrame(all_inputs)
    outcomes_df = pd.DataFrame(all_outcomes)

    key = ChronosCacheKey(
        source_data_hash=data_hash,
        config_hash=config_hash,
        package_pin=str(config.get("package_pin", CHRONOS_PACKAGE_PIN)),
        model_id=str(config.get("model_id", CHRONOS_MODEL_ID)),
        model_revision=str(config.get("model_revision", CHRONOS_MODEL_REVISION)),
        cache_version=CACHE_VERSION,
        seed=int(config["seed"]),
        origin_start=validated_origins[0],
        origin_end=validated_origins[-1],
        prediction_length=int(config.get("prediction_length", 2)),
        availability_lag=int(config.get("availability_lag", 1)),
        quantiles=tuple(float(q) for q in config.get("quantiles", DEFAULT_QUANTILES)),
    )

    return write_chronos_cache(root, key, inputs_df, outcomes_df)
