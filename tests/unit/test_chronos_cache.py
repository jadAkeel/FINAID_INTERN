import json
import numpy as np
import pandas as pd
import pytest

from forecast_select.chronos_cache import (
    CHRONOS_MODEL_ID,
    CHRONOS_MODEL_REVISION,
    CHRONOS_PACKAGE_PIN,
    ChronosCacheKey,
    OUTCOME_FORBIDDEN_COLUMNS,
    attach_chronos_outcomes,
    build_chronos_context,
    generate_chronos_origin_forecasts,
    load_chronos_cache,
    load_chronos_config,
    load_chronos_workbook,
    validate_chronos_config,
    validate_origin,
    write_chronos_cache,
)
from forecast_select.chronos_protocol import (
    DEFAULT_QUANTILES,
    ChronosForecastResult,
    DeterministicChronosProvider,
)
from forecast_select.io import sha256_file


def _dummy_workbook(num_rows: int = 267) -> pd.DataFrame:
    dates = pd.date_range("2004-01-31", periods=num_rows, freq="ME")
    return pd.DataFrame({
        "Dates": dates,
        "position": list(range(1, num_rows + 1)),
        "period": [str(d.to_period("M")) for d in dates],
        "X1": [float(i) * 1.5 + 10.0 for i in range(num_rows)],
        "X2": [float(i) * -0.5 + 50.0 for i in range(num_rows)],
    })


def test_chronos_config_loader_and_strict_validation():
    config = load_chronos_config()
    assert config["research_only"] is True
    assert config["allow_promotion"] is False
    assert config["locked_evaluation_read_allowed"] is False
    assert config["package_pin"] == "chronos-forecasting==2.3.2"
    assert config["model_id"] == "amazon/chronos-2"
    assert config["model_revision"] == "29ec3766d36d6f73f0696f85560a422f50e8498c"
    assert config["prediction_length"] == 2
    assert config["availability_lag"] == 1
    assert config["maximum_workbook_position"] == 267
    assert config["seed"] == 20260727
    assert config["bootstrap_blocks"] == 6
    assert config["bootstrap_replicates"] == 500

    # Test rejections on invalid configurations
    bad_config = dict(config)
    bad_config["maximum_workbook_position"] = 268
    with pytest.raises(ValueError, match="Expected maximum_workbook_position 267"):
        validate_chronos_config(bad_config)

    bad_config = dict(config)
    bad_config["research_only"] = False
    with pytest.raises(ValueError, match="research_only=True"):
        validate_chronos_config(bad_config)

    bad_config = dict(config)
    bad_config["model_id"] = "other-model"
    with pytest.raises(ValueError, match="Expected model amazon/chronos-2"):
        validate_chronos_config(bad_config)


def test_pure_context_builder_and_future_mutation_invariance():
    frame = _dummy_workbook(267)
    series = frame["X1"].to_numpy(dtype=float)

    # Context at origin 266 with lag 1 must end strictly at 265
    ctx_266 = build_chronos_context(series, origin_position=266, availability_lag=1)
    # Truncated raw length is 265, so first difference length is 265 - 1 = 264
    assert len(ctx_266) == 264
    expected_diff = np.diff(series[:265])
    np.testing.assert_allclose(ctx_266, expected_diff)

    # Future mutation invariance: mutate row 266 and 267 (indices 265 and 266)
    mutated_series = series.copy()
    mutated_series[265:] *= 999.0
    ctx_mutated = build_chronos_context(mutated_series, origin_position=266, availability_lag=1)
    np.testing.assert_allclose(ctx_266, ctx_mutated)


def test_spy_load_workbook_maximum_position_exactly_267(monkeypatch):
    calls = []

    def spy_load_workbook(path, maximum_position=None):
        calls.append(maximum_position)
        return _dummy_workbook(267)

    import forecast_select.chronos_cache as cache_module
    monkeypatch.setattr(cache_module, "load_workbook", spy_load_workbook)

    loaded = load_chronos_workbook("fake_path.xlsx", maximum_position=267)
    assert calls == [267]
    assert int(loaded["position"].max()) == 267

    # If position exceeds 267, assertion error must be raised
    def spy_leaking_workbook(path, maximum_position=None):
        return _dummy_workbook(268)

    monkeypatch.setattr(cache_module, "load_workbook", spy_leaking_workbook)
    with pytest.raises(AssertionError, match="exceeds maximum_position 267"):
        load_chronos_workbook("fake_path.xlsx", maximum_position=267)


def test_reject_requested_origins_outside_120_to_266():
    # Valid origin bounds
    assert validate_origin(120) == 120
    assert validate_origin(266) == 266

    # Origin 267 rejected
    with pytest.raises(ValueError, match="Origin 267 is invalid"):
        validate_origin(267)

    # Origin 268 rejected
    with pytest.raises(ValueError, match="Origin 268 is invalid"):
        validate_origin(268)

    # Locked origins >= 268 rejected
    with pytest.raises(ValueError, match="Origin 300 is invalid"):
        validate_origin(300)

    # Below 120 rejected
    with pytest.raises(ValueError, match="Origin 119 is invalid"):
        validate_origin(119)


def test_outcome_free_inputs_and_exact_key_matching():
    frame = _dummy_workbook(267)
    provider = DeterministicChronosProvider(drift=0.1)

    in_rows, out_rows = generate_chronos_origin_forecasts(
        frame=frame,
        origin_position=266,
        provider=provider,
        data_hash="dummy_data_hash",
        config_hash="dummy_config_hash",
    )

    inputs_df = pd.DataFrame(in_rows)
    outcomes_df = pd.DataFrame(out_rows)

    # Verify no outcome columns leak into inputs
    for col in OUTCOME_FORBIDDEN_COLUMNS:
        assert col not in inputs_df.columns, f"Forbidden outcome column {col} in inputs"

    # Verify outcomes have y_true
    assert "y_true" in outcomes_df.columns
    assert "value_t" in outcomes_df.columns
    assert "value_t1" in outcomes_df.columns

    # Verify merge on exact unique keys works
    attached = attach_chronos_outcomes(inputs_df, outcomes_df)
    assert len(attached) == len(inputs_df)
    assert "y_true" in attached.columns
    assert "origin_date" in attached.columns
    assert "origin_date_x" not in attached.columns
    assert "origin_date_y" not in attached.columns

    origin_266 = outcomes_df.loc[
        outcomes_df["origin_position"].eq(266) & outcomes_df["indicator_id"].eq("X1")
    ].iloc[0]
    assert origin_266["value_t"] == frame.loc[frame["position"].eq(266), "X1"].iloc[0]
    assert origin_266["value_t1"] == frame.loc[frame["position"].eq(267), "X1"].iloc[0]

    # If inputs contain outcome column, attach_chronos_outcomes must fail
    leaked_inputs = inputs_df.copy()
    leaked_inputs["y_true"] = 1.0
    with pytest.raises(ValueError, match="must be outcome-free"):
        attach_chronos_outcomes(leaked_inputs, outcomes_df)


def test_atomic_cache_round_trip_and_manifest_hashes(tmp_path):
    frame = _dummy_workbook(267)
    provider = DeterministicChronosProvider(drift=0.1)

    in_rows, out_rows = generate_chronos_origin_forecasts(
        frame=frame,
        origin_position=120,
        provider=provider,
        data_hash="data123",
        config_hash="conf123",
    )

    inputs_df = pd.DataFrame(in_rows)
    outcomes_df = pd.DataFrame(out_rows)

    key = ChronosCacheKey(
        source_data_hash="data123",
        config_hash="conf123",
        origin_start=120,
        origin_end=120,
    )

    published = write_chronos_cache(tmp_path, key, inputs_df, outcomes_df)
    manifest_path = published / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["locked_evaluation_read"] is False
    assert manifest["input_hash"] == sha256_file(published / "inputs.parquet")
    assert manifest["outcome_hash"] == sha256_file(published / "outcomes.parquet")
    assert manifest["package_pin"] == CHRONOS_PACKAGE_PIN
    assert manifest["model_id"] == CHRONOS_MODEL_ID
    assert manifest["model_revision"] == CHRONOS_MODEL_REVISION

    # Successful load
    loaded_in, loaded_out = load_chronos_cache(tmp_path, key)
    pd.testing.assert_frame_equal(loaded_in, inputs_df)
    pd.testing.assert_frame_equal(loaded_out, outcomes_df)

    # Tampering test: corrupt input file
    (published / "inputs.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="inputs.parquet hash mismatch"):
        load_chronos_cache(tmp_path, key)


def test_cache_boundary_rejects_origin_267():
    with pytest.raises(ValueError, match="Origin 267 is invalid"):
        ChronosCacheKey(
            source_data_hash="data123",
            config_hash="conf123",
            origin_start=267,
            origin_end=267,
        )


def test_provider_exception_and_leading_missing_history_fail_safely():
    frame = _dummy_workbook(267)
    frame.loc[:9, "X1"] = np.nan

    class RaisingProvider:
        def predict(self, context, **kwargs):
            raise RuntimeError("boom")

    in_rows, _ = generate_chronos_origin_forecasts(
        frame=frame,
        origin_position=120,
        provider=RaisingProvider(),
        data_hash="data123",
        config_hash="conf123",
    )
    x1 = next(row for row in in_rows if row["indicator_id"] == "X1")
    assert x1["error_flag"] is True
    assert x1["failure_flag"] is True
    assert x1["eligible"] is False
    assert x1["p_up"] == 0.5


def test_generate_forecasts_batches_valid_contexts_once():
    frame = _dummy_workbook(267)

    class BatchProvider:
        def __init__(self):
            self.calls = 0
            self.context_count = 0

        def predict_batch(self, contexts, **kwargs):
            self.calls += 1
            self.context_count = len(contexts)
            return [
                ChronosForecastResult(
                    quantiles=np.tile(np.linspace(-1.0, 1.0, 9), (2, 1)),
                    quantile_levels=DEFAULT_QUANTILES,
                    mean=np.zeros(2),
                )
                for _ in contexts
            ]

        def predict(self, context, **kwargs):
            raise AssertionError("Per-series path must not run with batch support")

    provider = BatchProvider()
    inputs, outcomes = generate_chronos_origin_forecasts(
        frame,
        origin_position=120,
        provider=provider,
        data_hash="data",
        config_hash="config",
    )
    eligible_contexts = sum(
        row["data_quality_ok"] and row["context_length"] >= 2 for row in inputs
    )
    assert provider.calls == 1
    assert provider.context_count == eligible_contexts
    assert len(inputs) == len(outcomes)
