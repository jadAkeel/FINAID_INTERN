import numpy as np
import pytest

from forecast_select.chronos_protocol import (
    CHRONOS_MODEL_ID,
    CHRONOS_MODEL_REVISION,
    DEFAULT_QUANTILES,
    ChronosForecastResult,
    ChronosProvider,
    ChronosRealProvider,
    DeterministicChronosProvider,
)


def test_chronos_provider_protocol_conformance():
    deterministic = DeterministicChronosProvider()
    assert isinstance(deterministic, ChronosProvider)

    real = ChronosRealProvider()
    assert isinstance(real, ChronosProvider)


def test_deterministic_provider_produces_consistent_2step_quantiles():
    provider = DeterministicChronosProvider(scale=2.0, drift=0.5)
    context = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    res1 = provider.predict(context, prediction_length=2, seed=20260727)
    res2 = provider.predict(context, prediction_length=2, seed=20260727)

    assert isinstance(res1, ChronosForecastResult)
    assert res1.error_flag is False
    assert res1.error_message is None
    assert res1.quantiles.shape == (2, 9)
    assert res1.quantile_levels == DEFAULT_QUANTILES
    assert res1.mean is not None
    assert res1.mean.shape == (2,)

    # Invariance and determinism
    np.testing.assert_allclose(res1.quantiles, res2.quantiles)
    np.testing.assert_allclose(res1.mean, res2.mean)

    # Step quantiles access
    q_step1 = res1.step_quantiles(step_index=0)
    q_step2 = res1.step_quantiles(step_index=1)
    assert q_step1.shape == (9,)
    assert q_step2.shape == (9,)
    assert np.all(np.diff(q_step2) > 0)  # monotonically increasing


def test_deterministic_provider_handles_fixed_outputs_and_errors():
    fixed_q = np.ones((2, 9)) * 5.0
    fixed_m = np.array([5.0, 5.0])
    provider = DeterministicChronosProvider(fixed_quantiles=fixed_q, fixed_mean=fixed_m)

    res = provider.predict([1.0, 2.0])
    np.testing.assert_allclose(res.quantiles, fixed_q)
    np.testing.assert_allclose(res.mean, fixed_m)

    error_provider = DeterministicChronosProvider(raise_error=RuntimeError("Device failure"))
    err_res = error_provider.predict([1.0, 2.0])
    assert err_res.error_flag is True
    assert "Device failure" in str(err_res.error_message)


def test_real_provider_pins_exact_model_and_revision_with_no_fallback():
    with pytest.raises(ValueError, match="Model ID must be exactly"):
        ChronosRealProvider(model_id="amazon/chronos-bolt-small")

    with pytest.raises(ValueError, match="Model revision must be exactly"):
        ChronosRealProvider(model_revision="main")

    provider = ChronosRealProvider(
        model_id=CHRONOS_MODEL_ID,
        model_revision=CHRONOS_MODEL_REVISION,
    )
    assert provider.model_id == CHRONOS_MODEL_ID
    assert provider.model_revision == CHRONOS_MODEL_REVISION


def test_real_provider_raises_actionable_import_error_on_missing_optional_dep(monkeypatch):
    provider = ChronosRealProvider()

    # Simulate missing chronos package
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("chronos", "chronos.chronos2"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError) as exc_info:
        provider._load_pipeline()

    assert "pip install -e .[pretrained]" in str(exc_info.value)


def test_real_provider_defensive_shape_parser():
    # 1. Standard (prediction_length=2, num_quantiles=9)
    raw_2x9 = np.arange(18, dtype=float).reshape(2, 9)
    q, m = ChronosRealProvider._defensive_shape_parse(raw_2x9, prediction_length=2, num_quantiles=9)
    assert q.shape == (2, 9)
    assert m is None

    # 2. Transposed (num_quantiles=9, prediction_length=2)
    raw_9x2 = raw_2x9.T
    q_t, _ = ChronosRealProvider._defensive_shape_parse(raw_9x2, prediction_length=2, num_quantiles=9)
    assert q_t.shape == (2, 9)
    np.testing.assert_allclose(q, q_t)

    # 3. Batch 3D (1, 2, 9)
    raw_1x2x9 = raw_2x9.reshape(1, 2, 9)
    q_b, _ = ChronosRealProvider._defensive_shape_parse(raw_1x2x9, prediction_length=2, num_quantiles=9)
    assert q_b.shape == (2, 9)
    np.testing.assert_allclose(q, q_b)

    # 4D official Chronos-2 univariate output: task, variate, horizon, quantile.
    raw_1x1x2x9 = raw_2x9.reshape(1, 1, 2, 9)
    q_4d, _ = ChronosRealProvider._defensive_shape_parse(
        raw_1x1x2x9, prediction_length=2, num_quantiles=9
    )
    np.testing.assert_allclose(q, q_4d)

    # 4. Tuple with mean
    mean_2 = np.array([1.5, 2.5])
    q_tup, m_tup = ChronosRealProvider._defensive_shape_parse(
        (raw_2x9, mean_2), prediction_length=2, num_quantiles=9
    )
    assert q_tup.shape == (2, 9)
    assert m_tup is not None
    assert m_tup.shape == (2,)
    np.testing.assert_allclose(m_tup, mean_2)

    q_list, m_list = ChronosRealProvider._defensive_shape_parse(
        [raw_1x1x2x9, mean_2.reshape(1, 1, 2)],
        prediction_length=2,
        num_quantiles=9,
    )
    np.testing.assert_allclose(q_list, q)
    np.testing.assert_allclose(m_list, mean_2)

    # 5. Dict output
    out_dict = {"quantiles": raw_2x9, "mean": mean_2}
    q_d, m_d = ChronosRealProvider._defensive_shape_parse(out_dict, prediction_length=2, num_quantiles=9)
    assert q_d.shape == (2, 9)
    assert m_d.shape == (2,)

    # 6. Invalid shapes raise ValueError
    with pytest.raises(ValueError, match="Unexpected quantiles shape"):
        ChronosRealProvider._defensive_shape_parse(np.ones((4, 4)), prediction_length=2, num_quantiles=9)


def test_real_provider_uses_official_sequence_input_and_list_output(monkeypatch):
    torch = pytest.importorskip("torch")

    class FakePipeline:
        def predict_quantiles(self, inputs, **kwargs):
            assert isinstance(inputs, list)
            assert len(inputs) == 2
            assert all(item.ndim == 1 for item in inputs)
            assert kwargs["prediction_length"] == 2
            quantiles = [torch.arange(18, dtype=torch.float32).reshape(1, 2, 9) for _ in inputs]
            means = [torch.tensor([[1.0, 2.0]], dtype=torch.float32) for _ in inputs]
            return quantiles, means

    provider = ChronosRealProvider()
    monkeypatch.setattr(provider, "_load_pipeline", lambda: FakePipeline())
    results = provider.predict_batch(
        [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])], seed=20260727
    )
    assert len(results) == 2
    assert all(result.quantiles.shape == (2, 9) for result in results)
    assert all(result.mean.shape == (2,) for result in results)


def test_real_provider_rejects_wrong_batch_output_count(monkeypatch):
    torch = pytest.importorskip("torch")

    class BadPipeline:
        def predict_quantiles(self, inputs, **kwargs):
            return [torch.zeros((1, 2, 9))], [torch.zeros((1, 2))]

    provider = ChronosRealProvider()
    monkeypatch.setattr(provider, "_load_pipeline", lambda: BadPipeline())
    with pytest.raises(ValueError, match="Expected 2 Chronos-2 quantile outputs"):
        provider.predict_batch([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
