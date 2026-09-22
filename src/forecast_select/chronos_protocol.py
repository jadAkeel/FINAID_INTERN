from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

CHRONOS_PACKAGE_PIN = "chronos-forecasting==2.3.2"
CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PRETRAINED_INSTALL_MSG = (
    "Chronos-2 requires chronos-forecasting==2.3.2 and torch. "
    "Please install optional dependencies via: pip install -e .[pretrained]"
)


@dataclass(frozen=True)
class ChronosForecastResult:
    """Dataclass forecast result contract for 2-step quantiles/means plus runtime/error metadata."""

    quantiles: np.ndarray  # shape: (prediction_length, num_quantiles)
    quantile_levels: tuple[float, ...]
    mean: np.ndarray | None = None  # shape: (prediction_length,)
    runtime_seconds: float = 0.0
    error_flag: bool = False
    error_message: str | None = None
    model_id: str = CHRONOS_MODEL_ID
    model_revision: str = CHRONOS_MODEL_REVISION
    seed: int | None = None

    def step_quantiles(self, step_index: int = 1) -> np.ndarray:
        """Return 1D array of quantiles for step_index (0-based, default 1 for step 2)."""
        if self.quantiles is None or self.quantiles.ndim < 2:
            raise ValueError("Quantiles array is not a 2D matrix")
        if step_index < 0 or step_index >= self.quantiles.shape[0]:
            raise IndexError(
                f"step_index {step_index} out of bounds for prediction_length {self.quantiles.shape[0]}"
            )
        return self.quantiles[step_index]

    def step_mean(self, step_index: int = 1) -> float | None:
        """Return mean forecast for step_index if available."""
        if self.mean is None:
            return None
        if step_index < 0 or step_index >= len(self.mean):
            raise IndexError(f"step_index {step_index} out of bounds for mean length {len(self.mean)}")
        return float(self.mean[step_index])


@runtime_checkable
class ChronosProvider(Protocol):
    """Runtime-checkable protocol for Chronos-2 forecast providers."""

    def predict(
        self,
        context: np.ndarray | Sequence[float],
        *,
        prediction_length: int = 2,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ChronosForecastResult:
        """Generate probabilistic forecasts for a given context sequence."""
        ...


class DeterministicChronosProvider:
    """Deterministic offline provider for tests with zero torch/chronos/network imports."""

    def __init__(
        self,
        *,
        model_id: str = CHRONOS_MODEL_ID,
        model_revision: str = CHRONOS_MODEL_REVISION,
        scale: float = 1.0,
        drift: float = 0.0,
        fixed_quantiles: np.ndarray | None = None,
        fixed_mean: np.ndarray | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.scale = scale
        self.drift = drift
        self.fixed_quantiles = fixed_quantiles
        self.fixed_mean = fixed_mean
        self.raise_error = raise_error

    def predict(
        self,
        context: np.ndarray | Sequence[float],
        *,
        prediction_length: int = 2,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ChronosForecastResult:
        start_time = time.perf_counter()

        if self.raise_error is not None:
            return ChronosForecastResult(
                quantiles=np.empty((prediction_length, len(quantile_levels)), dtype=float),
                quantile_levels=tuple(float(q) for q in quantile_levels),
                mean=None,
                runtime_seconds=time.perf_counter() - start_time,
                error_flag=True,
                error_message=str(self.raise_error),
                model_id=self.model_id,
                model_revision=self.model_revision,
                seed=seed,
            )

        levels = tuple(float(q) for q in quantile_levels)

        if self.fixed_quantiles is not None:
            q_arr = np.asarray(self.fixed_quantiles, dtype=float)
            m_arr = np.asarray(self.fixed_mean, dtype=float) if self.fixed_mean is not None else None
            return ChronosForecastResult(
                quantiles=q_arr,
                quantile_levels=levels,
                mean=m_arr,
                runtime_seconds=time.perf_counter() - start_time,
                error_flag=False,
                error_message=None,
                model_id=self.model_id,
                model_revision=self.model_revision,
                seed=seed,
            )

        ctx = np.asarray(context, dtype=float)
        base = float(ctx[-1]) if len(ctx) > 0 else 0.0

        quantiles = np.zeros((prediction_length, len(levels)), dtype=float)
        means = np.zeros(prediction_length, dtype=float)

        for step in range(prediction_length):
            step_base = base + self.drift * float(step + 1)
            means[step] = step_base
            for i, q in enumerate(levels):
                quantiles[step, i] = step_base + self.scale * (q - 0.5) * 2.0

        return ChronosForecastResult(
            quantiles=quantiles,
            quantile_levels=levels,
            mean=means,
            runtime_seconds=time.perf_counter() - start_time,
            error_flag=False,
            error_message=None,
            model_id=self.model_id,
            model_revision=self.model_revision,
            seed=seed,
        )


class ChronosRealProvider:
    """
    Real Chronos-2 provider that lazy-imports Chronos2Pipeline from chronos-forecasting==2.3.2.

    Calls from_pretrained with exact model and revision, enforces no fallback,
    and defensively handles official predict_quantiles shapes.
    """

    def __init__(
        self,
        *,
        model_id: str = CHRONOS_MODEL_ID,
        model_revision: str = CHRONOS_MODEL_REVISION,
        device_map: str | None = None,
        torch_dtype: Any = None,
    ) -> None:
        if model_id != CHRONOS_MODEL_ID:
            raise ValueError(
                f"Model ID must be exactly {CHRONOS_MODEL_ID!r} (no fallback), got {model_id!r}"
            )
        if model_revision != CHRONOS_MODEL_REVISION:
            raise ValueError(
                f"Model revision must be exactly {CHRONOS_MODEL_REVISION!r} (no fallback), got {model_revision!r}"
            )
        self.model_id = model_id
        self.model_revision = model_revision
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._pipeline: Any = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            try:
                from chronos import Chronos2Pipeline
            except ImportError:
                from chronos.chronos2 import Chronos2Pipeline
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(PRETRAINED_INSTALL_MSG) from exc

        kwargs: dict[str, Any] = {
            "revision": self.model_revision,
        }
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype

        # Exact model and revision call with no fallback
        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            **kwargs,
        )
        return self._pipeline

    def predict(
        self,
        context: np.ndarray | Sequence[float],
        *,
        prediction_length: int = 2,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ChronosForecastResult:
        start_time = time.perf_counter()
        pipeline = self._load_pipeline()

        # Fix seeds where supported
        if seed is not None:
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            np.random.seed(seed)

        import torch

        ctx = np.asarray(context, dtype=float)
        tensor_context = torch.tensor(ctx, dtype=torch.float32)

        levels = [float(q) for q in quantile_levels]
        predict_kwargs: dict[str, Any] = {
            "prediction_length": prediction_length,
            "quantile_levels": levels,
        }
        if seed is not None:
            predict_kwargs["seed"] = seed

        try:
            raw_output = pipeline.predict_quantiles(tensor_context, **predict_kwargs)
        except TypeError:
            predict_kwargs.pop("seed", None)
            raw_output = pipeline.predict_quantiles(tensor_context, **predict_kwargs)

        quantiles_arr, mean_arr = self._defensive_shape_parse(
            raw_output,
            prediction_length=prediction_length,
            num_quantiles=len(levels),
        )

        return ChronosForecastResult(
            quantiles=quantiles_arr,
            quantile_levels=tuple(levels),
            mean=mean_arr,
            runtime_seconds=time.perf_counter() - start_time,
            error_flag=False,
            error_message=None,
            model_id=self.model_id,
            model_revision=self.model_revision,
            seed=seed,
        )

    @staticmethod
    def _defensive_shape_parse(
        raw: Any,
        *,
        prediction_length: int,
        num_quantiles: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Defensively parse returned prediction_quantiles shapes.

        Supports:
        - tuple: (quantiles, mean) or (quantiles,)
        - dict: {"quantiles": ..., "mean": ...}
        - tensor or ndarray
        - (prediction_length, num_quantiles) or (num_quantiles, prediction_length)
        - (1, prediction_length, num_quantiles)
        """
        raw_quantiles: Any = raw
        raw_mean: Any = None

        if isinstance(raw, (tuple, list)):
            if len(raw) >= 2:
                raw_quantiles, raw_mean = raw[0], raw[1]
            elif len(raw) == 1:
                raw_quantiles = raw[0]
        elif isinstance(raw, dict):
            raw_quantiles = raw.get("quantiles", raw.get("forecast", raw))
            raw_mean = raw.get("mean")

        if hasattr(raw_quantiles, "detach"):
            raw_quantiles = raw_quantiles.detach().cpu().numpy()
        else:
            raw_quantiles = np.asarray(raw_quantiles, dtype=float)

        q = np.asarray(raw_quantiles, dtype=float)

        while q.ndim > 2 and q.shape[0] == 1:
            q = q[0]

        if q.ndim == 2:
            if q.shape == (prediction_length, num_quantiles):
                pass
            elif q.shape == (num_quantiles, prediction_length):
                q = q.T
            else:
                raise ValueError(
                    f"Unexpected quantiles shape {q.shape}; expected ({prediction_length}, {num_quantiles}) "
                    f"or ({num_quantiles}, {prediction_length})"
                )
        else:
            raise ValueError(f"Quantiles must be 2D after squeezing batch dim, got shape {q.shape}")

        m: np.ndarray | None = None
        if raw_mean is not None:
            if hasattr(raw_mean, "detach"):
                raw_mean = raw_mean.detach().cpu().numpy()
            m = np.asarray(raw_mean, dtype=float)
            while m.ndim > 1 and m.shape[0] == 1:
                m = m[0]
            if m.ndim > 1:
                m = m.squeeze()
            if m.shape == (prediction_length,):
                pass
            elif m.size == prediction_length:
                m = m.reshape((prediction_length,))
            else:
                m = None

        return q, m
