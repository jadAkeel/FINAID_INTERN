from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class ChronosDistributionSummary:
    """Summary of step-2 quantile distribution metrics and directional probability."""

    p_up: float
    predicted_direction: str
    median: float | None
    width_80: float | None
    width_50: float | None
    normalized_asymmetry: float | None
    dispersion_proxy: float | None
    crossing_flag: bool
    nan_flag: bool
    degenerate_flag: bool
    failure_flag: bool
    repaired_quantiles: tuple[float, ...] | None
    error_message: str | None = None


def validate_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    """
    Validate that quantile levels are non-empty, strictly increasing, and in (0, 1).
    Raises ValueError if invalid.
    """
    if not levels:
        raise ValueError("Quantile levels cannot be empty")

    levels_tuple = tuple(float(x) for x in levels)
    for i, q in enumerate(levels_tuple):
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantile level {q} at index {i} is not strictly in (0, 1)")
        if i > 0 and q <= levels_tuple[i - 1]:
            raise ValueError(
                f"Quantile levels must be strictly increasing: level[{i}]={q} <= level[{i-1}]={levels_tuple[i-1]}"
            )
    return levels_tuple


def repair_quantile_crossing(quantiles: np.ndarray | Sequence[float]) -> tuple[np.ndarray, bool]:
    """
    Deterministically repair quantile crossing by monotone rearrangement (sorting).
    Returns (repaired_quantiles, crossing_flag).
    """
    arr = np.asarray(quantiles, dtype=float)
    if len(arr) <= 1:
        return arr, False
    crossing = bool(np.any(np.diff(arr) < 0.0))
    if crossing:
        arr = np.sort(arr)
    else:
        arr = arr.copy()
    return arr, crossing


def interpolate_quantile_cdf_at_zero(
    repaired_quantiles: np.ndarray,
    quantile_levels: tuple[float, ...],
) -> float:
    """
    Compute cumulative probability F(0) = P(X <= 0) using piecewise-linear quantile CDF interpolation.
    Uses explicit deterministic linear tail behavior with no Gaussian assumption.
    """
    v = repaired_quantiles
    q = np.asarray(quantile_levels, dtype=float)
    n = len(v)

    # If 0 lies within the range [v[0], v[-1]]
    if v[0] <= 0.0 <= v[-1]:
        idx = int(np.searchsorted(v, 0.0, side="right"))
        if idx == 0:
            return float(q[0])
        if idx >= n:
            return float(q[-1])
        v_low = v[idx - 1]
        v_high = v[idx]
        q_low = q[idx - 1]
        q_high = q[idx]

        if v_high == v_low:
            # Step at 0: mid-quantile
            return float((q_low + q_high) / 2.0)

        slope = (q_high - q_low) / (v_high - v_low)
        cdf_val = q_low + slope * (0.0 - v_low)
        return float(np.clip(cdf_val, q_low, q_high))

    # Lower tail: 0 < v[0] (distribution is strictly to the right of 0)
    if 0.0 < v[0]:
        # Find first non-degenerate segment for edge slope
        j = 1
        while j < n and v[j] == v[0]:
            j += 1
        if j < n:
            slope = (q[j] - q[0]) / (v[j] - v[0])
            extrapolated = q[0] - slope * (v[0] - 0.0)
            return float(np.clip(extrapolated, 0.0, q[0]))
        return 0.0

    # Upper tail: 0 > v[-1] (distribution is strictly to the left of 0)
    if 0.0 > v[-1]:
        # Find last non-degenerate segment for edge slope
        j = n - 2
        while j >= 0 and v[j] == v[-1]:
            j -= 1
        if j >= 0:
            slope = (q[-1] - q[j]) / (v[-1] - v[j])
            extrapolated = q[-1] + slope * (0.0 - v[-1])
            return float(np.clip(extrapolated, q[-1], 1.0))
        return 1.0

    return 0.5


def analyze_step_distribution(
    quantiles: np.ndarray | Sequence[float],
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
) -> ChronosDistributionSummary:
    """
    Analyze step-2 first-difference quantiles:
    - Validates quantile levels and shape
    - Detects and deterministically repairs quantile crossing
    - Computes P(Up) = P(step-2 first difference > 0) = 1 - F(0)
    - Clips P(Up) to [0.001, 0.999] with no Gaussian assumption
    - Extracts median, 80% width, 50% width (only if supported), normalized asymmetry, dispersion proxy
    - Flags crossing, NaN, degenerate, and failures
    - Fails closed to p_up = 0.5 on invalid input
    """
    # Validate quantile levels
    try:
        levels = validate_quantile_levels(quantile_levels)
    except Exception as exc:
        return _fail_closed(
            error_message=f"Invalid quantile levels: {exc}",
            crossing_flag=False,
            nan_flag=False,
            degenerate_flag=False,
        )

    # Validate shape and finite values
    arr = np.asarray(quantiles, dtype=float)
    if arr.ndim != 1 or len(arr) != len(levels):
        return _fail_closed(
            error_message=f"Quantiles shape mismatch: expected 1D of length {len(levels)}, got shape {arr.shape}",
            crossing_flag=False,
            nan_flag=False,
            degenerate_flag=False,
        )

    if not np.all(np.isfinite(arr)):
        return _fail_closed(
            error_message="Quantiles contain non-finite (NaN or Inf) values",
            crossing_flag=False,
            nan_flag=True,
            degenerate_flag=False,
        )

    # Check for degenerate distribution (zero spread / all identical)
    spread = float(np.max(arr) - np.min(arr))
    if spread <= 0.0 or not np.isfinite(spread):
        return _fail_closed(
            error_message="Degenerate distribution: zero spread across all quantiles",
            crossing_flag=False,
            nan_flag=False,
            degenerate_flag=True,
            repaired_quantiles=tuple(float(x) for x in arr),
        )

    # Deterministic crossing repair
    repaired, crossing_flag = repair_quantile_crossing(arr)

    # Compute P(Up) = P(step-2 first difference > 0) = 1 - F(0)
    cdf_at_zero = interpolate_quantile_cdf_at_zero(repaired, levels)
    p_up_raw = 1.0 - cdf_at_zero
    p_up = float(np.clip(p_up_raw, 0.001, 0.999))
    direction = "Up" if p_up >= 0.5 else "Down"

    # Extract median
    if 0.5 in levels:
        idx_50 = levels.index(0.5)
        median_val = float(repaired[idx_50])
    else:
        # Interpolate median
        median_val = float(np.interp(0.5, levels, repaired))

    # Extract 80% width (0.9 quantile - 0.1 quantile)
    if 0.1 in levels and 0.9 in levels:
        idx_10 = levels.index(0.1)
        idx_90 = levels.index(0.9)
        width_80: float | None = float(repaired[idx_90] - repaired[idx_10])
    elif levels[0] <= 0.1 and levels[-1] >= 0.9:
        width_80 = float(np.interp(0.9, levels, repaired) - np.interp(0.1, levels, repaired))
    else:
        width_80 = None

    # Extract 50% width ONLY when supported (0.25 and 0.75 are in quantile levels)
    if 0.25 in levels and 0.75 in levels:
        idx_25 = levels.index(0.25)
        idx_75 = levels.index(0.75)
        width_50: float | None = float(repaired[idx_75] - repaired[idx_25])
    else:
        width_50 = None

    # Extract normalized asymmetry (Bowley skewness: (q90 + q10 - 2*q50) / (q90 - q10))
    if width_80 is not None and width_80 > 1e-12:
        if 0.1 in levels and 0.9 in levels:
            q10 = float(repaired[levels.index(0.1)])
            q90 = float(repaired[levels.index(0.9)])
        else:
            q10 = float(np.interp(0.1, levels, repaired))
            q90 = float(np.interp(0.9, levels, repaired))
        asym = (q90 + q10 - 2.0 * median_val) / width_80
        normalized_asymmetry: float | None = float(np.clip(asym, -1.0, 1.0))
    elif width_80 is not None:
        normalized_asymmetry = 0.0
    else:
        normalized_asymmetry = None

    # Dispersion proxy: interdecile range (width_80)
    dispersion_proxy = width_80

    return ChronosDistributionSummary(
        p_up=p_up,
        predicted_direction=direction,
        median=median_val,
        width_80=width_80,
        width_50=width_50,
        normalized_asymmetry=normalized_asymmetry,
        dispersion_proxy=dispersion_proxy,
        crossing_flag=crossing_flag,
        nan_flag=False,
        degenerate_flag=False,
        failure_flag=False,
        repaired_quantiles=tuple(float(x) for x in repaired),
        error_message=None,
    )


def _fail_closed(
    *,
    error_message: str,
    crossing_flag: bool,
    nan_flag: bool,
    degenerate_flag: bool,
    repaired_quantiles: tuple[float, ...] | None = None,
) -> ChronosDistributionSummary:
    """Return closed failure distribution summary with p_up = 0.5."""
    return ChronosDistributionSummary(
        p_up=0.5,
        predicted_direction="Up",
        median=None,
        width_80=None,
        width_50=None,
        normalized_asymmetry=None,
        dispersion_proxy=None,
        crossing_flag=crossing_flag,
        nan_flag=nan_flag,
        degenerate_flag=degenerate_flag,
        failure_flag=True,
        repaired_quantiles=repaired_quantiles,
        error_message=error_message,
    )
