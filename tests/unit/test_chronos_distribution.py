import numpy as np
import pytest

from forecast_select.chronos_distribution import (
    DEFAULT_QUANTILES,
    analyze_step_distribution,
    repair_quantile_crossing,
    validate_quantile_levels,
)


def test_validate_quantile_levels():
    # Valid levels
    assert validate_quantile_levels((0.1, 0.5, 0.9)) == (0.1, 0.5, 0.9)
    assert validate_quantile_levels(DEFAULT_QUANTILES) == DEFAULT_QUANTILES

    # Empty
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_quantile_levels([])

    # Out of bounds
    with pytest.raises(ValueError, match="not strictly in"):
        validate_quantile_levels([0.0, 0.5, 0.9])
    with pytest.raises(ValueError, match="not strictly in"):
        validate_quantile_levels([0.1, 0.5, 1.0])

    # Non-monotonic
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_quantile_levels([0.5, 0.2, 0.8])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_quantile_levels([0.1, 0.5, 0.5, 0.9])


def test_repair_quantile_crossing_deterministically():
    # Already ordered
    ordered = np.array([1.0, 2.0, 3.0, 4.0])
    repaired, flag = repair_quantile_crossing(ordered)
    assert flag is False
    np.testing.assert_allclose(repaired, ordered)

    # Inverted / crossing quantiles
    inverted = np.array([4.0, 2.5, 3.0, 1.0])
    repaired_inv, flag_inv = repair_quantile_crossing(inverted)
    assert flag_inv is True
    assert list(repaired_inv) == [1.0, 2.5, 3.0, 4.0]
    assert np.all(np.diff(repaired_inv) >= 0)


def test_cdf_piecewise_linear_interpolation_and_clipping():
    levels = (0.1, 0.3, 0.5, 0.7, 0.9)

    # 1. Zero at median -> F(0) = 0.5, P(Up) = 0.5
    v_centered = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    res_centered = analyze_step_distribution(v_centered, levels)
    assert pytest.approx(res_centered.p_up, abs=1e-4) == 0.5
    assert res_centered.median == 0.0
    assert res_centered.width_80 == 4.0
    assert res_centered.dispersion_proxy == 4.0
    assert pytest.approx(res_centered.normalized_asymmetry, abs=1e-4) == 0.0

    # 2. Linear interpolation between quantiles
    # 0 lies between -1.0 (q=0.3) and +1.0 (q=0.5) -> alpha = 0.5 -> F(0) = 0.4 -> P(Up) = 0.6
    v_skew = np.array([-3.0, -1.0, 1.0, 2.0, 3.0])
    res_skew = analyze_step_distribution(v_skew, levels)
    assert pytest.approx(res_skew.p_up, abs=1e-4) == 0.6
    assert res_skew.predicted_direction == "Up"

    # 3. Flat segment at 0
    v_flat = np.array([-2.0, 0.0, 0.0, 1.0, 2.0])
    # F(0) means P(X <= 0), so a zero plateau uses its right edge q=0.5.
    res_flat = analyze_step_distribution(v_flat, levels)
    assert pytest.approx(res_flat.p_up, abs=1e-4) == 0.5


def test_explicit_deterministic_tail_behavior_and_clipping():
    levels = (0.1, 0.2, 0.5, 0.8, 0.9)

    # All quantiles strictly positive (0 < v[0])
    v_high = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    res_high = analyze_step_distribution(v_high, levels)
    # Extrapolation reaches 0 CDF -> P(Up) clipped to 0.999
    assert res_high.p_up == 0.999
    assert res_high.predicted_direction == "Up"

    # All quantiles strictly negative (0 > v[-1])
    v_low = np.array([-50.0, -40.0, -30.0, -20.0, -10.0])
    res_low = analyze_step_distribution(v_low, levels)
    # Extrapolation reaches 1.0 CDF -> P(Up) clipped to 0.001
    assert res_low.p_up == 0.001
    assert res_low.predicted_direction == "Down"


def test_non_gaussian_assumption_preserves_quantile_shape():
    # Asymmetric distribution: long positive tail, tight negative side
    levels = (0.1, 0.5, 0.9)
    v_asym = np.array([-1.0, -0.2, 10.0])
    res_asym = analyze_step_distribution(v_asym, levels)
    # Bowley skewness should be positive
    assert res_asym.normalized_asymmetry is not None
    assert res_asym.normalized_asymmetry > 0.5
    # 0 is between -0.2 (q=0.5) and 10.0 (q=0.9)
    # F(0) = 0.5 + (0 - (-0.2))/(10 - (-0.2)) * (0.9 - 0.5) = 0.5 + 0.2/10.2 * 0.4 approx 0.5078
    # P(Up) approx 1 - 0.5078 = 0.4922
    assert pytest.approx(res_asym.p_up, abs=0.01) == 0.492


def test_50pct_width_only_when_supported():
    # Default quantiles [0.1..0.9] do NOT support 0.25 and 0.75
    v_default = np.linspace(1, 9, 9)
    res_default = analyze_step_distribution(v_default, DEFAULT_QUANTILES)
    assert res_default.width_80 is not None
    assert res_default.width_50 is None

    # Custom quantiles including 0.25 and 0.75
    custom_levels = (0.1, 0.25, 0.5, 0.75, 0.9)
    v_custom = np.array([1.0, 2.5, 5.0, 7.5, 9.0])
    res_custom = analyze_step_distribution(v_custom, custom_levels)
    assert res_custom.width_80 == 8.0  # 9.0 - 1.0
    assert res_custom.width_50 == 5.0  # 7.5 - 2.5


def test_invalid_and_degenerate_forecasts_fail_closed_to_0_5():
    # 1. Degenerate (all quantiles equal)
    v_deg = np.ones(9) * 2.0
    res_deg = analyze_step_distribution(v_deg, DEFAULT_QUANTILES)
    assert res_deg.p_up == 0.5
    assert res_deg.failure_flag is True
    assert res_deg.degenerate_flag is True

    # 2. NaN values
    v_nan = np.linspace(1, 9, 9)
    v_nan[3] = np.nan
    res_nan = analyze_step_distribution(v_nan, DEFAULT_QUANTILES)
    assert res_nan.p_up == 0.5
    assert res_nan.failure_flag is True
    assert res_nan.nan_flag is True

    # 3. Inf values
    v_inf = np.linspace(1, 9, 9)
    v_inf[0] = np.inf
    res_inf = analyze_step_distribution(v_inf, DEFAULT_QUANTILES)
    assert res_inf.p_up == 0.5
    assert res_inf.failure_flag is True
    assert res_inf.nan_flag is True

    # 4. Mismatched length
    res_short = analyze_step_distribution(np.array([1.0, 2.0]), DEFAULT_QUANTILES)
    assert res_short.p_up == 0.5
    assert res_short.failure_flag is True
