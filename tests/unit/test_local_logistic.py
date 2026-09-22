import numpy as np
import pandas as pd
import pytest

from forecast_select.local_logistic import (
    LOCAL_FEATURE_SETS,
    fit_interaction_logistic_model,
    fit_local_logistic_models,
    local_coefficients,
    local_feature_columns,
    predict_interaction_probability,
    predict_local_probability,
    sample_aware_weights,
    shrink_probabilities,
)
from forecast_select.local_logistic_pipeline import (
    LocalSpec,
    assert_origins_unlocked,
    effective_local,
)
from forecast_select.uptrend_model import FEATURE_COLUMNS


SETTINGS = {
    "locked_origins": [268, 315],
    "maximum_readable_position": 267,
}


def _panel(
    indicators=("X1", "X2", "X3"),
    months=80,
    seed=7,
    columns=None,
) -> pd.DataFrame:
    """A small synthetic panel with the columns the local models consume."""
    columns = list(columns or LOCAL_FEATURE_SETS["core8"])
    rng = np.random.default_rng(seed)
    rows = []
    for offset, indicator in enumerate(indicators):
        for origin in range(1, months + 1):
            row = {
                "origin_position": origin,
                "indicator_id": indicator,
                # Indicator-specific slope on momentum_3 so a local model can
                # actually learn something different per indicator.
                "y_true": float(0.0),
            }
            for column in columns:
                row[column] = float(rng.normal())
            signal = (1.0 if offset == 0 else -1.0) * row["momentum_3"]
            row["y_true"] = float(signal + rng.normal(scale=0.3) > 0)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature-set contract
# ---------------------------------------------------------------------------


def test_local_feature_sets_are_subsets_of_the_production_panel():
    for name, columns in LOCAL_FEATURE_SETS.items():
        assert len(columns) == len(set(columns)), name
        assert set(columns).issubset(set(FEATURE_COLUMNS)), name
        # The brief caps local predictors at roughly 15.
        assert 5 <= len(columns) <= 15, name


def test_unknown_local_feature_set_is_rejected():
    with pytest.raises(ValueError):
        local_feature_columns("does_not_exist")


# ---------------------------------------------------------------------------
# Local models see same-indicator history only
# ---------------------------------------------------------------------------


def test_local_model_uses_only_same_indicator_history():
    columns = LOCAL_FEATURE_SETS["core8"]
    train = _panel()
    model = fit_local_logistic_models(
        train,
        feature_columns=columns,
        seed=1,
        logistic_c=0.1,
        max_iter=500,
        minimum_local_rows=36,
        minimum_local_class_rows=8,
    )
    baseline = local_coefficients(model).set_index("indicator_id")

    # Perturbing X2 and X3 must leave the X1 model untouched. Both classes are
    # kept so those indicators still get a local model.
    perturbed = train.copy()
    mask = perturbed["indicator_id"].ne("X1")
    perturbed.loc[mask, "momentum_3"] = (
        perturbed.loc[mask, "y_true"].to_numpy(dtype=float) * 3.0
        - 1.5
    )
    changed = fit_local_logistic_models(
        perturbed,
        feature_columns=columns,
        seed=1,
        logistic_c=0.1,
        max_iter=500,
        minimum_local_rows=36,
        minimum_local_class_rows=8,
    )
    after = local_coefficients(changed).set_index("indicator_id")
    for column in [c for c in baseline.columns if c.startswith("coef_")]:
        assert baseline.loc["X1", column] == pytest.approx(after.loc["X1", column])
    assert baseline.loc["X2", "coef_momentum_3"] != pytest.approx(
        after.loc["X2", "coef_momentum_3"]
    )


def test_local_models_recover_opposite_indicator_specific_slopes():
    columns = LOCAL_FEATURE_SETS["core8"]
    model = fit_local_logistic_models(
        _panel(months=200),
        feature_columns=columns,
        seed=1,
        logistic_c=1.0,
        max_iter=500,
        minimum_local_rows=36,
        minimum_local_class_rows=8,
    )
    coefficients = local_coefficients(model).set_index("indicator_id")
    assert coefficients.loc["X1", "coef_momentum_3"] > 0.0
    assert coefficients.loc["X2", "coef_momentum_3"] < 0.0


# ---------------------------------------------------------------------------
# Minimum-sample fallback
# ---------------------------------------------------------------------------


def test_short_history_indicator_falls_back_to_the_global_probability():
    columns = LOCAL_FEATURE_SETS["core8"]
    train = _panel(months=80)
    short = _panel(indicators=("X9",), months=20, seed=11)
    train = pd.concat([train, short], ignore_index=True)
    model = fit_local_logistic_models(
        train,
        feature_columns=columns,
        seed=1,
        logistic_c=0.1,
        max_iter=500,
        minimum_local_rows=36,
        minimum_local_class_rows=8,
    )
    assert "X9" not in model.models
    assert model.training_rows["X9"] == 20

    test = train.groupby("indicator_id", as_index=False).tail(1).reset_index(drop=True)
    fallback = np.full(len(test), 0.42)
    probability, counts, used = predict_local_probability(model, test, fallback)
    short_slot = int(test.index[test["indicator_id"].eq("X9")][0])
    assert used[short_slot]
    assert probability[short_slot] == pytest.approx(0.42)
    assert counts[short_slot] == 20
    for slot in range(len(test)):
        if slot == short_slot:
            continue
        assert not used[slot]


def test_single_class_indicator_falls_back():
    columns = LOCAL_FEATURE_SETS["lean5"]
    train = _panel(indicators=("X1",), months=80)
    train["y_true"] = 1.0
    model = fit_local_logistic_models(
        train,
        feature_columns=columns,
        seed=1,
        logistic_c=0.1,
        max_iter=500,
        minimum_local_rows=36,
        minimum_local_class_rows=8,
    )
    assert model.models == {}


def test_minimum_class_rows_threshold_is_enforced():
    columns = LOCAL_FEATURE_SETS["lean5"]
    train = _panel(indicators=("X1",), months=80)
    train["y_true"] = 0.0
    train.loc[train.index[:5], "y_true"] = 1.0
    model = fit_local_logistic_models(
        train,
        feature_columns=columns,
        seed=1,
        logistic_c=0.1,
        max_iter=500,
        minimum_local_rows=36,
        minimum_local_class_rows=8,
    )
    assert model.models == {}


def test_effective_local_threshold_matches_a_refit_threshold():
    """Raising the minimum-history threshold post hoc equals refitting with it."""
    signals = pd.DataFrame({
        "p_global": [0.60, 0.55, 0.50],
        "p_local__core8_c0.1": [0.90, 0.10, 0.70],
        "fallback__core8_c0.1": [False, False, True],
        "local_training_rows": [80, 40, 100],
    })
    probability, fallback = effective_local(signals, "core8_c0.1", 48)
    assert fallback.tolist() == [False, True, True]
    assert probability.tolist() == [0.90, 0.55, 0.50]


# ---------------------------------------------------------------------------
# Shrinkage arithmetic
# ---------------------------------------------------------------------------


def test_shrinkage_is_a_convex_combination():
    p_global = np.array([0.2, 0.5, 0.8])
    p_local = np.array([0.9, 0.1, 0.4])
    blended = shrink_probabilities(p_global, p_local, 0.25)
    expected = 0.25 * p_local + 0.75 * p_global
    assert blended == pytest.approx(expected)
    assert shrink_probabilities(p_global, p_local, 0.0) == pytest.approx(p_global)
    assert shrink_probabilities(p_global, p_local, 1.0) == pytest.approx(p_local)


def test_shrinkage_rejects_weights_outside_the_unit_interval():
    with pytest.raises(ValueError):
        shrink_probabilities(np.array([0.5]), np.array([0.5]), 1.5)
    with pytest.raises(ValueError):
        shrink_probabilities(np.array([0.5]), np.array([0.5]), -0.1)


def test_sample_aware_weight_ramps_with_history_and_zeroes_on_fallback():
    rows = np.array([36.0, 60.0, 120.0, 400.0, 500.0])
    fallback = np.array([False, False, False, False, True])
    weights = sample_aware_weights(
        rows, fallback, maximum_weight=0.4, minimum_rows=36, reference_rows=120
    )
    assert weights[0] == pytest.approx(0.0)
    assert weights[1] == pytest.approx(0.4 * (60 - 36) / (120 - 36))
    assert weights[2] == pytest.approx(0.4)
    assert weights[3] == pytest.approx(0.4)  # clipped
    assert weights[4] == pytest.approx(0.0)  # fallback row
    assert np.all((weights >= 0.0) & (weights <= 0.4))


def test_sample_aware_weight_requires_a_reference_above_the_minimum():
    with pytest.raises(ValueError):
        sample_aware_weights(
            np.array([50.0]), np.array([False]), 0.3, minimum_rows=60,
            reference_rows=60,
        )


# ---------------------------------------------------------------------------
# Indicator-interaction encoding
# ---------------------------------------------------------------------------


def _interaction_panel(months=60, indicators=("X1", "X2")) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for indicator in indicators:
        for origin in range(1, months + 1):
            row = {"origin_position": origin, "indicator_id": indicator}
            for column in FEATURE_COLUMNS:
                row[column] = float(rng.normal())
            row["y_true"] = float(rng.random() > 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def test_interaction_design_width_and_block_layout():
    train = _interaction_panel()
    features = ["momentum_3", "direction_lag_1"]
    model = fit_interaction_logistic_model(
        train, seed=1, logistic_c=0.1, max_iter=200, interaction_features=features
    )
    indicators = list(model.encoder.categories_[0])
    coefficients = model.classifier.coef_[0]
    assert len(coefficients) == len(model.coefficient_names)
    # 44 numeric + one-hot block + (indicators x interaction features).
    assert model.coefficient_names[:len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    interaction_terms = [
        name for name in model.coefficient_names if "]x" in name
    ]
    assert len(interaction_terms) == len(indicators) * len(features)
    assert "indicator[X1]xmomentum_3" in interaction_terms
    assert "indicator[X2]xdirection_lag_1" in interaction_terms


def test_interaction_model_rejects_unknown_interaction_features():
    train = _interaction_panel()
    with pytest.raises(ValueError):
        fit_interaction_logistic_model(
            train, seed=1, logistic_c=0.1, max_iter=200,
            interaction_features=["not_a_feature"],
        )


def test_interaction_model_is_deterministic():
    train = _interaction_panel()
    test = train.groupby("indicator_id", as_index=False).tail(3)
    features = ["momentum_3", "direction_lag_1"]
    first = predict_interaction_probability(
        fit_interaction_logistic_model(
            train, seed=1, logistic_c=0.1, max_iter=200,
            interaction_features=features,
        ),
        test,
    )
    second = predict_interaction_probability(
        fit_interaction_logistic_model(
            train, seed=1, logistic_c=0.1, max_iter=200,
            interaction_features=features,
        ),
        test,
    )
    assert first == pytest.approx(second)


def test_local_models_are_deterministic():
    columns = LOCAL_FEATURE_SETS["core8"]
    train = _panel()
    kwargs = dict(
        feature_columns=columns, seed=1, logistic_c=0.1, max_iter=500,
        minimum_local_rows=36, minimum_local_class_rows=8,
    )
    first = local_coefficients(fit_local_logistic_models(train, **kwargs))
    second = local_coefficients(fit_local_logistic_models(train, **kwargs))
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Locked-origin guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("origin", [268, 280, 315, 316])
def test_locked_origins_are_refused(origin):
    with pytest.raises(AssertionError, match="Locked origins|readable horizon"):
        assert_origins_unlocked([120, origin], SETTINGS)


def test_permitted_origins_pass_the_guard():
    assert_origins_unlocked(range(120, 267), SETTINGS) is None


def test_local_spec_key_is_stable():
    assert LocalSpec("core8", 0.1).key == "core8_c0.1"
    assert LocalSpec("lean5", 0.01).key == "lean5_c0.01"
