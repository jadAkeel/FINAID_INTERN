"""Training boundary checks for the research directional mix model."""

import numpy as np
import pandas as pd

from research.trained_directional_mix import features, fit_correctness, score


def test_future_label_change_cannot_change_fitted_correctness_scores():
    origins = np.repeat(np.arange(120, 145), 2)
    panel = pd.DataFrame(
        {
            "origin_position": origins,
            "p_up_selection_score": np.tile([0.65, 0.45], 25),
            "logistic": np.tile([0.40, 0.60], 25),
            "regime_stress": np.repeat(np.linspace(0.2, 0.8, 25), 2),
            "y_true": np.tile([1, 0], 25),
        }
    )
    original = fit_correctness(panel, through_origin=143, regularization=1.0)
    mutated = panel.copy()
    mutated.loc[mutated["origin_position"].eq(144), "y_true"] = 1 - mutated.loc[
        mutated["origin_position"].eq(144), "y_true"
    ]
    changed = fit_correctness(mutated, through_origin=143, regularization=1.0)
    np.testing.assert_allclose(
        original.predict_proba(features(panel, 1)),
        changed.predict_proba(features(panel, 1)),
    )


def test_scores_do_not_need_future_labels():
    panel = pd.DataFrame(
        {
            "origin_position": np.repeat(np.arange(120, 144), 2),
            "p_up_selection_score": np.tile([0.65, 0.45], 24),
            "logistic": np.tile([0.40, 0.60], 24),
            "regime_stress": 0.5,
            "y_true": np.tile([1, 0], 24),
        }
    )
    model = fit_correctness(panel, through_origin=143, regularization=1.0)
    future = panel.tail(2).copy()
    future["y_true"] = np.nan
    scored = score(future, model)
    assert scored["up_cal"].between(0, 1).all()
    assert scored["down_cal"].between(0, 1).all()
