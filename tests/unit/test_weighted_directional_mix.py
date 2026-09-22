"""Weight semantics and causal calibration boundary for the research trial."""

import numpy as np
import pandas as pd

from research.weighted_directional_mix import (
    blend_arms,
    fit_arm_calibration,
    score_arms,
)


def _panel():
    return pd.DataFrame(
        {
            "origin_position": np.repeat(np.arange(120, 145), 2),
            "p_up_selection_score": np.tile([0.65, 0.45], 25),
            "logistic": np.tile([0.40, 0.60], 25),
            "indicator_prior": np.tile([0.60, 0.40], 25),
            "p_down_indicator_prior": np.tile([0.40, 0.60], 25),
            "y_true": np.tile([1, 0], 25),
        }
    )


def test_weight_endpoints_preserve_arm_meaning():
    panel = pd.DataFrame({
        "p_up_arm": [0.7], "p_down_arm": [0.6],
        "p_up_prior_arm": [0.65], "p_down_prior_arm": [0.55],
    })
    direct = blend_arms(panel, 1.0, 1.0)
    opposite = blend_arms(panel, 0.0, 0.0)
    assert direct.loc[0, "up_cal"] == 0.7
    assert direct.loc[0, "down_cal"] == 0.6
    assert np.isclose(opposite.loc[0, "up_cal"], 0.4)
    assert np.isclose(opposite.loc[0, "down_cal"], 0.3)
    prior = blend_arms(panel, 0.0, 0.0, "prior")
    assert prior.loc[0, "up_cal"] == 0.65
    assert prior.loc[0, "down_cal"] == 0.55


def test_future_label_mutation_does_not_change_calibration():
    panel = _panel()
    original = fit_arm_calibration(panel, through_origin=143)
    changed = panel.copy()
    changed.loc[changed["origin_position"].eq(144), "y_true"] = 1 - changed.loc[
        changed["origin_position"].eq(144), "y_true"
    ]
    mutated = fit_arm_calibration(changed, through_origin=143)
    original_scores = score_arms(panel, original)
    mutated_scores = score_arms(panel, mutated)
    np.testing.assert_allclose(original_scores["p_up_arm"], mutated_scores["p_up_arm"])
    np.testing.assert_allclose(original_scores["p_down_arm"], mutated_scores["p_down_arm"])
