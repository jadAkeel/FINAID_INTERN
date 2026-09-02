import numpy as np
import pandas as pd

from forecast_select.expansion_quality import (
    build_expansion_quality_panel,
    expansion_quality_cap_schedule,
    graduated_cap_schedule,
    paired_monthly_hit_bootstrap,
)
from forecast_select.expansion_experiment_runner import _bootstrap_gate


def _replay_panel() -> pd.DataFrame:
    rows = []
    for origin in range(120, 152):
        for rank in range(1, 21):
            rows.append({
                "origin_position": origin,
                "indicator_id": f"X{rank}",
                "base_up_rank": float(rank),
                "p_up_selection_score": 0.9 - rank / 100.0,
                "level_c_ready": True,
                "forecast_market_breadth": 0.55 + (origin % 3) * 0.1,
                "regime_stress": 0.4,
                "y_true": float((origin + rank) % 3 != 0),
                "asset_group": f"g{rank % 4}",
                "p_down_global": 0.3,
                "p_down_local": 0.3 + rank / 1000.0,
                "p_down_pattern": 0.3,
                "p_down_indicator_prior": 0.3,
                "p_up_generalized_calibrated": 0.8 - rank / 100.0,
                "p_up_calibrated": 0.79 - rank / 100.0,
                "accepted": rank <= 15,
                "predicted_direction": "Up",
            })
    return pd.DataFrame(rows)


def test_expansion_quality_forecast_uses_labels_only_through_t_minus_2():
    panel = build_expansion_quality_panel(
        _replay_panel(),
        minimum_history=6,
        ridge_alpha=10.0,
    )
    fitted = panel[panel["quality_fit_through_origin"].notna()]
    assert not fitted.empty
    assert fitted["quality_fit_through_origin"].le(
        fitted["origin_position"] - 2
    ).all()


def test_future_outcomes_do_not_change_prior_quality_forecasts():
    source = _replay_panel()
    before = build_expansion_quality_panel(source, minimum_history=6)
    changed = source.copy()
    changed.loc[changed["origin_position"].ge(149), "y_true"] = 0.0
    after = build_expansion_quality_panel(changed, minimum_history=6)
    columns = ["forecast_quality_16_17", "forecast_quality_18_20"]
    pd.testing.assert_frame_equal(
        before.loc[before["origin_position"].le(150), columns],
        after.loc[after["origin_position"].le(150), columns],
    )


def test_graduated_and_quality_schedules_use_only_15_17_20():
    panel = build_expansion_quality_panel(_replay_panel(), minimum_history=6)
    graduated = graduated_cap_schedule(
        panel,
        lower_threshold=0.60,
        upper_threshold=0.70,
    )
    quality = expansion_quality_cap_schedule(
        panel,
        rank_16_17_threshold=0.55,
        rank_18_20_threshold=0.60,
    )
    assert set(graduated.values()) == {15, 17, 20}
    assert set(quality.values()) <= {15, 17, 20}
    assert {15, 17} <= set(graduated.values())


def test_paired_monthly_bootstrap_reports_exact_observed_delta():
    baseline = _replay_panel()
    candidate = baseline.copy()
    candidate.loc[
        candidate["origin_position"].eq(130)
        & candidate["base_up_rank"].eq(16),
        "accepted",
    ] = True
    result = paired_monthly_hit_bootstrap(
        candidate,
        baseline,
        [120, 151],
        block_months=6,
        replicates=50,
        seed=7,
    )
    expected = int(
        candidate.loc[
            candidate["origin_position"].eq(130)
            & candidate["base_up_rank"].eq(16),
            "y_true",
        ].iloc[0]
    )
    assert result["total_hit_delta"] == expected
    assert np.isfinite(result["bootstrap_p10"])


def test_bootstrap_gate_requires_nonnegative_paired_lower_bound():
    assert not _bootstrap_gate(
        {"bootstrap_p10": 0.55},
        {"bootstrap_p10": 0.54},
        {"bootstrap_p10": -0.01},
    )
    assert _bootstrap_gate(
        {"bootstrap_p10": 0.55},
        {"bootstrap_p10": 0.54},
        {"bootstrap_p10": 0.0},
    )


def test_expansion_panel_rejects_duplicate_base_ranks():
    source = _replay_panel()
    mask = source["origin_position"].eq(120) & source["base_up_rank"].eq(20)
    source.loc[mask, "base_up_rank"] = 19.0
    with np.testing.assert_raises_regex(ValueError, "unique base ranks"):
        build_expansion_quality_panel(source, minimum_history=6)


def test_expansion_panel_rejects_non_integral_base_ranks():
    source = _replay_panel()
    mask = source["origin_position"].eq(120) & source["base_up_rank"].eq(20)
    source.loc[mask, "base_up_rank"] = 19.5
    with np.testing.assert_raises_regex(ValueError, "non-integral"):
        build_expansion_quality_panel(source, minimum_history=6)


def test_expansion_panel_rejects_fewer_than_twenty_candidates():
    source = _replay_panel()
    source = source[
        ~(
            source["origin_position"].eq(120)
            & source["base_up_rank"].eq(20)
        )
    ]
    with np.testing.assert_raises_regex(ValueError, "fewer than 20"):
        build_expansion_quality_panel(source, minimum_history=6)
