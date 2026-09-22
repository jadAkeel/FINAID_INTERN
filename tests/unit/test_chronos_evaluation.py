from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_select.chronos_evaluation import (
    ACTIVE_WEIGHT_GRID,
    apply_causal_calibration,
    compute_diversity_metrics,
    compute_probability_metrics,
    evaluate_chronos_gate,
    evaluate_chronos_zero_shot,
    extract_active_probability,
    origin_blocked_paired_bootstrap,
    select_active_weight,
    validate_evaluation_dataframe,
)


def _make_synthetic_frames(
    start_origin: int = 120,
    end_origin: int = 266,
    num_indicators: int = 25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    origins = list(range(start_origin, end_origin + 1))
    indicators = [f"X{i}" for i in range(1, num_indicators + 1)]

    in_rows = []
    out_rows = []
    act_rows = []

    for o in origins:
        # Choose 15 to 20 indicators to be accepted in active
        k = int(rng.integers(min(15, num_indicators), min(20, num_indicators) + 1))
        accepted_set = set(rng.choice(indicators, size=k, replace=False))

        for ind in indicators:
            p_raw = float(rng.uniform(0.1, 0.9))
            p_act_cal = float(rng.uniform(0.1, 0.9))
            y_val = float(rng.choice([0.0, 1.0]))

            in_rows.append({
                "origin_position": o,
                "indicator_id": ind,
                "p_up": p_raw,
                "p_up_raw": p_raw,
                "eligible": True,
                "data_quality_ok": True,
                "error_flag": False,
                "locked_evaluation_read": False,
            })
            out_rows.append({
                "origin_position": o,
                "indicator_id": ind,
                "y_true": y_val,
                "change_t_to_t1": 1.0 if y_val == 1.0 else -1.0,
                "locked_evaluation_read": False,
            })
            act_rows.append({
                "origin_position": o,
                "indicator_id": ind,
                "p_up_calibrated": p_act_cal,
                "p_up": p_act_cal,
                "selection_score": 999.0,  # Must never be used as probability
                "accepted": ind in accepted_set,
                "predicted_direction": "Up" if p_act_cal >= 0.5 else "Down",
                "eligible": True,
                "locked_evaluation_read": False,
                "y_true": y_val,
            })

    inputs_df = pd.DataFrame(in_rows)
    outcomes_df = pd.DataFrame(out_rows)
    active_df = pd.DataFrame(act_rows)
    return inputs_df, outcomes_df, active_df


def test_origin_boundary_and_locked_rejections():
    # Max origin <= 266 respected, origin >= 267 rejected
    bad_267 = pd.DataFrame({
        "origin_position": [120, 267],
        "indicator_id": ["X1", "X1"],
    })
    with pytest.raises(ValueError, match="exceeds maximum allowed 266"):
        validate_evaluation_dataframe(bad_267, "test_267")

    bad_268 = pd.DataFrame({
        "origin_position": [120, 268],
        "indicator_id": ["X1", "X1"],
    })
    with pytest.raises(ValueError, match="exceeds maximum allowed 266"):
        validate_evaluation_dataframe(bad_268, "test_268")

    bad_min = pd.DataFrame({
        "origin_position": [119, 200],
        "indicator_id": ["X1", "X1"],
    })
    with pytest.raises(ValueError, match="below minimum allowed 120"):
        validate_evaluation_dataframe(bad_min, "test_min")

    # Duplicate keys rejected
    dup_df = pd.DataFrame({
        "origin_position": [120, 120],
        "indicator_id": ["X1", "X1"],
    })
    with pytest.raises(ValueError, match="duplicate origin_position and indicator_id keys"):
        validate_evaluation_dataframe(dup_df, "test_dup")

    # locked_evaluation_read=True rejected
    locked_df = pd.DataFrame({
        "origin_position": [120, 150],
        "indicator_id": ["X1", "X2"],
        "locked_evaluation_read": [False, True],
    })
    with pytest.raises(ValueError, match="locked_evaluation_read=True"):
        validate_evaluation_dataframe(locked_df, "test_locked")


def test_active_probability_extraction():
    df = pd.DataFrame({
        "p_up_calibrated": [0.65, np.nan, 0.80],
        "p_up": [0.55, 0.40, 0.70],
        "selection_score": [10.0, 20.0, 30.0],
    })
    probs = extract_active_probability(df)
    # Row 0: finite p_up_calibrated takes precedence over p_up
    assert probs.iloc[0] == 0.65
    # Row 1: p_up_calibrated is NaN, falls back to p_up
    assert probs.iloc[1] == 0.40
    # Row 2: uses p_up_calibrated
    assert probs.iloc[2] == 0.80
    # selection_score is completely ignored
    assert (probs != df["selection_score"]).all()


def test_tuning_oof_fit_through_bound_and_prequential_causality():
    inputs_df, outcomes_df, active_df = _make_synthetic_frames(
        start_origin=120,
        end_origin=179,
        num_indicators=10,
        seed=123,
    )
    from forecast_select.chronos_cache import attach_chronos_outcomes

    combined = attach_chronos_outcomes(inputs_df, outcomes_df)
    active_sub = active_df[["origin_position", "indicator_id", "p_up_calibrated", "accepted", "predicted_direction"]].rename(
        columns={"accepted": "active_accepted", "predicted_direction": "active_predicted_direction"}
    )
    active_sub["p_up_active"] = active_df["p_up_calibrated"]
    active_sub["active_eligible"] = True
    joined = combined.merge(active_sub, on=["origin_position", "indicator_id"])
    joined["p_up_chronos_raw"] = joined["p_up"]
    joined["chronos_eligible"] = True
    joined["common_eligible"] = True

    calibrated = apply_causal_calibration(joined, min_history_origins=12)

    # In Tuning: check fit_through_origin <= origin_position - 2
    for orig, rows in calibrated.groupby("origin_position"):
        fit_through = rows["calibration_fit_through_origin"].iloc[0]
        if fit_through is not None:
            assert int(fit_through) <= int(orig) - 2


def test_validation_and_confirmation_label_mutation_invariance():
    inputs_df, outcomes_df, active_df = _make_synthetic_frames(
        start_origin=120,
        end_origin=266,
        num_indicators=20,
        seed=42,
    )

    # Baseline evaluation
    cal_base, _, sum_base = evaluate_chronos_zero_shot(
        inputs_df,
        outcomes_df,
        active_df,
        min_history_origins=12,
        seed=20260727,
    )

    # Mutate labels unavailable at the first Validation origin: origin 179 and later.
    mutated_outcomes = outcomes_df.copy()
    mutated_active = active_df.copy()
    val_conf_mask = mutated_outcomes["origin_position"].ge(179)
    # Invert binary labels
    mutated_outcomes.loc[val_conf_mask, "y_true"] = 1.0 - mutated_outcomes.loc[val_conf_mask, "y_true"]
    active_mask = mutated_active["origin_position"].ge(179)
    mutated_active.loc[active_mask, "y_true"] = 1.0 - mutated_active.loc[active_mask, "y_true"]

    cal_mut, _, sum_mut = evaluate_chronos_zero_shot(
        inputs_df,
        mutated_outcomes,
        mutated_active,
        min_history_origins=12,
        seed=20260727,
    )

    # 1. Tuning Chronos calibrated probabilities must be identical
    tuning_mask = cal_base["origin_position"].between(120, 179)
    np.testing.assert_allclose(
        cal_base.loc[tuning_mask, "p_up_chronos_calibrated"].to_numpy(),
        cal_mut.loc[tuning_mask, "p_up_chronos_calibrated"].to_numpy(),
    )

    # 2. Selected active weight must be identical
    assert sum_base["selected_active_weight"] == sum_mut["selected_active_weight"]

    # 3. Tuning grid searches must be identical
    assert sum_base["active_weight_grid_search"] == sum_mut["active_weight_grid_search"]

    # Frozen Validation calibration is also invariant to origin 179+ labels.
    validation_mask = cal_base["origin_position"].between(180, 219)
    np.testing.assert_allclose(
        cal_base.loc[validation_mask, "p_up_chronos_calibrated"].to_numpy(),
        cal_mut.loc[validation_mask, "p_up_chronos_calibrated"].to_numpy(),
    )
    assert sum_base["frozen_policy_fit_through_origin"] == 178


def test_active_weight_selection_and_tie_breaking():
    # Test tie-breaking prefers 1.0
    tuning_df = pd.DataFrame({
        "common_eligible": [True, True],
        "y_true": [1.0, 0.0],
        # If active prob and chronos prob are identical, all blend weights have identical Brier score
        "p_up_active": [0.6, 0.4],
        "p_up_chronos_calibrated": [0.6, 0.4],
    })
    selected_weight, grid_results = select_active_weight(tuning_df, grid=ACTIVE_WEIGHT_GRID)
    # Tie must prefer 1.0
    assert selected_weight == 1.0

    # Test weight where pure active (weight 1.0) is clearly best
    tuning_df2 = pd.DataFrame({
        "common_eligible": [True, True],
        "y_true": [1.0, 0.0],
        "p_up_active": [1.0, 0.0],
        "p_up_chronos_calibrated": [0.0, 1.0],
    })
    weight2, _ = select_active_weight(tuning_df2, grid=ACTIVE_WEIGHT_GRID)
    assert weight2 == 1.0

    # Test weight where pure chronos (weight 0.0) is clearly best
    tuning_df3 = pd.DataFrame({
        "common_eligible": [True, True],
        "y_true": [1.0, 0.0],
        "p_up_active": [0.0, 1.0],
        "p_up_chronos_calibrated": [1.0, 0.0],
    })
    weight3, _ = select_active_weight(tuning_df3, grid=ACTIVE_WEIGHT_GRID)
    assert weight3 == 0.0


def test_matched_monthly_coverage_equal():
    inputs_df, outcomes_df, active_df = _make_synthetic_frames(
        start_origin=120,
        end_origin=266,
        num_indicators=25,
        seed=999,
    )
    calibrated, matched_df, summary = evaluate_chronos_zero_shot(
        inputs_df,
        outcomes_df,
        active_df,
        min_history_origins=6,
        seed=20260727,
    )

    records = summary["matched_coverage"]["validation"]["monthly_records"]
    assert len(records) == 40  # 180 to 219 = 40 origins

    for rec in records:
        orig = rec["origin_position"]
        act_calls = rec["calls"]
        # Candidate accepted rows for this origin in matched_df
        cand_rows = matched_df[matched_df["origin_position"] == orig]
        assert len(cand_rows) == act_calls
        # Check active calls in original active_df
        act_orig_calls = int(active_df[(active_df["origin_position"] == orig) & (active_df["accepted"])]["indicator_id"].nunique())
        assert act_calls == act_orig_calls


def test_blocked_bootstrap_deterministic():
    deltas = [1, 0, -1, 2, 1, 0, -1, 1, 2, 0, 1, -1] * 3  # 36 origins
    res1 = origin_blocked_paired_bootstrap(deltas, block_size=6, replicates=500, seed=20260727)
    res2 = origin_blocked_paired_bootstrap(deltas, block_size=6, replicates=500, seed=20260727)

    assert res1["bootstrap_p10"] == res2["bootstrap_p10"]
    assert res1["bootstrap_p50"] == res2["bootstrap_p50"]
    assert res1["bootstrap_p90"] == res2["bootstrap_p90"]
    assert res1["total_hit_delta"] == res2["total_hit_delta"]
    assert res1["mean_monthly_hit_delta"] == res2["mean_monthly_hit_delta"]


def test_promotion_false_even_when_gate_passes():
    val_matched = {
        "total_hit_delta": 15.0,
        "bootstrap_p10": 2.5,
    }
    val_metrics = {
        "blend": {"brier": 0.20},
        "active": {"brier": 0.25},
    }
    conf_matched = {
        "total_hit_delta": 5.0,
    }
    gate = evaluate_chronos_gate(
        validation_matched_bootstrap=val_matched,
        validation_metrics=val_metrics,
        confirmation_matched_bootstrap=conf_matched,
        locked_evaluation_read=False,
        max_origin=266,
    )

    # All 5 criteria pass
    assert gate["nonlocked_gate_passed"] is True
    # Hard barrier: promotion_eligible is strictly False
    assert gate["promotion_eligible"] is False
    assert gate["promotion_requires_locked_evaluation"] is True
    assert gate["allow_promotion"] is False
    assert gate["active_model_changed"] is False


def test_window_metrics_and_diversity():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p1 = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4])
    p2 = np.array([0.8, 0.2, 0.3, 0.1, 0.9, 0.4, 0.3, 0.7])

    m = compute_probability_metrics(y, p1)
    assert m["n"] == 8
    assert m["hits"] == 8
    assert m["accuracy"] == 1.0
    assert 0.0 <= m["brier"] <= 0.1
    assert m["roc_auc"] == 1.0
    assert 0.0 <= m["ece"] <= 0.5

    d = compute_diversity_metrics(p1, p2, y)
    assert d["n"] == 8
    assert isinstance(d["pearson"], float)
    assert isinstance(d["spearman"], float)
    assert d["disagreement_count"] >= 0
    assert 0.0 <= d["disagreement_rate"] <= 1.0
    assert d["double_fault_count"] >= 0
