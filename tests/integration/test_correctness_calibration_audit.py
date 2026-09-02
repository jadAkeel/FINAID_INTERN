from __future__ import annotations

import json

import pandas as pd

from forecast_select.calibration_audit import (
    build_correctness_calibration_audit,
    correctness_calibration_root,
)


def test_correctness_calibration_audit_is_reproducible_and_honest() -> None:
    summary_path = build_correctness_calibration_audit()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["locked_evaluation_read"] is False
    assert summary["locked_evaluation_path_read"] is False
    assert summary["decision"] == "no_individual_correctness_probability"
    assert summary["individual_probability_release_gate"]["passed"] is False

    root = correctness_calibration_root()
    comparison = pd.read_csv(root / "metrics/model_comparison.csv")
    reliability = pd.read_csv(root / "metrics/reliability_buckets.csv")
    coverage = pd.read_csv(root / "metrics/accuracy_coverage.csv")
    assert {"active", "adaptive"} == set(comparison["model"])
    assert {"all", "validation", "confirmation"}.issubset(
        set(coverage["window"])
    )
    assert {
        "observed_wilson_ci95_low",
        "observed_wilson_ci95_high",
    }.issubset(reliability.columns)
    for model in ["active", "adaptive"]:
        upgraded = pd.read_parquet(root / f"artifacts/{model}_scored.parquet")
        assert upgraded["correctness_probability"].isna().all()
        assert upgraded["correctness_lcb"].isna().all()
        ready = upgraded[upgraded["correctness_fit_through_origin"].notna()]
        assert (
            ready["correctness_fit_through_origin"]
            <= ready["origin_position"] - 2
        ).all()
