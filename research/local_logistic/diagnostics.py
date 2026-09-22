"""Follow-up diagnostics for the Local / Interaction Logistic study.

Run after ``python -m forecast_select.local_logistic_runner all``:

    python research/local_logistic/diagnostics.py

Answers four questions the headline tables leave open:

1. Are the per-indicator local slopes reproducible across time, measured on
   origins far enough apart that the expanding training windows differ
   substantially?
2. Is the spread of per-indicator (local - global) accuracy larger than
   sampling noise would produce on its own?
3. Do the indicators where local modelling helps ever reach the top 15?
4. How much of the top-15 result depends on the logistic signal at all,
   compared with the trailing 48-month indicator prior?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "research/local_logistic/artifacts"
METRICS = ROOT / "research/local_logistic/metrics"
SEED = 20260727


def coefficient_reproducibility() -> pd.DataFrame:
    """Correlate per-indicator slopes fitted at two widely separated origins.

    Expanding windows overlap, so consecutive origins are nearly identical by
    construction. Comparing origin 179 with origin 266 uses training slices
    that differ by 87 additional months.
    """
    coefficients = pd.read_parquet(ARTIFACTS / "local_coefficients.parquet")
    frozen = json.loads(
        (METRICS / "frozen_configuration.json").read_text(encoding="utf-8")
    )
    spec = (
        f"{frozen['fixed_shrinkage']['feature_set']}"
        f"_c{float(frozen['fixed_shrinkage']['logistic_c']):g}"
    )
    subset = coefficients[coefficients["spec"].eq(spec)]
    early = subset[subset["origin_position"].eq(179)].set_index("indicator_id")
    late = subset[subset["origin_position"].eq(266)].set_index("indicator_id")
    shared = early.index.intersection(late.index)
    records = []
    for column in [c for c in subset.columns if c.startswith("coef_")]:
        a = early.loc[shared, column].to_numpy(dtype=float)
        b = late.loc[shared, column].to_numpy(dtype=float)
        records.append({
            "feature": column.removeprefix("coef_"),
            "indicators": int(len(shared)),
            "correlation_179_vs_266": float(np.corrcoef(a, b)[0, 1]),
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
            "std_origin_179": float(a.std(ddof=1)),
            "std_origin_266": float(b.std(ddof=1)),
        })
    return pd.DataFrame(records)


def per_indicator_spread_null(replicates: int = 2000) -> dict:
    """Compare the observed spread of (local - global) accuracy to a null.

    The null keeps each indicator's out-of-sample row count and the pooled
    disagreement rate between the two models, then reassigns which model was
    right on each disagreeing row by a fair coin. That is the spread expected
    when neither model is systematically better for any indicator.
    """
    predictions = pd.read_parquet(ARTIFACTS / "predictions.parquet")
    rows = predictions[
        predictions["period"].isin(["validation", "confirmation"])
        & predictions["y_true"].notna()
    ].copy()
    y = rows["y_true"].astype(int).to_numpy()
    global_hit = ((rows["p_global"].to_numpy() >= 0.5).astype(int) == y)
    local_hit = ((rows["p_local"].to_numpy() >= 0.5).astype(int) == y)
    disagree = global_hit != local_hit

    frame = pd.DataFrame({
        "indicator_id": rows["indicator_id"].to_numpy(),
        "global_hit": global_hit,
        "local_hit": local_hit,
        "disagree": disagree,
    })
    observed = frame.groupby("indicator_id").apply(
        lambda g: g["local_hit"].mean() - g["global_hit"].mean(),
        include_groups=False,
    )
    observed_std = float(observed.std(ddof=1))

    rng = np.random.default_rng(SEED)
    indicator_codes = pd.Categorical(frame["indicator_id"]).codes
    disagreement_positions = np.flatnonzero(disagree)
    null_std = []
    for _ in range(replicates):
        simulated_local = global_hit.copy()
        flips = rng.random(len(disagreement_positions)) < 0.5
        simulated_local[disagreement_positions] = np.where(
            flips, ~global_hit[disagreement_positions],
            global_hit[disagreement_positions],
        )
        difference = (
            np.bincount(indicator_codes, weights=simulated_local.astype(float))
            - np.bincount(indicator_codes, weights=global_hit.astype(float))
        ) / np.bincount(indicator_codes)
        null_std.append(float(difference.std(ddof=1)))
    null_std = np.array(null_std)
    return {
        "indicators": int(observed.size),
        "observed_std_of_accuracy_difference": observed_std,
        "null_median_std": float(np.median(null_std)),
        "null_p95_std": float(np.quantile(null_std, 0.95)),
        "observed_exceeds_null_p95": bool(observed_std > np.quantile(null_std, 0.95)),
        "empirical_p_value": float(np.mean(null_std >= observed_std)),
        "disagreement_rate": float(disagree.mean()),
        "rows": int(len(frame)),
    }


def selection_reach() -> pd.DataFrame:
    """Did local modelling help on indicators the selector ever picks?"""
    predictions = pd.read_parquet(ARTIFACTS / "predictions.parquet")
    rows = predictions[
        predictions["period"].isin(["validation", "confirmation"])
        & predictions["y_true"].notna()
    ].copy()
    y = rows["y_true"].astype(int)
    rows["global_hit"] = (rows["p_global"] >= 0.5).astype(int).eq(y)
    rows["local_hit"] = (rows["p_local"] >= 0.5).astype(int).eq(y)
    grouped = rows.groupby("indicator_id").agg(
        rows=("y_true", "size"),
        selected_global=("selected_global", "sum"),
        selected_hybrid=("selected_fixed_shrinkage", "sum"),
        global_accuracy=("global_hit", "mean"),
        local_accuracy=("local_hit", "mean"),
    ).reset_index()
    grouped["local_minus_global"] = (
        grouped["local_accuracy"] - grouped["global_accuracy"]
    )
    grouped["ever_selected"] = grouped["selected_global"] > 0
    return grouped


def signal_contribution() -> pd.DataFrame:
    """Replace the logistic signal with uninformative ones and re-run the selector.

    ``constant`` gives every indicator the same ``p_up``, so the ranking is
    driven purely by the trailing 48-month indicator prior. ``shuffled``
    permutes the global probabilities within each origin, keeping their
    distribution but destroying which indicator they belong to.
    """
    from forecast_select.local_logistic_metrics import selector_metrics
    from forecast_select.local_logistic_pipeline import (
        apply_selector,
        prepare_experiment_panel,
    )

    shared = prepare_experiment_panel(ROOT)
    signals = pd.read_parquet(ARTIFACTS / "full_signals.parquet")
    windows = {
        "tuning": tuple(shared.settings["tuning_origins"]),
        "validation": tuple(shared.settings["validation_origins"]),
        "confirmation": tuple(shared.settings["confirmation_origins"]),
    }
    rng = np.random.default_rng(SEED)
    signals["_constant"] = 0.5
    signals["_shuffled"] = signals.groupby("origin_position")["p_global"].transform(
        lambda values: rng.permutation(values.to_numpy())
    )
    records = []
    for name, column in (
        ("global_logistic", "p_global"),
        ("constant_prior_only", "_constant"),
        ("shuffled_global", "_shuffled"),
    ):
        result = apply_selector(signals, column, shared)
        record = {"variant": name}
        for period, window in windows.items():
            metrics = selector_metrics(result, window)
            record[f"{period}_hits"] = metrics["hits"]
            record[f"{period}_calls"] = metrics["calls"]
            record[f"{period}_accuracy"] = metrics["accuracy"]
        records.append(record)
    return pd.DataFrame(records)


def main() -> int:
    METRICS.mkdir(parents=True, exist_ok=True)

    reproducibility = coefficient_reproducibility()
    reproducibility.to_csv(METRICS / "coefficient_reproducibility.csv", index=False)
    print("== Local slope reproducibility, origin 179 vs origin 266 ==")
    print(reproducibility.to_string(index=False))

    null = per_indicator_spread_null()
    (METRICS / "per_indicator_spread_null.json").write_text(
        json.dumps(null, indent=2) + "\n", encoding="utf-8"
    )
    print("\n== Per-indicator accuracy spread against a coin-flip null ==")
    print(json.dumps(null, indent=2))

    reach = selection_reach()
    reach.to_csv(METRICS / "selection_reach.csv", index=False)
    selected = reach[reach["ever_selected"]]
    never = reach[~reach["ever_selected"]]
    print("\n== Where local modelling helps, split by selector reach ==")
    print(
        f"indicators ever selected by the baseline: {len(selected)}; "
        f"never selected: {len(never)}"
    )
    for label, block in (("ever selected", selected), ("never selected", never)):
        print(
            f"  {label}: mean (local - global) accuracy "
            f"{block['local_minus_global'].mean() * 100:+.2f} pp, "
            f"SD {block['local_minus_global'].std(ddof=1) * 100:.2f} pp, "
            f"max {block['local_minus_global'].max() * 100:+.2f} pp"
        )
    weighted = float(
        (reach["local_minus_global"] * reach["selected_global"]).sum()
        / max(1, reach["selected_global"].sum())
    )
    print(
        "  selection-weighted mean (local - global) accuracy: "
        f"{weighted * 100:+.2f} pp"
    )

    contribution = signal_contribution()
    contribution.to_csv(METRICS / "signal_contribution.csv", index=False)
    print("\n== How much the top-15 result depends on the logistic signal ==")
    print(contribution.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
