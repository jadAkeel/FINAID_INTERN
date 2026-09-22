"""Historical cohort evidence for the active selector's delivery output."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_reliability(
    predictions: pd.DataFrame,
    *,
    first_origin: int = 180,
    last_origin: int = 219,
    block_months: int = 6,
    replicates: int = 5000,
    seed: int = 20260727,
) -> dict:
    """Summarize selected-call accuracy with a date-block bootstrap.

    This describes a historical cohort, not correctness odds for a new call.
    """
    if block_months < 1 or replicates < 1:
        raise ValueError("block_months and replicates must be positive")
    if last_origin < first_origin:
        raise ValueError("The historical origin window is reversed")
    selected = predictions.loc[
        predictions["accepted"].fillna(False).astype(bool)
        & predictions["origin_position"].between(first_origin, last_origin)
        & predictions["y_true"].notna()
    ].copy()
    if not selected["predicted_direction"].isin({"Up", "Down"}).all():
        raise ValueError("Selected directions must be Up or Down")
    selected["hit"] = np.where(
        selected["predicted_direction"].eq("Up"),
        selected["y_true"].eq(1),
        selected["y_true"].eq(0),
    ).astype(int)
    monthly = selected.groupby("origin_position", sort=True)["hit"].agg(
        ["sum", "count"]
    )
    expected = np.arange(first_origin, last_origin + 1)
    if not np.array_equal(monthly.index.to_numpy(dtype=int), expected):
        raise ValueError("Historical reliability requires every month in the window")
    if not monthly["count"].between(15, 20).all():
        raise ValueError("Historical monthly coverage must stay within 15–20")
    n_months = len(monthly)
    block_months = min(block_months, n_months)
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, n_months - block_months + 1,
        size=(replicates, (n_months + block_months - 1) // block_months),
    )
    sampled_months = (
        starts[:, :, None] + np.arange(block_months)[None, None, :]
    ).reshape(replicates, -1)[:, :n_months]
    hits = monthly["sum"].to_numpy(dtype=int)
    calls = monthly["count"].to_numpy(dtype=int)
    bootstrap_rates = hits[sampled_months].sum(axis=1) / calls[
        sampled_months
    ].sum(axis=1)
    total_hits = int(hits.sum())
    total_calls = int(calls.sum())
    return {
        "scope": "historical_selected_call_cohort_not_individual_confidence",
        "origins": [first_origin, last_origin],
        "months": n_months,
        "hits": total_hits,
        "calls": total_calls,
        "observed_accuracy": total_hits / total_calls,
        "monthly_block_bootstrap_p05_accuracy": float(
            np.quantile(bootstrap_rates, 0.05)
        ),
        "monthly_block_bootstrap_p95_accuracy": float(
            np.quantile(bootstrap_rates, 0.95)
        ),
        "bootstrap_block_months": block_months,
        "bootstrap_replicates": replicates,
        "individual_correctness_probability": None,
        "interpretation": (
            "Descriptive replay on previously viewed months; neither the observed "
            "rate nor its bootstrap range guarantees future or individual accuracy."
        ),
    }


def historical_reliability_or_unavailable(
    predictions: pd.DataFrame, **kwargs
) -> dict:
    """Return the reliability summary, or a reason why it is unavailable.

    `historical_reliability` is deliberately strict: it refuses to report a rate
    computed on a partial window. That strictness must not take the forecast
    itself down, so the delivery path reports the refusal instead of raising.
    An unavailable summary is never a reliability claim.
    """
    try:
        return historical_reliability(predictions, **kwargs)
    except (ValueError, KeyError) as exc:
        return {
            "scope": "historical_selected_call_cohort_not_individual_confidence",
            "available": False,
            "reason": str(exc),
            "individual_correctness_probability": None,
            "interpretation": (
                "No historical reliability evidence is reported for this run; "
                "the required selected-call history was not available."
            ),
        }
