"""Extreme-down sensing and guarded Down replacement policy logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "ret1_z",
    "mom3_z",
    "consec_down",
    "prev_shock",
    "market_down_share_prev",
    "dispersion_ratio_prev",
]

DEFAULT_POLICY = {
    "breadth_gate": 0.50,
    "risk_quantile": 0.90,
    "max_replacements": 2,
    "conviction_ceiling": 0.60,
}


def build_extreme_down_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal indicator-level features observable through each month."""
    indicators = [c for c in frame.columns if c.startswith("X")]
    values = frame[indicators].astype(float)
    rets = values.diff()

    def zscore(series: pd.DataFrame, window: int, minimum: int) -> pd.DataFrame:
        mean = series.shift(1).rolling(window, min_periods=minimum).mean()
        std = series.shift(1).rolling(window, min_periods=minimum).std()
        return ((series - mean) / std.replace(0, np.nan)).astype(float)

    ret1_z = zscore(rets, 60, 24)
    mom3_z = zscore(values.diff(3), 60, 24)

    down = rets.le(0).astype(float)
    prev1 = down.shift(1)
    prev2 = down.shift(2)
    prev3 = down.shift(3)
    consec_down = prev1 + prev1 * prev2 + prev1 * prev2 * prev3

    lower_tail = rets.shift(1).rolling(60, min_periods=24).quantile(0.05)
    prev_shock = rets.lt(0).mul(rets.le(lower_tail)).astype(float).shift(1)

    market_down_share = down.mean(axis=1)
    dispersion = rets.std(axis=1)
    dispersion_ratio = dispersion.div(
        dispersion.shift(1).rolling(12, min_periods=6).mean()
    ).replace([np.inf, -np.inf], np.nan)

    positions = np.repeat(frame["position"].to_numpy(), len(indicators))
    identifiers = np.tile(np.asarray(indicators, dtype=object), len(frame))
    index = pd.MultiIndex.from_arrays(
        [positions, identifiers], names=["position", "indicator_id"]
    )
    flat = pd.DataFrame(index=index)
    for name, panel in [
        ("ret1_z", ret1_z),
        ("mom3_z", mom3_z),
        ("consec_down", consec_down),
        ("prev_shock", prev_shock),
    ]:
        flat[name] = panel.to_numpy().ravel(order="F")
    flat["market_down_share_prev"] = (
        market_down_share.shift(1).reindex(positions).to_numpy()
    )
    flat["dispersion_ratio_prev"] = (
        dispersion_ratio.shift(1).reindex(positions).to_numpy()
    )
    flat = flat.reset_index().rename(columns={"position": "origin_position"})
    return flat


def extreme_down_labels(shock_frame: pd.DataFrame) -> pd.DataFrame:
    """Tail-shock label from the canonical sudden-drop return series."""
    target_return = pd.to_numeric(shock_frame["target_return"], errors="coerce")
    lower_tail = pd.to_numeric(shock_frame["shock_lower_tail"], errors="coerce")
    label = target_return.lt(0) & target_return.le(lower_tail)
    return pd.DataFrame({
        "origin_position": shock_frame["origin_position"].to_numpy(),
        "indicator_id": shock_frame["indicator_id"].astype(str).to_numpy(),
        "extreme_down_next": label.astype(float).where(
            target_return.notna() & lower_tail.notna()
        ).to_numpy(),
    })


def fit_extreme_down_model(train: pd.DataFrame, settings: dict) -> Pipeline | None:
    data = train.dropna(subset=FEATURE_COLUMNS + ["extreme_down_next"])
    if data["extreme_down_next"].nunique() < 2 or len(data) < int(
        settings.get("minimum_train_rows", 800)
    ):
        return None
    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=float(settings.get("logistic_c", 0.3)),
                max_iter=int(settings.get("logistic_max_iter", 1500)),
                class_weight="balanced",
            ),
        ),
    ])
    model.fit(data[FEATURE_COLUMNS], data["extreme_down_next"])
    return model


def score_extreme_down_walk_forward(
    features: pd.DataFrame,
    settings: dict,
    train_lag: int = 2,
    start_origin: int = 120,
    end_origin: int = 266,
) -> pd.DataFrame:
    """Score every origin training only on origins <= origin - train_lag."""
    rows = []
    origins = range(int(start_origin), int(end_origin) + 1)
    positions = features["origin_position"]
    medians = features[FEATURE_COLUMNS].median()
    for origin in origins:
        train = features[positions <= origin - train_lag]
        model = fit_extreme_down_model(train, settings)
        if model is None:
            continue
        test = features[positions == origin]
        if test.empty:
            continue
        matrix = test[FEATURE_COLUMNS].fillna(medians)
        probability = model.predict_proba(matrix)[:, 1]
        rows.append(pd.DataFrame({
            "origin_position": origin,
            "indicator_id": test["indicator_id"].astype(str).to_numpy(),
            "p_extreme_down": probability,
            "extreme_fit_through_origin": origin - train_lag,
        }))
    return pd.concat(rows, ignore_index=True)


def apply_guarded_down_policy(
    selected: pd.DataFrame,
    scores: pd.DataFrame,
    policy: dict,
) -> pd.DataFrame:
    """Reassign at most max_replacements weak Up calls to Down under gates.

    ``selected`` holds one origin's accepted rows with p_up_base,
    predicted_direction and y_true. ``scores`` maps indicator_id to
    p_extreme_down for the same origin.
    """
    result = selected.copy()
    result["policy_direction"] = result["predicted_direction"].astype(str)
    result["policy_replaced"] = False

    breadth = pd.to_numeric(
        result.get("forecast_market_breadth"), errors="coerce"
    )
    gate_ok = breadth.notna().all() and float(breadth.iloc[0]) < float(
        policy["breadth_gate"]
    )
    if not gate_ok or result.empty:
        return result

    merged = result.merge(
        scores.rename(columns={"origin_position": "_origin"}),
        left_on="indicator_id",
        right_on="indicator_id",
        how="left",
    )
    threshold = merged["p_extreme_down"].quantile(float(policy["risk_quantile"]))
    candidates = merged[
        merged["p_extreme_down"].ge(threshold)
        & merged["p_extreme_down"].notna()
    ]
    weak_up = merged[
        merged["policy_direction"].eq("Up")
        & merged["p_up_base"].lt(float(policy["conviction_ceiling"]))
    ].sort_values(["p_up_base", "p_extreme_down"], ascending=[True, False])
    replacements = min(
        int(policy["max_replacements"]), len(candidates), len(weak_up)
    )
    if replacements <= 0:
        return result
    chosen = weak_up.head(replacements).index
    for row_index in chosen:
        indicator = merged.loc[row_index, "indicator_id"]
        mask = result["indicator_id"] == indicator
        result.loc[mask, "policy_direction"] = "Down"
        result.loc[mask, "policy_replaced"] = True
    return result


def evaluate_window(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    policy: dict,
) -> dict:
    totals = {
        "months": 0,
        "calls": 0,
        "hits_baseline": 0,
        "hits_policy": 0,
        "down_calls": 0,
        "down_hits": 0,
        "replaced_calls": 0,
    }
    for origin, rows in panel.groupby("origin_position"):
        selected = rows[rows["accepted"] == True]  # noqa: E712
        if selected.empty:
            continue
        origin_scores = scores[
            scores["origin_position"] == origin
        ][["indicator_id", "p_extreme_down"]]
        governed = apply_guarded_down_policy(selected, origin_scores, policy)
        base_hits = (
            (governed["y_true"] == 1) & (governed["predicted_direction"] == "Up")
        ) | ((governed["y_true"] == 0) & (governed["predicted_direction"] == "Down"))
        policy_hits = (
            (governed["y_true"] == 1) & (governed["policy_direction"] == "Up")
        ) | ((governed["y_true"] == 0) & (governed["policy_direction"] == "Down"))
        down_mask = governed["policy_direction"] == "Down"
        totals["months"] += 1
        totals["calls"] += int(len(governed))
        totals["hits_baseline"] += int(base_hits.sum())
        totals["hits_policy"] += int(policy_hits.sum())
        totals["down_calls"] += int(down_mask.sum())
        totals["down_hits"] += int((down_mask & (governed["y_true"] == 0)).sum())
        totals["replaced_calls"] += int(governed["policy_replaced"].sum())
    delta = totals["hits_policy"] - totals["hits_baseline"]
    summary = dict(totals)
    summary["delta_hits"] = int(delta)
    summary["accuracy_delta_pp"] = round(
        100.0 * delta / totals["calls"], 4
    ) if totals["calls"] else 0.0
    summary["down_precision"] = round(
        totals["down_hits"] / totals["down_calls"], 4
    ) if totals["down_calls"] else None
    return summary


def select_variant(
    tuning_results: dict[str, dict],
    subwindow_results: dict[str, list[dict]],
) -> tuple[str | None, str]:
    """Predeclared rule: non-negative everywhere, then max tuning delta."""
    eligible = []
    for name, stats in tuning_results.items():
        windows = subwindow_results.get(name, [])
        stable = all(entry["delta_hits"] >= 0 for entry in windows)
        if stats["delta_hits"] >= 0 and stable:
            eligible.append(name)
    if not eligible:
        return None, "no_variant_cleared_predeclared_gates"
    ranked = sorted(
        eligible,
        key=lambda name: (
            -tuning_results[name]["delta_hits"],
            int(tuning_results[name].get("down_calls", 0)),
            float(tuning_results[name].get("risk_quantile", 0.0)),
            int(tuning_results[name].get("max_replacements", 0)),
        ),
    )
    return ranked[0], "selected_on_tuning_with_internal_stability"


def paired_block_bootstrap_delta(
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    policy: dict,
    block_months: int = 6,
    replicates: int = 500,
    seed: int = 20260823,
) -> dict:
    """Bootstrap distribution of monthly mean hit delta on one window."""
    rng = np.random.default_rng(seed)
    monthly = []
    for origin, rows in panel.groupby("origin_position"):
        selected = rows[rows["accepted"] == True]  # noqa: E712
        if selected.empty:
            continue
        origin_scores = scores[
            scores["origin_position"] == origin
        ][["indicator_id", "p_extreme_down"]]
        governed = apply_guarded_down_policy(selected, origin_scores, policy)
        base = (
            ((governed["y_true"] == 1) & (governed["predicted_direction"] == "Up"))
            | ((governed["y_true"] == 0) & (governed["predicted_direction"] == "Down"))
        ).sum()
        after = (
            ((governed["y_true"] == 1) & (governed["policy_direction"] == "Up"))
            | ((governed["y_true"] == 0) & (governed["policy_direction"] == "Down"))
        ).sum()
        monthly.append({
            "origin_position": int(origin),
            "base": int(base),
            "after": int(after),
            "delta": int(after - base),
        })
    frame = pd.DataFrame(monthly).sort_values("origin_position")
    blocks = [
        frame.iloc[start : start + block_months]
        for start in range(0, len(frame), block_months)
    ]
    deltas = []
    for _ in range(replicates):
        sample = rng.choice(len(blocks), size=len(blocks), replace=True)
        pooled = pd.concat([blocks[i] for i in sample])
        calls = pooled["calls" if False else "base"].sum()
        deltas.append(pooled["delta"].sum() / max(calls, 1))
    return {
        "monthly_mean_delta": round(float(frame["delta"].mean()), 5),
        "bootstrap_p10": round(float(np.quantile(deltas, 0.10)), 5),
        "bootstrap_p50": round(float(np.quantile(deltas, 0.50)), 5),
        "bootstrap_p90": round(float(np.quantile(deltas, 0.90)), 5),
        "replicates": replicates,
        "block_months": block_months,
    }
