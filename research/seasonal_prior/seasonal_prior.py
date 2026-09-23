"""Calendar-seasonal prior experiment (pre-registered 2026-09-23).

Question: does an indicator's Up-frequency in the same calendar month carry
directional information beyond its trailing Up-rate?

Why it was never tested: the Structured Logistic's lag-12 features are aligned to
the latest *known* change (t-2 -> t-1), so `direction_lag_12` is the t-14 -> t-13
move. The target is the t -> t+1 move, whose same-calendar-month counterpart one
year earlier is t-12 -> t-11. The model's annual lags sit two months off season.

Causal contract at origin t (the project's 1-based `position`, as set by
forecast_select.io.load_workbook):
  * direction label d_s = 1[X_{s+1} > X_s], usable only for s <= t-2;
  * the scored target is d_t, cross-checked against the production artifact;
  * the workbook is read with maximum_position=267 (267 is the target level of
    origin 266), so the locked evaluation 268-315 is untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from forecast_select.io import load_workbook  # noqa: E402
DATA = ROOT / "data/monthly_indicators.xlsx"
ARTIFACT = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
OUT = ROOT / "research/seasonal_prior/metrics"

MAX_ORIGIN = 266
MAX_LEVEL_POSITION = MAX_ORIGIN + 1
WINDOWS = {"tuning": (120, 179), "validation": (180, 219), "confirmation": (220, 266)}
TUNING_LAST_LABEL = 177  # the newest label usable at origin 179
PRIOR_WINDOW = 48
PRIOR_MIN_LABELS = 24
PRIOR_WEIGHT = 0.5  # production's own indicator_prior_weight
MIN_CELL = 3
SEED = 20260923
BLOCK_MONTHS = 6
REPLICATES = 5000
PERMUTATIONS = 2000
ENSEMBLE_WINDOWS = (36, 48, 60, 96, 0)
PRIMARY = "H1_overlay"
CANDIDATES = (
    "H1_overlay",
    "H1_raw_overlay",
    "H1_prior",
    "prior_only",
    "eb_base",
    "window_ensemble",
)


def load_directions() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Direction labels, target calendar month and year, all indexed by 1-based position."""
    frame = load_workbook(DATA, maximum_position=MAX_LEVEL_POSITION)
    if int(frame["position"].max()) != MAX_LEVEL_POSITION:
        raise ValueError("Workbook is shorter than the non-locked range")
    frame = frame.set_index("position")
    levels = frame[[c for c in frame.columns if str(c).startswith("X")]]
    following = levels.shift(-1)
    directions = (following > levels).astype(float).where(
        following.notna() & levels.notna()
    )
    # The change s -> s+1 lands in the calendar month of position s+1, computed from
    # position s so the date of position 268 is never needed.
    month = frame["Dates"].dt.month
    change_month = month % 12 + 1
    change_year = frame["Dates"].dt.year + (month == 12).astype(int)
    return directions, change_month, change_year


def load_universe(directions: pd.DataFrame) -> pd.DataFrame:
    """Production's own per-origin selection universe, with its calls and scores."""
    columns = [
        "origin_position",
        "indicator_id",
        "level_c_ready",
        "adaptive_data_quality_excluded",
        "accepted",
        "predicted_direction",
        "y_true",
        "selection_score",
        "indicator_prior",
    ]
    frame = pd.read_parquet(ARTIFACT, columns=columns)
    if int(frame["origin_position"].max()) > MAX_ORIGIN:
        raise ValueError("Production artifact contains locked origins")
    for column in ("level_c_ready", "adaptive_data_quality_excluded", "accepted"):
        frame[column] = frame[column].fillna(False).astype(bool)
    in_universe = frame["level_c_ready"] & ~frame["adaptive_data_quality_excluded"]
    if (frame["accepted"] & ~in_universe).any():
        raise ValueError("Production accepted a row outside its own universe")
    universe = frame[in_universe].copy()
    mine = [
        directions.at[int(origin), indicator]
        for origin, indicator in zip(universe["origin_position"], universe["indicator_id"])
    ]
    if not np.array_equal(
        np.asarray(mine, dtype=float), universe["y_true"].to_numpy(dtype=float)
    ):
        raise ValueError("Target alignment disagrees with the production artifact")
    universe["prod_down"] = universe["accepted"] & universe["predicted_direction"].eq("Down")
    return universe


def _cell_counts(labels: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    onehot = (months[:, None] == np.arange(1, 13)[None, :]).astype(float)
    observed = (~np.isnan(labels)).astype(float)
    return onehot.T @ observed, onehot.T @ np.nan_to_num(labels)


def _base_rate(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = (~np.isnan(labels)).sum(0).astype(float)
    base = np.nansum(labels, 0) / np.where(count > 0, count, np.nan)
    return base, count


def _heterogeneity(labels: np.ndarray, months: np.ndarray) -> float:
    n, k = _cell_counts(labels, months)
    base, _ = _base_rate(labels)
    pq = base * (1 - base)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = n * (k / n - base) ** 2 / pq
    valid = (n >= MIN_CELL) & (pq > 0)
    return float(stat[valid].sum())


def _split_half(labels: np.ndarray, months: np.ndarray, years: np.ndarray) -> float:
    halves = []
    for parity in (0, 1):
        rows = years % 2 == parity
        n, k = _cell_counts(labels[rows], months[rows])
        base, _ = _base_rate(labels[rows])
        with np.errstate(invalid="ignore", divide="ignore"):
            halves.append((k / n - base, n))
    (dev_a, n_a), (dev_b, n_b) = halves
    valid = (n_a >= MIN_CELL) & (n_b >= MIN_CELL) & np.isfinite(dev_a) & np.isfinite(dev_b)
    return float(np.corrcoef(dev_a[valid], dev_b[valid])[0, 1])


def _breadth_seasonality(labels: np.ndarray, months: np.ndarray) -> float:
    breadth = np.nanmean(labels, axis=1)
    keep = np.isfinite(breadth)
    breadth, months = breadth[keep], months[keep]
    grand = breadth.mean()
    between = sum(
        (months == m).sum() * (breadth[months == m].mean() - grand) ** 2
        for m in range(1, 13)
        if (months == m).any()
    )
    return float(between / breadth.var())


def existence_tests(
    directions: pd.DataFrame, change_month: pd.Series, change_year: pd.Series
) -> dict:
    """Tuning-era labels only (s <= 177); months are permuted jointly across indicators."""
    labels = directions.loc[:TUNING_LAST_LABEL].to_numpy(dtype=float)
    months = change_month.loc[:TUNING_LAST_LABEL].to_numpy()
    years = change_year.loc[:TUNING_LAST_LABEL].to_numpy()
    keep = np.isfinite(labels).sum(0) >= 60
    labels = labels[:, keep]
    observed = {
        "heterogeneity": _heterogeneity(labels, months),
        "split_half_r": _split_half(labels, months, years),
        "breadth_seasonality": _breadth_seasonality(labels, months),
    }
    rng = np.random.default_rng(SEED)
    null = {name: [] for name in observed}
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(months)
        null["heterogeneity"].append(_heterogeneity(labels, shuffled))
        null["split_half_r"].append(_split_half(labels, shuffled, years))
        null["breadth_seasonality"].append(_breadth_seasonality(labels, shuffled))
    result = {
        "labels_through_position": TUNING_LAST_LABEL,
        "indicators_tested": int(keep.sum()),
        "permutations": PERMUTATIONS,
        "null": (
            "calendar-month labels permuted jointly across indicators, "
            "which keeps same-month cross-sectional dependence"
        ),
    }
    for name, value in observed.items():
        draws = np.asarray(null[name])
        result[name] = {
            "observed": value,
            "null_median": float(np.median(draws)),
            "null_p95": float(np.quantile(draws, 0.95)),
            "p_value": float((1 + (draws >= value).sum()) / (1 + len(draws))),
        }
    result["seasonality_detected_on_tuning"] = result["heterogeneity"]["p_value"] < 0.05
    return result


def label_history(directions: pd.DataFrame, origin: int, window: int = 0) -> pd.DataFrame:
    """Labels usable at `origin`: positions <= origin-2, optionally the newest `window`."""
    newest = origin - 2
    oldest = max(1, newest - window + 1) if window else 1
    return directions.loc[oldest:newest]


def trailing_rate(directions: pd.DataFrame, origin: int, window: int) -> pd.Series:
    history = label_history(directions, origin, window)
    return history.mean().where(history.count() >= PRIOR_MIN_LABELS)


def seasonal_deviation(
    directions: pd.DataFrame, change_month: pd.Series, origin: int
) -> tuple[pd.Series, pd.Series, float]:
    """Empirical-Bayes shrunk deviation of the target month's Up-rate from the all-history rate.

    The seasonal variance tau^2 is a method-of-moments estimate pooled over every
    indicator and calendar month with labels <= origin-2, so nothing is tuned.
    """
    history = label_history(directions, origin)
    labels = history.to_numpy(dtype=float)
    n, k = _cell_counts(labels, change_month.loc[history.index].to_numpy())
    base, count = _base_rate(labels)
    pq = base * (1 - base)
    with np.errstate(invalid="ignore", divide="ignore"):
        deviation = k / n - base[None, :]
        sampling = pq[None, :] * (1.0 / n - 1.0 / count[None, :])
    valid = (
        (n >= MIN_CELL)
        & (count[None, :] >= PRIOR_MIN_LABELS)
        & (pq[None, :] > 0)
        & (sampling > 0)
    )
    tau2 = max(0.0, float(np.mean((deviation**2 - sampling)[valid]))) if valid.any() else 0.0
    cell = int(change_month.loc[origin]) - 1  # calendar month of the change origin -> origin+1
    ok = valid[cell]
    if tau2 > 0:
        weight = np.where(ok, tau2 / (tau2 + np.where(ok, sampling[cell], 1.0)), 0.0)
    else:
        weight = np.zeros(len(base))
    shrunk = np.where(ok, weight * deviation[cell], 0.0)
    raw = np.where(ok, deviation[cell], 0.0)
    return (
        pd.Series(shrunk, index=history.columns),
        pd.Series(raw, index=history.columns),
        tau2,
    )


def shrunk_base(rate: pd.Series, count: pd.Series) -> pd.Series:
    """Empirical-Bayes shrinkage of trailing Up-rates toward the cross-sectional mean."""
    valid = rate.notna() & (count > 0)
    p, n = rate[valid], count[valid]
    mean = float(p.mean())
    sampling = mean * (1 - mean) / n
    tau2 = max(0.0, float(p.var(ddof=1) - sampling.mean()))
    weight = tau2 / (tau2 + sampling) if tau2 > 0 else 0.0
    return (mean + weight * (p - mean)).reindex(rate.index)


def _pick(current: pd.DataFrame, score: pd.Series, k: int) -> pd.DataFrame:
    ranked = current.assign(score=score.reindex(current["indicator_id"]).to_numpy())
    ranked = ranked.dropna(subset=["score"])
    ranked = ranked.sort_values(
        ["score", "indicator_id"], ascending=[False, True], kind="mergesort"
    )
    return ranked.head(k)


def _hits(picks: pd.DataFrame) -> int:
    """Up is scored unless production itself called Down on the same row."""
    up_hit = picks["y_true"].eq(1) & ~picks["prod_down"]
    down_hit = picks["y_true"].eq(0) & picks["prod_down"]
    return int((up_hit | down_hit).sum())


def evaluate(
    directions: pd.DataFrame, change_month: pd.Series, universe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, strength = [], []
    for origin, current in universe.groupby("origin_position", sort=True):
        origin = int(origin)
        current = current.reset_index(drop=True)
        cap = int(current["accepted"].sum())
        by_id = current.set_index("indicator_id")
        delta, raw, tau2 = seasonal_deviation(directions, change_month, origin)
        delta = delta.reindex(by_id.index).fillna(0.0)
        raw = raw.reindex(by_id.index).fillna(0.0)
        prior48 = trailing_rate(directions, origin, PRIOR_WINDOW).reindex(by_id.index)
        count48 = label_history(directions, origin, PRIOR_WINDOW).count().reindex(by_id.index)
        ranks = pd.concat(
            [
                trailing_rate(directions, origin, window).reindex(by_id.index).rank(pct=True)
                for window in ENSEMBLE_WINDOWS
            ],
            axis=1,
        )
        ensemble = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= 3)
        score, prior = by_id["selection_score"], by_id["indicator_prior"]
        scores = {
            "H1_overlay": score + PRIOR_WEIGHT * delta,
            "H1_raw_overlay": score + PRIOR_WEIGHT * raw,
            "H1_prior": prior + delta,
            "prior_only": prior,
            "eb_base": shrunk_base(prior48, count48),
            "window_ensemble": ensemble,
        }
        production = current[current["accepted"]]
        row = {"origin_position": origin, "cap": cap, "production_hits": _hits(production)}
        for name, candidate in scores.items():
            picks = _pick(current, candidate, cap)
            row[f"{name}_calls"] = len(picks)
            row[f"{name}_hits"] = _hits(picks)
            row[f"{name}_overlap"] = int(
                picks["indicator_id"].isin(production["indicator_id"]).sum()
            )
        up_only = current.assign(prod_down=False)
        for name in ("H1_prior", "prior_only"):
            picks = _pick(up_only, scores[name], 15)
            row[f"{name}_fixed15_calls"] = len(picks)
            row[f"{name}_fixed15_hits"] = _hits(picks)
        records.append(row)
        strength.append({
            "origin_position": origin,
            "seasonal_tau2": tau2,
            "seasonal_tau": float(np.sqrt(tau2)),
            "mean_abs_shrunk_deviation": float(delta.abs().mean()),
            "max_abs_shrunk_deviation": float(delta.abs().max()),
            "prior48_max_abs_diff_vs_artifact": float((prior48 - prior).abs().max()),
        })
    return pd.DataFrame(records), pd.DataFrame(strength)


def block_bootstrap(deltas: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(deltas)
    block = min(BLOCK_MONTHS, n)
    starts = rng.integers(0, n - block + 1, size=(REPLICATES, -(-n // block)))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(REPLICATES, -1)
    totals = deltas[index[:, :n]].sum(axis=1)
    return {f"p{q}": float(np.quantile(totals, q / 100)) for q in (10, 50, 90)}


def summarize(records: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for window, (lo, hi) in WINDOWS.items():
        part = records[records["origin_position"].between(lo, hi)]
        prod_hits, calls = int(part["production_hits"].sum()), int(part["cap"].sum())
        rows.append({
            "candidate": "production", "window": window, "hits": prod_hits,
            "calls": calls, "accuracy": prod_hits / calls, "delta_vs_production": 0,
        })
        for name in CANDIDATES:
            deltas = (part[f"{name}_hits"] - part["production_hits"]).to_numpy(dtype=float)
            hits, n_calls = int(part[f"{name}_hits"].sum()), int(part[f"{name}_calls"].sum())
            rows.append({
                "candidate": name, "window": window, "hits": hits, "calls": n_calls,
                "accuracy": hits / n_calls, "delta_vs_production": int(deltas.sum()),
                "overlap_with_production": int(part[f"{name}_overlap"].sum()),
                **{f"bootstrap_{k}": v for k, v in block_bootstrap(deltas, rng).items()},
            })
        for name in ("H1_prior", "prior_only"):
            hits = int(part[f"{name}_fixed15_hits"].sum())
            n_calls = int(part[f"{name}_fixed15_calls"].sum())
            rows.append({
                "candidate": f"{name}_fixed15", "window": window, "hits": hits,
                "calls": n_calls, "accuracy": hits / n_calls,
            })
        deltas = (
            part["H1_prior_fixed15_hits"] - part["prior_only_fixed15_hits"]
        ).to_numpy(dtype=float)
        rows.append({
            "candidate": "seasonal_gain_fixed15 (H1_prior minus prior_only)",
            "window": window, "delta_vs_production": int(deltas.sum()),
            **{f"bootstrap_{k}": v for k, v in block_bootstrap(deltas, rng).items()},
        })
    table = pd.DataFrame(rows)
    primary = table[table["candidate"].eq(PRIMARY)].set_index("window")
    gate = {
        "tuning_delta_nonnegative": bool(primary.at["tuning", "delta_vs_production"] >= 0),
        "validation_delta_positive": bool(primary.at["validation", "delta_vs_production"] > 0),
        "validation_bootstrap_p10_nonnegative": bool(
            primary.at["validation", "bootstrap_p10"] >= 0
        ),
        "confirmation_delta_nonnegative": bool(
            primary.at["confirmation", "delta_vs_production"] >= 0
        ),
        "no_locked_reads": True,
    }
    return table, gate


def main() -> dict:
    directions, change_month, change_year = load_directions()
    universe = load_universe(directions)
    OUT.mkdir(parents=True, exist_ok=True)
    existence = existence_tests(directions, change_month, change_year)
    (OUT / "existence_tests.json").write_text(json.dumps(existence, indent=2), encoding="utf-8")
    records, strength = evaluate(directions, change_month, universe)
    records.to_csv(OUT / "monthly_records.csv", index=False)
    strength.to_csv(OUT / "seasonal_strength.csv", index=False)
    table, gate = summarize(records)
    table.to_csv(OUT / "window_results.csv", index=False)
    summary = {
        "experiment_id": "seasonal_prior",
        "pre_registered": "2026-09-23",
        "primary_candidate": PRIMARY,
        "primary_definition": (
            "production selection_score + 0.5 x EB-shrunk same-calendar-month "
            "Up-rate deviation, at production's per-origin call count"
        ),
        "existence_detected_on_tuning": existence["seasonality_detected_on_tuning"],
        "gate": gate,
        "nonlocked_gate_passed": all(gate.values()),
        "max_origin_read": MAX_ORIGIN,
        "max_level_position_read": MAX_LEVEL_POSITION,
        "locked_evaluation_read": False,
        "production_changed": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"existence": existence, "summary": summary, "table": table}


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["existence"], indent=2))
    print(json.dumps(result["summary"], indent=2))
    pd.set_option("display.width", 220)
    print(result["table"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
