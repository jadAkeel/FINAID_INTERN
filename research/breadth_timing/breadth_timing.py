"""Pre-registered breadth timing study (see README.md).

Can any causal market-level signal predict the months when production's accepted calls mostly
fall? Reads the workbook with maximum_position=267, so the locked evaluation is untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/seasonal_prior"))
from forecast_select.io import load_workbook  # noqa: E402
from seasonal_prior import WINDOWS, load_directions, load_universe  # noqa: E402

DATA = ROOT / "data/monthly_indicators.xlsx"
OUT = ROOT / "research/breadth_timing/metrics"
MAX_LEVEL_POSITION = 267
FIRST_ORIGIN = 40
EXCLUDED = ("X16",)
EQUITY = [f"X{i}" for i in range(1, 16)] + [f"X{i}" for i in range(21, 38)]
FIXED_INCOME = [f"X{i}" for i in range(38, 44)]
SIGNALS = (
    "ma10_breadth", "mom12_breadth", "shock_magnitude", "shock_dispersion",
    "risk_on_spread", "breadth_12", "vol_regime",
)
QUANTILES = (0.10, 0.20, 0.30)
LOGIT_C = 1.0
MIN_TRAIN = 36
SEED = 20260923
BLOCK_MONTHS = 6
REPLICATES = 5000
PERMUTATIONS = 2000
MIN_SHIFT = 6


def load_levels() -> pd.DataFrame:
    frame = load_workbook(DATA, maximum_position=MAX_LEVEL_POSITION).set_index("position")
    columns = [c for c in frame.columns if str(c).startswith("X") and c not in EXCLUDED]
    levels = frame[columns].astype(float)
    if int(levels.index.max()) != MAX_LEVEL_POSITION:
        raise ValueError("Workbook is shorter than the non-locked range")
    return levels


def signals_at_origin(levels: pd.DataFrame) -> pd.DataFrame:
    """Each signal computed at level position p, then indexed by origin t = p + 1."""
    change = np.log(levels).diff()
    scale = change.rolling(36, min_periods=24).std().shift(1)
    z = change / scale
    absz = z.abs()
    ma10 = levels.rolling(10, min_periods=10).mean()
    above = (levels > ma10).where(ma10.notna() & levels.notna())
    rising12 = (levels > levels.shift(12)).where(levels.shift(12).notna() & levels.notna())
    equity = z[[c for c in EQUITY if c in z]].mean(axis=1)
    fixed = z[[c for c in FIXED_INCOME if c in z]].mean(axis=1)
    # direction label of the move p -> p+1 is known at level position p+1
    following = levels.shift(-1)
    labels = (following > levels).astype(float).where(following.notna() & levels.notna())
    known_labels = labels.shift(1)  # at position p: the move p-1 -> p
    label_sum = known_labels.sum(axis=1, min_count=1).rolling(12, min_periods=12).sum()
    label_count = known_labels.count(axis=1).rolling(12, min_periods=12).sum()
    at_level = pd.DataFrame({
        "ma10_breadth": above.mean(axis=1),
        "mom12_breadth": rising12.mean(axis=1),
        "shock_magnitude": absz.mean(axis=1),
        "shock_dispersion": z.std(axis=1),
        "risk_on_spread": (equity - fixed).rolling(3, min_periods=3).mean(),
        "breadth_12": label_sum / label_count.replace(0, np.nan),
        "vol_regime": (
            absz.rolling(3, min_periods=3).mean() - absz.rolling(36, min_periods=24).mean()
        ).mean(axis=1),
    })
    signals = at_level.shift(1)  # origin t uses level positions <= t-1
    signals.index.name = "origin_position"
    return signals


def training_target(levels: pd.DataFrame) -> pd.Series:
    """Breadth of the move s -> s+1 over all available indicators is below one half."""
    following = levels.shift(-1)
    labels = (following > levels).astype(float).where(following.notna() & levels.notna())
    breadth = labels.mean(axis=1)
    return (breadth < 0.5).astype(float).where(breadth.notna())


def combined_logit(signals: pd.DataFrame, target: pd.Series, last_origin: int) -> pd.Series:
    """Walk-forward logistic: at origin t, train on origins s <= t-2."""
    predictions = {}
    for origin in range(FIRST_ORIGIN, last_origin + 1):
        train = signals.loc[FIRST_ORIGIN: origin - 2].join(
            target.rename("y").loc[FIRST_ORIGIN: origin - 2]
        ).dropna()
        row = signals.loc[[origin]]
        if len(train) < MIN_TRAIN or row.isna().any(axis=None) or train["y"].nunique() < 2:
            continue
        x = train[list(SIGNALS)]
        mean, std = x.mean(), x.std().replace(0, 1.0)
        model = LogisticRegression(C=LOGIT_C, max_iter=1000)
        model.fit((x - mean) / std, train["y"])
        predictions[origin] = float(model.predict_proba((row[list(SIGNALS)] - mean) / std)[0, 1])
    return pd.Series(predictions, name="combined_logit")


def monthly_outcomes(universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for origin, current in universe.groupby("origin_position", sort=True):
        picks = current[current["accepted"]]
        prod_hits = int(
            (picks["y_true"].eq(1) & ~picks["prod_down"]).sum()
            + (picks["y_true"].eq(0) & picks["prod_down"]).sum()
        )
        rows.append({
            "origin_position": int(origin),
            "calls": len(picks),
            "production_hits": prod_hits,
            "flipped_hits": int(picks["y_true"].eq(0).sum()),
            "flip_target": int(picks["y_true"].mean() < 0.5),
            "universe_broad_down": int(current["y_true"].mean() < 0.5),
        })
    frame = pd.DataFrame(rows).set_index("origin_position")
    frame["window"] = [
        next(name for name, (lo, hi) in WINDOWS.items() if lo <= o <= hi) for o in frame.index
    ]
    return frame


def _auc(score: pd.Series, target: pd.Series) -> float:
    joined = pd.concat([score, target], axis=1).dropna()
    if joined.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(joined.iloc[:, 1], joined.iloc[:, 0]))


def expanding_threshold(score: pd.Series, q: float) -> pd.Series:
    """Causal (1 - q) quantile of the score's own values at origins FIRST_ORIGIN..t-1."""
    history = score.loc[FIRST_ORIGIN:]
    return history.expanding(min_periods=24).quantile(1 - q).shift(1)


def flip_gain(score: pd.Series, q: float, months: pd.DataFrame) -> pd.Series:
    threshold = expanding_threshold(score, q).reindex(months.index)
    flip = score.reindex(months.index) >= threshold
    gain = (months["flipped_hits"] - months["production_hits"]).where(flip, 0)
    return gain.rename("gain"), flip.rename("flip")


def block_bootstrap_p05(gains: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(gains)
    starts = rng.integers(0, n - BLOCK_MONTHS + 1, size=(REPLICATES, -(-n // BLOCK_MONTHS)))
    index = (starts[:, :, None] + np.arange(BLOCK_MONTHS)).reshape(REPLICATES, -1)[:, :n]
    totals = gains[index].sum(1)
    return {"p05": float(np.quantile(totals, 0.05)), "p50": float(np.quantile(totals, 0.5)),
            "p95": float(np.quantile(totals, 0.95))}


def circular_permutation_p(score: pd.Series, target: pd.Series) -> float:
    joined = pd.concat([score, target], axis=1).dropna()
    s, y = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    observed = roc_auc_score(y, s)
    rng = np.random.default_rng(SEED)
    shifts = rng.integers(MIN_SHIFT, len(s) - MIN_SHIFT + 1, size=PERMUTATIONS)
    null = np.array([roc_auc_score(y, np.roll(s, k)) for k in shifts])
    return float((1 + (null >= observed).sum()) / (1 + PERMUTATIONS))


def evaluate_finalist(name: str, score: pd.Series, months: pd.DataFrame) -> dict:
    tuning = months[months["window"].eq("tuning")]
    tuning_gain = {q: int(flip_gain(score, q, tuning)[0].sum()) for q in QUANTILES}
    q = max(QUANTILES, key=lambda value: (tuning_gain[value], -value))
    gain, flip = flip_gain(score, q, months)
    oos = months.index[months["window"].isin(["validation", "confirmation"])]
    result = {"finalist": name, "tuning_gain_by_q": tuning_gain, "chosen_q": q, "windows": {}}
    for window in [*WINDOWS, "validation_confirmation"]:
        part = oos if window == "validation_confirmation" else months.index[months["window"].eq(window)]
        flipped = flip.loc[part]
        result["windows"][window] = {
            "months": len(part),
            "flip_target_auc": _auc(score.reindex(part), months.loc[part, "flip_target"]),
            "universe_breadth_auc": _auc(score.reindex(part), months.loc[part, "universe_broad_down"]),
            "flipped_months": int(flipped.sum()),
            "flip_precision": float(months.loc[part, "flip_target"][flipped].mean())
            if flipped.any() else None,
            "hit_gain": int(gain.loc[part].sum()),
            "production_hits": int(months.loc[part, "production_hits"].sum()),
            "calls": int(months.loc[part, "calls"].sum()),
        }
    windows = result["windows"]
    pooled = windows["validation_confirmation"]
    result["permutation_p_pooled_vc"] = circular_permutation_p(
        score.reindex(oos), months.loc[oos, "flip_target"]
    )
    result["bootstrap_pooled_vc_gain"] = block_bootstrap_p05(gain.loc[oos].to_numpy(float))
    result["gate"] = {
        "G1_auc_pooled_ge_0.60": pooled["flip_target_auc"] >= 0.60,
        "G1_auc_each_ge_0.55": all(
            windows[w]["flip_target_auc"] >= 0.55 for w in ("validation", "confirmation")
        ),
        "G1_permutation_p_lt_0.05": result["permutation_p_pooled_vc"] < 0.05,
        "G2_gain_both_windows": all(
            windows[w]["hit_gain"] > 0 for w in ("validation", "confirmation")
        ),
        "G2_bootstrap_p05_gt_0": result["bootstrap_pooled_vc_gain"]["p05"] > 0,
    }
    result["gate"] = {k: bool(v) for k, v in result["gate"].items()}
    result["passed"] = all(result["gate"].values())
    return result


def main() -> dict:
    levels = load_levels()
    directions, _, _ = load_directions()
    universe = load_universe(directions)
    months = monthly_outcomes(universe)
    signals = signals_at_origin(levels)
    if signals.index.max() > MAX_LEVEL_POSITION + 1 or months.index.max() > 266:
        raise ValueError("Research range exceeded")
    scores = signals.copy()
    scores["combined_logit"] = combined_logit(signals, training_target(levels), 266)

    diagnostics = []
    for name in [*SIGNALS, "combined_logit"]:
        for window in [*WINDOWS, "validation_confirmation"]:
            part = months.index[
                months["window"].isin(["validation", "confirmation"])
                if window == "validation_confirmation" else months["window"].eq(window)
            ]
            diagnostics.append({
                "signal": name, "window": window,
                "flip_target_auc": _auc(scores[name].reindex(part), months.loc[part, "flip_target"]),
                "universe_breadth_auc": _auc(
                    scores[name].reindex(part), months.loc[part, "universe_broad_down"]
                ),
            })
    diagnostics = pd.DataFrame(diagnostics)

    tuning = diagnostics[diagnostics["window"].eq("tuning") & diagnostics["signal"].isin(SIGNALS)]
    best = tuning.loc[(tuning["flip_target_auc"] - 0.5).abs().idxmax()]
    sign = 1.0 if best["flip_target_auc"] >= 0.5 else -1.0
    s_star = (sign * scores[best["signal"]]).rename(best["signal"])

    finalists = [
        evaluate_finalist(f"S*={best['signal']} (sign {sign:+.0f})", s_star, months),
        evaluate_finalist("C1=combined_logit", scores["combined_logit"], months),
    ]
    # Diagnostic only: every single signal with its Tuning-fixed sign and q.
    all_single = []
    for name in SIGNALS:
        tuning_auc = float(tuning.set_index("signal").at[name, "flip_target_auc"])
        oriented = (1.0 if tuning_auc >= 0.5 else -1.0) * scores[name]
        record = evaluate_finalist(name, oriented, months)
        all_single.append({
            "signal": name, "tuning_auc": tuning_auc, "chosen_q": record["chosen_q"],
            **{f"{w}_gain": record["windows"][w]["hit_gain"]
               for w in ("tuning", "validation", "confirmation")},
            "vc_oriented_auc": record["windows"]["validation_confirmation"]["flip_target_auc"],
        })

    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUT / "signal_auc.csv", index=False)
    pd.DataFrame(all_single).to_csv(OUT / "single_signal_flip_diagnostic.csv", index=False)
    scores.join(months, how="right").to_csv(OUT / "monthly_scores.csv")
    summary = {
        "study": "breadth_timing",
        "preregistered": True,
        "max_level_position_read": MAX_LEVEL_POSITION,
        "locked_evaluation_read": False,
        "flip_target_base_rate": {
            w: float(months.loc[months["window"].eq(w), "flip_target"].mean()) for w in WINDOWS
        },
        "selected_single_signal": best["signal"],
        "selected_sign": sign,
        "finalists": finalists,
        "any_passed": any(f["passed"] for f in finalists),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = main()
    pd.set_option("display.width", 200)
    print(pd.read_csv(OUT / "signal_auc.csv").pivot(
        index="signal", columns="window", values="flip_target_auc").round(3))
    print(pd.read_csv(OUT / "single_signal_flip_diagnostic.csv").round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != "finalists"}, indent=2))
    for finalist in result["finalists"]:
        print(json.dumps(finalist, indent=2, default=float))
