"""Architecture comparison A / B / C / D using Down V2 against the frozen baseline.

A - current production: the frozen active artifact's `accepted` set.
B - limited independent Down insertion: keep A, insert a small number of
    out-of-pool indicators as Down calls, displacing the weakest Up calls.
C - independent pools: the monthly cap is split between an Up pool and a Down
    pool, with the Down count driven by how many clear a Tuning-screened floor.
D - unified bidirectional: one ranking on a common calibrated correctness score.

Every free parameter is screened on Tuning 120-179 only. Validation 180-219 is
a gate, Confirmation 220-266 descriptive. Locked origins 268-315 are never read.
No production code is touched; this reads the frozen artifacts only.
"""
from __future__ import annotations

from json_output import strict_json_dumps
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
V2 = ROOT / "research/down_v2/artifacts/down_v2_predictions.parquet"
OUT_METRICS = ROOT / "research/down_v2/metrics/architecture_comparison.json"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}
RNG = np.random.default_rng(20260727)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def month_summary(sel: pd.DataFrame, cap_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-origin hits, calls and direction mix for a candidate selection."""
    g = sel.groupby("origin_position").agg(
        calls=("y_true", "size"),
        hits=("y_true", "sum"),
        up_calls=("is_down_call", lambda s: int((~s).sum())),
        down_calls=("is_down_call", "sum"),
        down_hits=("down_correct", "sum"),
    )
    g["up_hits"] = g["hits"] - g["down_hits"]
    return g


def score_selection(
    name: str, sel: pd.DataFrame, months: int
) -> dict[str, float | int]:
    if sel.empty:
        return {
            "calls": 0,
            "hits": 0,
            "hit_rate": float("nan"),
            "down_calls": 0,
            "down_hits": 0,
            "down_precision": float("nan"),
            "months": 0,
        }
    y = sel["y_true"].to_numpy()
    is_down = sel["is_down_call"].to_numpy()
    return {
        "calls": int(len(sel)),
        "hits": int(y.sum()),
        "hit_rate": float(y.mean()),
        "down_calls": int(is_down.sum()),
        "down_hits": int(sel.loc[is_down, "down_correct"].sum()),
        "down_precision": float(sel.loc[is_down, "down_correct"].mean())
        if is_down.any()
        else float("nan"),
        "months": int(sel["origin_position"].nunique()),
    }


def per_window(sel: pd.DataFrame) -> dict[str, dict]:
    return {
        wname: score_selection(wname, sel[sel["origin_position"].between(a, b)], 0)
        for wname, (a, b) in WINDOWS.items()
    }


def bootstrap_delta(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000
) -> dict[str, float]:
    """Origin-level bootstrap of the hit-rate delta (B/C/D minus A)."""
    ha = a.groupby("origin_position")["y_true"].mean()
    hb = b.groupby("origin_position")["y_true"].mean()
    idx = ha.index.union(hb.index)
    xa = ha.reindex(idx).fillna(0.0).to_numpy()
    xb = hb.reindex(idx).fillna(0.0).to_numpy()
    n = len(idx)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pick = RNG.integers(0, n, size=n)
        deltas[i] = xb[pick].mean() - xa[pick].mean()
    return {
        "delta_mean": float((xb - xa).mean()),
        "p10": float(np.quantile(deltas, 0.10)),
        "p90": float(np.quantile(deltas, 0.90)),
        "p_delta_positive": float((deltas > 0).mean()),
    }


def select_up_pool(rows: pd.DataFrame, cap: int) -> pd.DataFrame:
    """The production Up-first base pool: top `cap` by p_up_selection_score."""
    return rows.sort_values(
        ["p_up_selection_score", "indicator_id"], ascending=[False, True]
    ).head(cap)


def build_designs(j: pd.DataFrame, thr: float, insert_cap: int) -> dict[str, pd.DataFrame]:
    """Materialise A/B/C/D selections for one (threshold, insert_cap) setting."""
    out: dict[str, pd.DataFrame] = {}

    for origin, g in j.groupby("origin_position", sort=True):
        cap = int(g["regime_cap"].iloc[0])
        ready = g[g["level_c_ready"].fillna(False).astype(bool)]
        prod = g[g["accepted"].fillna(False).astype(bool)].copy()
        prod["is_down_call"] = prod["predicted_direction"].fillna("Up").eq("Down")
        prod["down_correct"] = 1.0 - prod["y_true"]
        out.setdefault("A", []).append(prod)

        base = select_up_pool(ready, cap).copy()
        base["is_down_call"] = False
        base["down_correct"] = 1.0 - base["y_true"]
        pool_ids = set(base["indicator_id"])
        outside = ready[~ready["indicator_id"].isin(pool_ids)].copy()
        outside = outside.sort_values(
            ["p_down_v2", "indicator_id"], ascending=[False, True]
        )
        elig = outside[outside["p_down_v2"].ge(thr)]

        # --- B: limited insertion into the production selection ---
        keep = prod.sort_values(
            ["p_up_selection_score", "indicator_id"], ascending=[False, True]
        )
        bsel = keep.iloc[: max(0, cap - min(insert_cap, len(elig)))].copy()
        bsel["is_down_call"] = bsel["predicted_direction"].fillna("Up").eq("Down")
        bsel["down_correct"] = 1.0 - bsel["y_true"]
        for _, r in elig.head(insert_cap).iterrows():
            row = pd.DataFrame(
                {
                    "origin_position": [origin],
                    "indicator_id": [r["indicator_id"]],
                    "y_true": [r["y_true"]],
                    "p_up_selection_score": [r["p_up_selection_score"]],
                    "p_down_v2": [r["p_down_v2"]],
                    "predicted_direction": ["Down"],
                }
            )
            row["is_down_call"] = True
            row["down_correct"] = 1.0 - row["y_true"]
            bsel = pd.concat([bsel, row], ignore_index=True)
        out.setdefault("B", []).append(bsel)

        # --- C: independent pools, Down count driven by confidence ---
        n_down = int(min(len(elig), max(0, cap // 2)))
        n_up = cap - n_down
        cup = select_up_pool(ready, n_up).copy()
        cup["is_down_call"] = False
        cup["down_correct"] = 1.0 - cup["y_true"]
        cdown = elig.head(n_down).copy()
        if len(cdown):
            cdown = cdown[
                ["origin_position", "indicator_id", "y_true", "p_up_selection_score", "p_down_v2"]
            ].copy()
            cdown["predicted_direction"] = "Down"
            cdown["is_down_call"] = True
            cdown["down_correct"] = 1.0 - cdown["y_true"]
            out.setdefault("C", []).append(pd.concat([cup, cdown], ignore_index=True))
        else:
            out.setdefault("C", []).append(cup)

        # --- D: unified bidirectional on a common calibrated correctness score ---
        g = g.copy()
        g["is_down_call"] = g["unified_direction"].eq("Down")
        g["down_correct"] = 1.0 - g["y_true"]
        dsel = g.sort_values(
            ["unified_score", "indicator_id"], ascending=[False, True]
        ).head(cap)
        out.setdefault("D", []).append(
            dsel[
                [
                    "origin_position",
                    "indicator_id",
                    "y_true",
                    "p_up_selection_score",
                    "p_down_v2",
                    "predicted_direction",
                    "is_down_call",
                    "down_correct",
                ]
            ]
        )

    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def main() -> None:
    started = time.perf_counter()
    art = pd.read_parquet(ACTIVE)
    v2 = pd.read_parquet(V2)[
        ["origin_position", "indicator_id", "p_platt", "p_isotonic", "p_raw"]
    ].rename(columns={"p_platt": "p_down_v2"})
    j = art.merge(v2, on=["origin_position", "indicator_id"], how="left")
    j["p_down_v2"] = j["p_down_v2"].fillna(j["p_down"].fillna(0.0))
    j["down_correct"] = 1.0 - j["y_true"]

    # --- common correctness scale for D: per-direction calibration on Tuning ---
    tune = j[j["origin_position"].between(120, 179)]
    up_platt = LogisticRegression(C=1e6, max_iter=1000)
    up_platt.fit(
        logit(tune["p_up_selection_score"]).reshape(-1, 1), tune["y_true"].astype(int)
    )
    j["p_up_cal"] = np.clip(
        up_platt.predict_proba(logit(j["p_up_selection_score"]).reshape(-1, 1))[:, 1],
        1e-6,
        1 - 1e-6,
    )
    j["unified_score"] = np.maximum(j["p_up_cal"], j["p_down_v2"])
    j["unified_direction"] = np.where(
        j["p_down_v2"] > j["p_up_cal"], "Down", "Up"
    )
    print(
        f"Up-calibrator on Tuning: AUC="
        f"{roc_auc_score(tune['y_true'], j.loc[tune.index, 'p_up_cal']):.4f} "
        f"Brier={brier_score_loss(tune['y_true'], j.loc[tune.index, 'p_up_cal']):.4f}"
    )
    iso_up = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_up.fit(tune["p_up_selection_score"], tune["y_true"].astype(int))

    # --- Tuning screen for the Down floor (minimum 20 calls, as in V2) ---
    tune_ready = tune[tune["level_c_ready"].fillna(False).astype(bool)]
    print("\nFloor screen on Tuning (out-of-pool rows by V2 calibrated p_down):")
    prec_by_thr = {}
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60):
        rows = []
        for _, g in tune_ready.groupby("origin_position"):
            cap = int(g["regime_cap"].iloc[0])
            pool = set(select_up_pool(g, cap)["indicator_id"])
            e = g[~g["indicator_id"].isin(pool) & g["p_down_v2"].ge(thr)]
            rows.append(e)
        if rows:
            r = pd.concat(rows)
            prec_by_thr[thr] = (len(r), float(r["down_correct"].mean()) if len(r) else float("nan"))
    for thr, (n, p) in prec_by_thr.items():
        print(f"  floor={thr:.2f} -> n={n:4d} down_precision={p:.4f}")
    eligible = {t for t, (n, _) in prec_by_thr.items() if n >= 20}
    best_thr = max(eligible, key=lambda t: prec_by_thr[t][1]) if eligible else 0.45
    print(f"selected floor={best_thr}")

    # --- Tuning screen for the B insertion cap ---
    designs = build_designs(j, best_thr, 1)
    b_tune = designs["B"][
        designs["B"]["origin_position"].between(120, 179)
    ]
    best_cap, best_hits = 1, int(b_tune["y_true"].sum())
    for cap_i in (1, 2, 3, 4):
        d = build_designs(j, best_thr, cap_i)["B"]
        d = d[d["origin_position"].between(120, 179)]
        h = int(d["y_true"].sum())
        print(f"  insert_cap={cap_i}: tuning hits={h}")
        if h > best_hits:
            best_cap, best_hits = cap_i, h
    print(f"selected insert_cap={best_cap}")

    final = build_designs(j, best_thr, best_cap)
    results: dict[str, object] = {
        "screen": {
            "down_floor": best_thr,
            "insert_cap": best_cap,
            "tuning_floor_precision": prec_by_thr.get(best_thr),
        },
    }
    a = final["A"]
    for name, sel in final.items():
        results[name] = {
            "per_window": per_window(sel),
            "bootstrap_vs_A": bootstrap_delta(a, sel)
            if name != "A"
            else None,
        }
        # monthly-cap sanity: total calls per origin must match A
        ca = a.groupby("origin_position")["y_true"].size()
        cb = sel.groupby("origin_position")["y_true"].size()
        diff = (ca - cb).abs().sum()
        print(f"{name}: call-count deviation vs A = {int(diff)}")

    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Architecture comparison (screened on Tuning, Validation is a gate)")
    print("=" * 78)
    for name in ("A", "B", "C", "D"):
        r = results[name]
        print(f"\n--- {name} ---")
        for wname, wm in r["per_window"].items():
            print(
                f"  {wname:12s} calls={wm['calls']:4d} hits={wm['hits']:4d} "
                f"hit_rate={wm['hit_rate']:.4f}  down={wm['down_calls']:3d}"
                f"/{wm['down_calls'] + (wm['calls'] - wm['down_calls']):3d}"
                f"  down_prec={wm['down_precision']:.4f}"
            )
        if r["bootstrap_vs_A"]:
            b = r["bootstrap_vs_A"]
            print(
                f"  bootstrap vs A (validation+): delta={b['delta_mean']:+.4f} "
                f"p10={b['p10']:+.4f} p90={b['p90']:+.4f} "
                f"P(delta>0)={b['p_delta_positive']:.3f}"
            )
    print(f"\nsaved {OUT_METRICS} ({time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    main()
