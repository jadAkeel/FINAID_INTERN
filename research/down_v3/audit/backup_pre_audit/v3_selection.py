"""Down V3 selection engine: replacement delta + dynamic 0-5 Down policy.

The primary metric of the new brief. A Down candidate never exists in
isolation: it takes the slot of an Up candidate. So the measure that maps to
the product goal is the *replacement delta*:

    +1  Down candidate correct, Up victim wrong
    -1  Down candidate wrong,   Up victim correct
     0  both right or both wrong

Two independent Down paths, per the brief:

  Path B (overlay)   - an indicator already inside the Up pool is re-examined
                       with Down evidence; if Down is reliably stronger the
                       direction flips. Call count is unchanged, so its delta
                       is (1 - y) - y on the same row.
  Path A (external)  - the Down model searches ALL eligible indicators,
                       including those outside the Up pool. A candidate enters
                       only if down_expected >= up_victim_expected + margin,
                       displacing the weakest remaining Up call.

Combined cap: total Down calls <= MAX_DOWN per origin, from either path.
All margins/floors screened on Tuning only, frozen before Validation.

Writes: research/down_v3/metrics/v3_selection.json
         research/down_v3/artifacts/v3_selections_<variant>.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from json_output import strict_json_dumps  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))

ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
V2PRED = ROOT / "research/down_v2/artifacts/down_v2_predictions.parquet"
V3PRED = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
OUT_MET = ROOT / "research/down_v3/metrics/v3_selection.json"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}
MAX_DOWN = 5
RNG = np.random.default_rng(20260921)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(tune: pd.DataFrame, score: str, ycol: str) -> LogisticRegression:
    platt = LogisticRegression(C=1e6, max_iter=1000)
    platt.fit(logit(tune[score].to_numpy()).reshape(-1, 1), tune[ycol].astype(int))
    return platt


def apply_platt(platt: LogisticRegression, s) -> np.ndarray:
    return np.clip(platt.predict_proba(logit(s).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def month_policy(
    g: pd.DataFrame, cap: int, margin: float, floor: float, max_down: int,
    down_col: str = "p_down_v3_cal",
) -> pd.DataFrame:
    g = g.copy()
    g["p_down_cal"] = g[down_col]
    ready = g[g["level_c_ready"].fillna(False).astype(bool)].copy()
    ranked = ready.sort_values(
        ["p_up_selection_score", "indicator_id"], ascending=[False, True]
    )
    base = ranked.head(cap).copy()
    base["source"] = "up"
    base["is_down_call"] = base["predicted_direction"].fillna("Up").eq("Down")
    base["victim_id"] = ""
    base["victim_y"] = np.nan

    outside = ranked.iloc[cap:].copy()

    # ---- Path B: overlay correction (no call-count change) ----
    flips = 0
    for idx, row in base.iterrows():
        if bool(row["is_down_call"]):
            continue
        if row["p_down_cal"] >= row["p_up_cal"] + margin and row["p_down_cal"] >= floor:
            base.loc[idx, "is_down_call"] = True
            base.loc[idx, "source"] = "overlay_flip"
            # the victim is the same row's former Up call
            base.loc[idx, "victim_id"] = row["indicator_id"]
            base.loc[idx, "victim_y"] = row["y_true"]
            flips += 1
    budget = max_down - flips

    # ---- Path A: external insertion, displacing the weakest Up call ----
    cand = outside[
        outside["p_down_cal"].ge(floor)
        & outside["p_down_cal"].ge(outside["p_up_cal"] + margin)
    ].sort_values("p_down_cal", ascending=False)
    n_insert = 0
    for _, crow in cand.iterrows():
        if n_insert >= budget:
            break
        # weakest remaining Up call is the victim
        remaining = base[~base["is_down_call"]]
        if remaining.empty:
            break
        vid = remaining["p_up_cal"].idxmin()
        if crow["p_down_cal"] < base.loc[vid, "p_up_cal"] + margin:
            break
        # the displaced Up call's outcome is carried by the inserted row so
        # the replacement delta survives the drop of the victim row
        victim_y = float(base.loc[vid, "y_true"])
        victim_id = str(base.loc[vid, "indicator_id"])
        base = base.drop(index=vid)
        new = pd.DataFrame(
            {
                "origin_position": [crow["origin_position"]],
                "indicator_id": [crow["indicator_id"]],
                "y_true": [crow["y_true"]],
                "p_up_selection_score": [crow["p_up_selection_score"]],
                "p_up_cal": [crow["p_up_cal"]],
                "p_down_cal": [crow["p_down_cal"]],
                "predicted_direction": ["Down"],
                "down_correct": [1.0 - crow["y_true"]],
                "is_down_call": [True],
                "source": ["external_insert"],
                "victim_id": [victim_id],
                "victim_y": [victim_y],
            },
            index=[f"ins_{crow['indicator_id']}_{n_insert}"],
        )
        base = pd.concat([base, new], ignore_index=False)
        n_insert += 1

    base["flips"] = flips
    base["inserts"] = n_insert
    return base


def month_down_count(g: pd.DataFrame, cap: int, margin: float, floor: float) -> int:
    """Cheap dry-run of the policy to count Down calls (for the Tuning screen)."""
    sel = month_policy(g, cap, margin, floor, MAX_DOWN, down_col="p_down_v3_cal")
    return int(sel["is_down_call"].sum())


def main() -> None:
    art = pd.read_parquet(ACTIVE)
    v3 = pd.read_parquet(V3PRED)[
        ["origin_position", "indicator_id", "logistic_platt", "logistic"]
    ]
    v2 = pd.read_parquet(V2PRED)[["origin_position", "indicator_id", "p_platt"]].rename(
        columns={"p_platt": "p_down_v2"}
    )
    j = art.merge(v3, on=["origin_position", "indicator_id"], how="left")
    j = j.merge(v2, on=["origin_position", "indicator_id"], how="left")
    j["p_down_v2"] = j["p_down_v2"].fillna(j["p_down"].fillna(0.0))
    j["down_correct"] = 1.0 - j["y_true"]

    # ---- common calibrated correctness scale (fit on Tuning only) ----
    # NOTE: `tune` must be re-sliced after each new column is added to `j`,
    # otherwise the view does not see the new column.
    j["p_up_cal"] = apply_platt(
        fit_platt(j[j["origin_position"].between(120, 179)], "p_up_selection_score", "y_true"),
        j["p_up_selection_score"].to_numpy(),
    )
    j["p_down_v3_cal"] = apply_platt(
        fit_platt(
            j[j["origin_position"].between(120, 179)], "logistic_platt", "down_correct"
        ),
        j["logistic_platt"].to_numpy(),
    )
    j["p_down_v2_cal"] = apply_platt(
        fit_platt(
            j[j["origin_position"].between(120, 179)], "p_down_v2", "down_correct"
        ),
        j["p_down_v2"].to_numpy(),
    )
    tune = j[j["origin_position"].between(120, 179)]
    print(
        f"common scale (Tuning): up AUC={roc_auc_score(tune['y_true'], tune['p_up_cal']):.4f} "
        f"downV3 AUC={roc_auc_score(tune['down_correct'], tune['p_down_v3_cal']):.4f} "
        f"downV2 AUC={roc_auc_score(tune['down_correct'], tune['p_down_v2_cal']):.4f}"
    )

    results: dict[str, object] = {"max_down": MAX_DOWN}
    best_params: dict[str, dict] = {}
    tune_j = j[j["origin_position"].between(120, 179)]
    groups = {o: g for o, g in tune_j.groupby("origin_position")}

    for ver, col in (("v2", "p_down_v2_cal"), ("v3", "p_down_v3_cal")):
        # Screen on Tuning delta. Ties are broken toward the STRICTEST setting
        # that still ties: a looser margin admits more calls for the same
        # (zero) evidence of gain, which is not a reason to prefer it.
        best = None
        for margin in (0.08, 0.05, 0.03, 0.02, 0.0):
            for floor in (0.65, 0.60, 0.55, 0.50):
                sel_all = []
                for origin, g in groups.items():
                    cap = int(g["regime_cap"].iloc[0])
                    sel_all.append(
                        month_policy(g, cap, margin, floor, MAX_DOWN, col)
                    )
                s = pd.concat(sel_all)
                d = s[s["is_down_call"]]
                if len(d) == 0:
                    continue
                delta = int(
                    d["down_correct"].astype(float).sub(d["victim_y"].astype(float)).sum()
                )
                print(
                    f"  {ver} margin={margin:.2f} floor={floor:.2f}: "
                    f"calls={len(d):4d} delta={delta:+4d} mean={delta/len(d):+.4f}"
                )
                key = (delta, -len(d), margin, floor)
                if best is None or key > best[0]:
                    best = (key, margin, floor, len(d), delta)
        if best is None:
            best = (None, 0.05, 0.55, 0, 0)
        best_params[ver] = {
            "margin": best[1],
            "floor": best[2],
            "tuning_delta": best[4],
            "tuning_calls": best[3],
        }
        print(
            f"selected {ver}: margin={best[1]:.2f} floor={best[2]:.2f} "
            f"(tuning delta={best[4]:+d} on {best[3]} calls)"
        )
    results["screen"] = best_params

    # ---- run the frozen policy on every window ----
    all_groups = {o: g for o, g in j.groupby("origin_position")}
    for ver, col in (("v2", "p_down_v2_cal"), ("v3", "p_down_v3_cal")):
        margin = best_params[ver]["margin"]
        floor = best_params[ver]["floor"]
        sels = []
        for origin, g in all_groups.items():
            cap = int(g["regime_cap"].iloc[0])
            sels.append(month_policy(g, cap, margin, floor, MAX_DOWN, col))
        sel = pd.concat(sels)
        sel["variant"] = ver
        sel.to_parquet(
            ROOT / f"research/down_v3/artifacts/v3_selections_{ver}.parquet", index=False
        )

        per = {}
        for wname, (a, b) in WINDOWS.items():
            w = sel[sel["origin_position"].between(a, b)]
            d = w[w["is_down_call"]]
            rdelta = int(d["down_correct"].astype(float).sub(d["victim_y"].astype(float)).sum())
            per[wname] = {
                "calls": int(len(w)),
                "hits": int(w["y_true"].sum()),
                "hit_rate": float(w["y_true"].mean()),
                "down_calls": int(len(d)),
                "down_hits": int(d["down_correct"].sum()),
                "down_precision": float(d["down_correct"].mean()) if len(d) else float("nan"),
                "overlay_flips": int((d["source"] == "overlay_flip").sum()),
                "external_inserts": int((d["source"] == "external_insert").sum()),
                "replacement_delta": rdelta,
                "delta_mean": rdelta / len(d) if len(d) else float("nan"),
                "monthly_down_dist": {
                    str(k): int(v)
                    for k, v in w.groupby("origin_position")["is_down_call"]
                    .sum()
                    .value_counts()
                    .sort_index()
                    .items()
                },
            }
        a_hits = j[j["accepted"].fillna(False).astype(bool)].groupby("origin_position")[
            "y_true"
        ].sum()
        b_hits = sel.groupby("origin_position")["y_true"].sum()
        idx = a_hits.index.union(b_hits.index)
        xa = a_hits.reindex(idx).fillna(0.0).to_numpy()
        xb = b_hits.reindex(idx).fillna(0.0).to_numpy()
        n = len(idx)
        deltas = np.array(
            [
                xb[pick].mean() - xa[pick].mean()
                for pick in (RNG.integers(0, n, size=n) for _ in range(2000))
            ]
        )
        results[ver] = {
            "per_window": per,
            "bootstrap_vs_A": {
                "delta_mean": float((xb - xa).mean()),
                "p10": float(np.quantile(deltas, 0.10)),
                "p90": float(np.quantile(deltas, 0.90)),
                "p_delta_positive": float((deltas > 0).mean()),
            },
        }

    OUT_MET.parent.mkdir(parents=True, exist_ok=True)
    OUT_MET.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Down selection policy (overlay + external insertion, cap 5)")
    print("=" * 78)
    for ver in ("v2", "v3"):
        r = results[ver]
        print(f"\n--- {ver} ---")
        for wname, d in r["per_window"].items():
            print(
                f"  {wname:12s} calls={d['calls']:4d} hits={d['hits']:4d} "
                f"rate={d['hit_rate']:.4f} down={d['down_calls']:3d} "
                f"prec={d['down_precision']:.4f} "
                f"(flips={d['overlay_flips']} inserts={d['external_inserts']}) "
                f"delta={d['replacement_delta']:+d}"
            )
        b = r["bootstrap_vs_A"]
        print(
            f"  bootstrap vs A: delta={b['delta_mean']:+.4f} p10={b['p10']:+.4f} "
            f"p90={b['p90']:+.4f} P(>0)={b['p_delta_positive']:.3f}"
        )
    print(f"\nsaved {OUT_MET}")


if __name__ == "__main__":
    main()
