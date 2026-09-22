"""Final architecture comparison A-G, with the 0-5 dynamic Down policy.

  A  Current production                        (Up-first + existing Down overlay)
  B  Current Up + Down V2 overlay only
  C  Current Up + Down V2 overlay + external insertion
  D  Current Up + Down V3 overlay only
  E  Current Up + Down V3 overlay + external insertion
  F  Up V3 + Down V2
  G  Up V3 + Down V3

All designs share: 15-20 total calls (the frozen regime_cap per origin),
0-5 Down calls, no forced Down quota. Margins/floors screened on Tuning
only and frozen before Validation. Validation is the gate; Confirmation
is descriptive.

Writes: research/down_v3/metrics/final_comparison.json
"""
from __future__ import annotations

from json_output import strict_json_dumps
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"
V2PRED = ROOT / "research/down_v2/artifacts/down_v2_predictions.parquet"
V3PRED = ROOT / "research/down_v3/artifacts/v3_predictions.parquet"
UPV3 = ROOT / "research/down_v3/artifacts/up_v3_predictions.parquet"
OUT = ROOT / "research/down_v3/metrics/final_comparison.json"

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
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(logit(tune[score].to_numpy()).reshape(-1, 1), tune[ycol].astype(int))
    return m


def apply_platt(m: LogisticRegression, s) -> np.ndarray:
    return np.clip(m.predict_proba(logit(s).reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)


def call_correct(direction: pd.Series, y_true: pd.Series) -> np.ndarray:
    """Direction-aware correctness.

    `y_true` in the frozen artifacts is the **Up-correct label**: 1 iff the
    realized direction was Up. A Down call is therefore correct iff
    `y_true == 0`. Scoring every selected row with `y_true` — which the
    original version of this script did — counts a Down call as a hit exactly
    when it is wrong.
    """
    is_down = direction.fillna("Up").eq("Down").to_numpy()
    y = np.asarray(y_true, dtype=float)
    return np.where(is_down, 1.0 - y, y)


def month_select(
    g: pd.DataFrame,
    cap: int,
    margin: float,
    floor: float,
    use_overlay: bool,
    use_insert: bool,
    max_down: int = MAX_DOWN,
) -> pd.DataFrame:
    """One origin. Up ranking first, then optional overlay and insertion."""
    ready = g[g["level_c_ready"].fillna(False).astype(bool)].copy()
    ranked = ready.sort_values(
        ["up_score", "indicator_id"], ascending=[False, True]
    )
    base = ranked.head(cap).copy()
    base["source"] = "up"
    # `pred_dir` is the production direction, including production's own Down
    # overlay. Without it the base pool's existing Down calls are invisible and
    # become eligible victims for an external Down insert.
    base["pred_dir"] = base.get("pred_dir", base["predicted_direction"])
    base["final_direction"] = base["pred_dir"].fillna("Up")
    base["call_correct"] = call_correct(base["final_direction"], base["y_true"])
    base["down_correct"] = base["call_correct"]
    base["is_down_call"] = base["final_direction"].eq("Down")
    base["victim_id"] = ""
    base["victim_dir"] = ""
    base["victim_correct"] = np.nan
    # retained for traceability: the raw Up-correct label of the displaced call
    base["victim_y"] = np.nan
    outside = ranked.iloc[cap:].copy()

    flips = 0
    if use_overlay:
        for idx, row in base.iterrows():
            if bool(row["is_down_call"]):
                continue
            if row["p_down_cal"] >= row["p_up_cal"] + margin and row["p_down_cal"] >= floor:
                base.loc[idx, "is_down_call"] = True
                base.loc[idx, "final_direction"] = "Down"
                base.loc[idx, "source"] = "overlay_flip"
                base.loc[idx, "call_correct"] = 1.0 - row["y_true"]
                base.loc[idx, "down_correct"] = 1.0 - row["y_true"]
                base.loc[idx, "victim_id"] = row["indicator_id"]
                base.loc[idx, "victim_dir"] = "Up"
                base.loc[idx, "victim_correct"] = row["y_true"]
                base.loc[idx, "victim_y"] = row["y_true"]
                flips += 1
    budget = max_down - flips
    if not use_insert:
        budget = 0

    cand = outside[
        outside["p_down_cal"].ge(floor)
        & outside["p_down_cal"].ge(outside["p_up_cal"] + margin)
    ].sort_values("p_down_cal", ascending=False)
    n_insert = 0
    for _, crow in cand.iterrows():
        if n_insert >= budget:
            break
        remaining = base[~base["is_down_call"]]
        if remaining.empty:
            break
        vid = remaining["p_up_cal"].idxmin()
        if crow["p_down_cal"] < base.loc[vid, "p_up_cal"] + margin:
            break
        # the displaced call's value is what it was actually worth under its
        # own direction, not its Up label
        victim_dir = str(base.loc[vid, "final_direction"])
        victim_y = float(base.loc[vid, "y_true"])
        victim_correct = float(base.loc[vid, "call_correct"])
        victim_id = str(base.loc[vid, "indicator_id"])
        base = base.drop(index=vid)
        # carry the full candidate row so provenance columns (config_hash,
        # model_version, regime_cap, ...) survive the insert
        new = crow.copy()
        new["is_down_call"] = True
        new["final_direction"] = "Down"
        new["call_correct"] = 1.0 - crow["y_true"]
        new["down_correct"] = 1.0 - crow["y_true"]
        new["source"] = "external_insert"
        new["pred_dir"] = "Down"
        new["victim_id"] = victim_id
        new["victim_dir"] = victim_dir
        new["victim_correct"] = victim_correct
        new["victim_y"] = victim_y
        new.name = f"ins_{crow['indicator_id']}_{n_insert}"
        base = pd.concat([base, pd.DataFrame([new])], ignore_index=False)
        n_insert += 1

    base["flips"] = flips
    base["inserts"] = n_insert
    return base


def design_selections(
    j: pd.DataFrame, up_col: str, down_col: str, margin: float, floor: float,
    use_overlay: bool, use_insert: bool,
) -> pd.DataFrame:
    sels = []
    for origin, g in j.groupby("origin_position"):
        cap = int(g["regime_cap"].iloc[0])
        g = g.copy()
        g["pred_dir"] = g["predicted_direction"]
        sels.append(
            month_select(g, cap, margin, floor, use_overlay, use_insert)
        )
    sel = pd.concat(sels)
    return sel


def summarize(sel: pd.DataFrame) -> dict:
    per = {}
    for wname, (a, b) in WINDOWS.items():
        w = sel[sel["origin_position"].between(a, b)].copy()
        d = w[w["is_down_call"]]
        # replacement delta only over recorded events; base-pool Down rows have
        # no victim and contribute nothing
        ev = d[d["victim_correct"].notna()]
        rd = int((ev["down_correct"].astype(float) - ev["victim_correct"].astype(float)).sum())
        w = w.sort_values(
            ["origin_position", "is_down_call", "p_up_cal"], ascending=[True, False, False]
        )
        up = w[~w["is_down_call"]]
        per[wname] = {
            "calls": int(len(w)),
            # direction-aware: a Down call is a hit iff y_true == 0
            "hits": int(w["call_correct"].sum()),
            "hit_rate": float(w["call_correct"].mean()),
            "hits_up_label_deprecated": int(w["y_true"].sum()),
            "up_calls": int(len(up)),
            "up_hits": int(up["call_correct"].sum()),
            # explicit semantics: `final_down_calls` counts every selected row
            # whose final direction is Down (production Down rows + overlay
            # flips + external inserts). `external_down_inserts` counts only the
            # new out-of-pool insertions. The two must not be reported under one
            # name -- that is what made section 9 and section 15 disagree.
            "final_down_calls": int(len(d)),
            "final_down_hits": int(d["call_correct"].sum()),
            "down_precision": float(d["call_correct"].mean()) if len(d) else float("nan"),
            "down_hits": int(d["call_correct"].sum()),
            "external_down_inserts": int((d["source"] == "external_insert").sum()),
            "overlay_flips": int((d["source"] == "overlay_flip").sum()),
            "external_inserts": int((d["source"] == "external_insert").sum()),
            "prod_down_rows_in_base": int(
                ((d["source"] != "external_insert") & d["pred_dir"].fillna("Up").eq("Down")).sum()
            ),
            "replacement_delta": rd,
            "delta_mean": rd / len(ev) if len(ev) else float("nan"),
            "monthly_down_dist": {
                str(k): int(v)
                for k, v in w.groupby("origin_position")["is_down_call"]
                .sum().value_counts().sort_index().items()
            },
        }
    return per


def bootstrap(a_hits: pd.Series, b_hits: pd.Series, n_boot: int = 2000) -> dict:
    idx = a_hits.index.union(b_hits.index)
    xa = a_hits.reindex(idx).fillna(0.0).to_numpy()
    xb = b_hits.reindex(idx).fillna(0.0).to_numpy()
    n = len(idx)
    deltas = np.array(
        [
            xb[p].mean() - xa[p].mean()
            for p in (RNG.integers(0, n, size=n) for _ in range(n_boot))
        ]
    )
    return {
        "delta_mean": float((xb - xa).mean()),
        "p10": float(np.quantile(deltas, 0.10)),
        "p90": float(np.quantile(deltas, 0.90)),
        "p_delta_positive": float((deltas > 0).mean()),
    }


def main() -> None:
    art = pd.read_parquet(ACTIVE)
    v3 = pd.read_parquet(V3PRED)[["origin_position", "indicator_id", "logistic_platt"]]
    v2 = pd.read_parquet(V2PRED)[["origin_position", "indicator_id", "p_platt"]].rename(
        columns={"p_platt": "p_down_v2"}
    )
    up3 = pd.read_parquet(UPV3)[["origin_position", "indicator_id", "p_up_v3"]]
    j = art.merge(v3, on=["origin_position", "indicator_id"], how="left")
    j = j.merge(v2, on=["origin_position", "indicator_id"], how="left")
    j = j.merge(up3, on=["origin_position", "indicator_id"], how="left")
    j["p_down_v2"] = j["p_down_v2"].fillna(j["p_down"].fillna(0.0))
    j["down_correct"] = 1.0 - j["y_true"]

    # ---- common calibrated scale, all calibrators fit on Tuning only ----
    tmask = j["origin_position"].between(120, 179)
    j["p_up_cal"] = apply_platt(
        fit_platt(j[tmask], "p_up_selection_score", "y_true"),
        j["p_up_selection_score"].to_numpy(),
    )
    j["p_up_v3_cal"] = apply_platt(
        fit_platt(j[tmask & j["p_up_v3"].notna()], "p_up_v3", "y_true"),
        j["p_up_v3"].to_numpy(),
    )
    j["p_down_v3_cal"] = apply_platt(
        fit_platt(j[tmask], "logistic_platt", "down_correct"),
        j["logistic_platt"].to_numpy(),
    )
    j["p_down_v2_cal"] = apply_platt(
        fit_platt(j[tmask], "p_down_v2", "down_correct"),
        j["p_down_v2"].to_numpy(),
    )
    j["p_up_v3_cal"] = j["p_up_v3_cal"].fillna(j["p_up_cal"])

    # screen margins/floors on Tuning for each (up, down) pairing, strictest tie-break
    screens: dict[str, dict] = {}
    tune_groups = {o: g for o, g in j[tmask].groupby("origin_position")}
    for up_col, down_col, up_tag, down_tag in (
        ("p_up_selection_score", "p_down_v2_cal", "cur", "v2"),
        ("p_up_selection_score", "p_down_v3_cal", "cur", "v3"),
        ("p_up_v3_cal", "p_down_v2_cal", "up3", "v2"),
        ("p_up_v3_cal", "p_down_v3_cal", "up3", "v3"),
    ):
        key = f"{up_tag}_{down_tag}"
        best = None
        for margin in (0.08, 0.05, 0.03, 0.02, 0.0):
            for floor in (0.65, 0.60, 0.55, 0.50):
                sels = []
                for origin, g in tune_groups.items():
                    g = g.copy()
                    g["up_score"] = g[up_col]
                    g["p_down_cal"] = g[down_col]
                    g["pred_dir"] = g["predicted_direction"]
                    cap = int(g["regime_cap"].iloc[0])
                    sels.append(month_select(g, cap, margin, floor, True, True))
                s = pd.concat(sels)
                d = s[s["is_down_call"]]
                if not len(d):
                    continue
                # direction-aware, events only (base-pool Down rows have no victim)
                ev = d[d["victim_correct"].notna()]
                delta = int(
                    (ev["down_correct"].astype(float) - ev["victim_correct"].astype(float)).sum()
                )
                cand_key = (delta, -len(d), margin, floor)
                if best is None or cand_key > best[0]:
                    best = (cand_key, margin, floor, len(d), delta)
                print(
                    f"  {key} margin={margin:.2f} floor={floor:.2f}: "
                    f"calls={len(d)} delta={delta:+d}"
                )
        if best is None:
            best = (None, 0.05, 0.55, 0, 0)
        screens[key] = {
            "margin": best[1], "floor": best[2],
            "tuning_calls": best[3], "tuning_delta": best[4],
        }
        print(f"selected {key}: margin={best[1]:.2f} floor={best[2]:.2f} calls={best[3]} delta={best[4]:+d}")

    # ---- materialise A-G ----
    designs = {
        "A": ("p_up_selection_score", "p_down_v3_cal", "cur_v3", False, False),
        "B": ("p_up_selection_score", "p_down_v2_cal", "cur_v2", True, False),
        "C": ("p_up_selection_score", "p_down_v2_cal", "cur_v2", True, True),
        "D": ("p_up_selection_score", "p_down_v3_cal", "cur_v3", True, False),
        "E": ("p_up_selection_score", "p_down_v3_cal", "cur_v3", True, True),
        "F": ("p_up_v3_cal", "p_down_v2_cal", "up3_v2", True, True),
        "G": ("p_up_v3_cal", "p_down_v3_cal", "up3_v3", True, True),
    }
    results: dict[str, object] = {"screens": screens, "max_down": MAX_DOWN}
    a_sel = None
    for name, (up_col, down_col, key, ov, ins) in designs.items():
        scr = screens[key]
        j2 = j.copy()
        j2["up_score"] = j2[up_col]
        j2["p_down_cal"] = j2[down_col]
        sel = design_selections(
            j2, up_col, down_col, scr["margin"], scr["floor"], ov, ins
        )
        sel["design"] = name
        sel.to_parquet(
            ROOT / f"research/down_v3/artifacts/final_sel_{name}.parquet", index=False
        )
        if name == "A":
            a_sel = sel
        results[name] = {
            "per_window": summarize(sel),
            "bootstrap_vs_A": None if name == "A" else bootstrap(
                a_sel.groupby("origin_position")["call_correct"].sum(),
                sel.groupby("origin_position")["call_correct"].sum(),
            ),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("Final architecture comparison (Tuning screen, Validation gate)")
    print("=" * 78)
    for name in designs:
        r = results[name]
        print(f"\n--- {name} ---")
        for wname, d in r["per_window"].items():
            print(
                f"  {wname:12s} calls={d['calls']:4d} hits={d['hits']:4d} "
                f"rate={d['hit_rate']:.4f} up={d['up_calls']}/{d['up_hits']} "
                f"down={d['final_down_calls']}/{d['final_down_hits']} "
                f"(prec={d['down_precision']:.4f}) "
                f"flips={d['overlay_flips']} ins={d['external_down_inserts']} "
                f"prod_down={d['prod_down_rows_in_base']} "
                f"delta={d['replacement_delta']:+d}"
            )
        b = r["bootstrap_vs_A"]
        if b:
            print(
                f"  bootstrap vs A: delta={b['delta_mean']:+.4f} "
                f"p10={b['p10']:+.4f} p90={b['p90']:+.4f} P(>0)={b['p_delta_positive']:.3f}"
            )

    # ---- arithmetic identity gate: every hit difference must be event-level ----
    # Decomposition:
    #   arch_hit_delta = replacement_delta + overlay_flip_delta
    #                    + up_ranking_delta + residual
    # `up_ranking_delta` is only non-zero for designs that replace the Up
    # ranking itself (F, G). It is measured directly, by re-running the same
    # design with both Down paths disabled and differencing against A.
    a_sel2 = results["A"]["per_window"]
    up_only: dict[str, dict] = {}
    for name, (up_col, down_col, key, ov, ins) in designs.items():
        if not ov and not ins:
            continue  # A itself, and B/C/D reduce to it
        j3 = j.copy()
        j3["up_score"] = j3[up_col]
        j3["p_down_cal"] = j3[down_col]
        scr = screens[key]
        sel_u = design_selections(j3, up_col, down_col, scr["margin"], scr["floor"], False, False)
        up_only[name] = {
            w: int(x["hits"]) for w, x in summarize(sel_u).items()
        }

    failures = []
    decomposition = {}
    for name in designs:
        if name == "A":
            continue
        dec = {}
        for wname, d in results[name]["per_window"].items():
            arch_delta = d["hits"] - a_sel2[wname]["hits"]
            flip_delta = 0  # overlay flips score (1-y)-y on the same row
            rep_delta = d["replacement_delta"]
            ur_delta = (up_only.get(name, {}).get(wname, a_sel2[wname]["hits"])
                        - a_sel2[wname]["hits"])
            residual = arch_delta - rep_delta - ur_delta - flip_delta
            dec[wname] = {
                "arch_hit_delta": arch_delta,
                "replacement_delta": rep_delta,
                "overlay_flip_delta": flip_delta,
                "up_ranking_delta": ur_delta,
                "residual": residual,
            }
            if residual != 0:
                failures.append(
                    f"{name}/{wname}: arch_hit_delta={arch_delta:+d} "
                    f"rep={rep_delta:+d} up_rank={ur_delta:+d} "
                    f"residual={residual:+d}"
                )
        decomposition[name] = dec
    results["decomposition"] = decomposition
    OUT.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        print("\n" + "!" * 78)
        print("RECONCILIATION FAILURES")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)
    print("\nreconciliation OK: arch_hit_delta == replacement_delta + up_ranking_delta"
          " for every design/window (residual 0)")
    print("\ndecomposition (validation):")
    for name, dec in decomposition.items():
        v = dec["validation"]
        print(
            f"  {name}: arch={v['arch_hit_delta']:+d} rep={v['replacement_delta']:+d} "
            f"flip={v['overlay_flip_delta']:+d} up_rank={v['up_ranking_delta']:+d} "
            f"residual={v['residual']:+d}"
        )
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
