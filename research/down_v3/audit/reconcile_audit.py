"""Row-level reconciliation audit for the Down V3 architecture comparison.

Independent of v3_selection.py / final_comparison.py. Reads only the frozen
selection artifacts and the production prediction artifact, and rebuilds every
reported number from individual rows.

Metric definitions (authoritative for this audit)
--------------------------------------------------
`y_true` in the frozen artifacts is the **Up-correct label**: 1 iff the realized
direction of (indicator, target_date) was Up. Therefore, for a selected call:

    call_correct = y_true                     if final direction == Up
    call_correct = 1 - y_true                 if final direction == Down

`hits` and `hit_rate` are defined on `call_correct`. Scoring a Down call with
`y_true` scores it as a hit exactly when it is wrong; that is the defect this
audit exists to detect.

    hits                 = sum(call_correct) over the selected set
    down_calls           = # selected rows with final direction Down
    down_hits            = sum(call_correct) over down rows
    down_precision       = down_hits / down_calls
    external_inserts     = down rows whose indicator was outside the Up base pool
    overlay_flips        = base-pool rows whose direction changed Up -> Down
    victim_correct       = call_correct of the displaced row (direction-aware)
    replacement_delta    = down_correct - victim_correct   (per replacement event)
    arch_hit_delta       = challenger_hits - baseline_hits (direction-aware)

Identity under test (no other selection change):

    arch_hit_delta == sum(replacement_delta) + sum(flip_delta)

Writes:
    research/down_v3/audit/replacement_ledger.parquet
    research/down_v3/audit/architecture_reconciliation.csv
    research/down_v3/audit/audit_findings.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from json_output import strict_json_dumps  # noqa: E402
V3 = ROOT / "research/down_v3"
AUDIT = V3 / "audit"
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}
DESIGNS = ["A", "B", "C", "D", "E", "F", "G"]


def call_correct(df: pd.DataFrame) -> pd.Series:
    """Direction-aware correctness. `y_true` is the Up-correct label."""
    is_down = df["final_direction"].fillna("Up").eq("Down")
    return np.where(is_down, 1.0 - df["y_true"], df["y_true"])


def load_design(name: str) -> pd.DataFrame:
    """Load a final_sel_<name> artifact and normalise the audit columns.

    The frozen artifacts store the *production* `predicted_direction` on base
    rows and NaN on inserted rows; inserted rows are always Down. The final
    direction of a selected row is therefore:

        base row      -> predicted_direction  (production's own overlay included)
        inserted row  -> Down
    """
    p = V3 / f"artifacts/final_sel_{name}.parquet"
    df = pd.read_parquet(p)
    df["final_direction"] = np.where(
        df["source"].eq("external_insert"),
        "Down",
        df["predicted_direction"].fillna("Up"),
    )
    df["call_correct"] = call_correct(df)
    df["arch"] = name
    return df


def window_of(o: pd.Series) -> pd.Series:
    out = pd.Series("outside", index=o.index, dtype=object)
    for w, (lo, hi) in WINDOWS.items():
        out[o.between(lo, hi)] = w
    return out


# --------------------------------------------------------------------------
# 1. Replacement ledger
# --------------------------------------------------------------------------
def build_ledger(designs: dict[str, pd.DataFrame], prod: pd.DataFrame) -> pd.DataFrame:
    """One row per replacement event (external insert or overlay flip).

    The victim's outcome is looked up in the *production* artifact so that a
    victim which was itself a Down-direction call is scored direction-aware:
    what the displaced call was actually worth, not its Up label.
    """
    # base pool per origin of design A = the Up-first ranking head(cap)
    a = designs["A"]
    base_keys = set(zip(a["origin_position"], a["indicator_id"]))

    # production call_correct lookup (origin, indicator) -> (dir, correct)
    prod_c = prod[prod["accepted"].fillna(False).astype(bool)].copy()
    prod_c["p_dir"] = prod_c["predicted_direction"].fillna("Up")
    prod_c["p_correct"] = np.where(
        prod_c["p_dir"].eq("Down"), 1 - prod_c["y_true"], prod_c["y_true"]
    )
    lut = prod_c.set_index(["origin_position", "indicator_id"])[
        ["p_dir", "p_correct", "selection_rank", "p_up_selection_score", "p_up_calibrated"]
    ]

    # scalar lookups: the victim is unique per (origin, indicator)
    lut = lut.groupby(level=["origin_position", "indicator_id"]).first()

    # inserted rows in the frozen artifacts drop provenance columns that the
    # base rows carry (regime_cap, origin_date, target_date). Fill per origin.
    ctx = {}
    for name, df in designs.items():
        g = df.groupby("origin_position").agg(
            cap=("regime_cap", "max"),
            odate=("origin_date", "max"),
            tdate=("target_date", "max"),
        )
        ctx[name] = g

    recs = []
    for name, df in designs.items():
        d = df[df["source"].isin(["external_insert", "overlay_flip"])].copy()
        for _, r in d.iterrows():
            o = int(r["origin_position"])
            c = ctx[name].loc[o]
            key = (o, str(r["victim_id"]))
            if key in lut.index:
                v = lut.loc[key]
                v_dir = str(v["p_dir"])
                v_correct = float(v["p_correct"])
                v_rank = float(v["selection_rank"]) if pd.notna(v["selection_rank"]) else np.nan
                v_upcal = float(v["p_up_calibrated"])
                v_upsel = float(v["p_up_selection_score"])
            else:
                # victim was not a production-accepted row (can happen when the
                # research ranking differs from production's own pool). Fall
                # back to the recorded Up label.
                v_dir, v_correct, v_rank, v_upcal, v_upsel = "Up", float(r["victim_y"]), np.nan, np.nan, np.nan
            down_correct = float(r["call_correct"])
            recs.append(
                {
                    "arch": name,
                    "origin": int(r["origin_position"]),
                    "down_indicator": r["indicator_id"],
                    "down_final_direction": r["final_direction"],
                    "down_correct": down_correct,
                    "down_y_true": float(r["y_true"]),
                    "down_p_down_cal": float(r.get("p_down_cal", np.nan)),
                    "down_p_down_raw": float(r.get("p_down", np.nan)),
                    "down_p_up_cal": float(r.get("p_up_cal", np.nan)),
                    "down_up_score": float(r.get("up_score", np.nan)),
                    "in_A_base_pool": (int(r["origin_position"]), str(r["indicator_id"])) in base_keys,
                    "victim_indicator": r["victim_id"],
                    "victim_direction": v_dir,
                    "victim_up_correct": v_correct,
                    "victim_y_true": float(r["victim_y"]),
                    "victim_p_up_cal": v_upcal,
                    "victim_up_score": v_upsel,
                    "victim_rank": v_rank,
                    "path": "overlay" if r["source"] == "overlay_flip" else "external",
                    "replacement_delta": down_correct - v_correct,
                    "recorded_delta_label": float(r["victim_y"]),
                    "monthly_cap": int(c["cap"]),
                    "origin_date": c["odate"],
                    "target_date": c["tdate"],
                }
            )
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# 2. Per-origin reconciliation + set classification
# --------------------------------------------------------------------------
def reconcile(designs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = designs["A"]
    rows, cls_rows = [], []
    for name, df in designs.items():
        if name == "A":
            continue
        merged = []
        for o in sorted(a["origin_position"].unique()):
            ra = a[a["origin_position"] == o]
            rb = df[df["origin_position"] == o]
            ka = set(zip(ra["indicator_id"], ra["final_direction"]))
            kb = set(zip(rb["indicator_id"], rb["final_direction"]))
            ib = {i for i, _ in kb}
            for i, d in ka:
                cat = "same_indicator_same_dir" if (i, d) in kb else (
                    "same_indicator_dir_diff" if i in ib else "in_baseline_only"
                )
                r = ra[(ra["indicator_id"] == i)].iloc[0]
                merged.append(
                    {
                        "arch": name, "origin": o, "indicator_id": i,
                        "baseline_dir": d, "category": cat,
                        "baseline_correct": float(r["call_correct"]),
                        "challenger_correct": float(
                            rb[rb["indicator_id"] == i].iloc[0]["call_correct"]
                        ) if i in ib else np.nan,
                    }
                )
            for i, d in kb - ka:
                r = rb[(rb["indicator_id"] == i)].iloc[0]
                merged.append(
                    {
                        "arch": name, "origin": o, "indicator_id": i,
                        "baseline_dir": None, "category": "in_challenger_only",
                        "baseline_correct": np.nan,
                        "challenger_correct": float(r["call_correct"]),
                    }
                )
        m = pd.DataFrame(merged)
        m["event_delta"] = m["challenger_correct"] - m["baseline_correct"]
        cat = m.groupby("category").agg(
            n=("event_delta", "size"), sum_delta=("event_delta", "sum"),
            base_correct=("baseline_correct", "sum"), chall_correct=("challenger_correct", "sum"),
        )
        cls_rows.append({"arch": name, **{f"{k}_{c}": v for c, row in cat.iterrows() for k, v in row.items()}})

        # per-origin roll-up
        g = m.groupby("origin").agg(
            baseline_hits=("baseline_correct", "sum"),
            challenger_hits=("challenger_correct", "sum"),
            n_base=("baseline_correct", "size"),
            n_chall=("challenger_correct", "count"),
        )
        g["direct_hit_delta"] = g["challenger_hits"] - g["baseline_hits"]
        g["arch"] = name
        rows.append(g.reset_index())

    per_origin = pd.concat(rows)
    led = build_ledger(designs, pd.read_parquet(ACTIVE))
    ev = led.groupby(["arch", "origin"])["replacement_delta"].sum().rename("sum_replacement_delta")
    per_origin = per_origin.merge(ev, on=["arch", "origin"], how="left")
    per_origin["sum_replacement_delta"] = per_origin["sum_replacement_delta"].fillna(0.0)
    per_origin["unexplained_delta"] = (
        per_origin["direct_hit_delta"] - per_origin["sum_replacement_delta"]
    )
    return per_origin, pd.DataFrame(cls_rows)


# --------------------------------------------------------------------------
# 3. Corrected window table
# --------------------------------------------------------------------------
def window_table(designs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for name, df in designs.items():
        df = df.assign(win=window_of(df["origin_position"]))
        for w, g in df.groupby("win"):
            if w == "outside":
                continue
            d = g[g["final_direction"].eq("Down")]
            b = designs["A"].assign(win=window_of(designs["A"]["origin_position"]))
            bw = b[(b["win"] == w)]
            hits = int(g["call_correct"].sum())
            a_hits = int(bw["call_correct"].sum())
            out.append(
                {
                    "arch": name, "window": w, "calls": int(len(g)),
                    "hits_dir_aware": hits,
                    "hit_rate_dir_aware": hits / len(g),
                    "hits_y_true_reported": int(g["y_true"].sum()),
                    "final_up_calls": int((~g["final_direction"].eq("Down")).sum()),
                    "final_down_calls": int(len(d)),
                    "down_hits": int(d["call_correct"].sum()),
                    "down_precision": float(d["call_correct"].mean()) if len(d) else np.nan,
                    "overlay_flips": int((g["source"] == "overlay_flip").sum()),
                    "external_inserts": int((g["source"] == "external_insert").sum()),
                    "prod_down_rows_in_base": int(
                        ((~g["source"].eq("external_insert"))
                         & g["predicted_direction"].fillna("Up").eq("Down")).sum()
                    ),
                    "A_hits_dir_aware": a_hits,
                    "direct_hit_delta_vs_A": hits - a_hits,
                }
            )
    return pd.DataFrame(out)


def up_ranks(prod: pd.DataFrame) -> pd.DataFrame:
    """Rank of every ready indicator within its origin under the Up ranking.

    Reproduces `month_select`'s `ranked` order exactly: ready rows sorted by
    `p_up_selection_score` desc, `indicator_id` asc. Rank 1..cap is the base
    pool; rank > cap is the out-of-pool candidate universe.
    """
    r = prod[prod["level_c_ready"].fillna(False).astype(bool)].copy()
    r = r.sort_values(
        ["origin_position", "p_up_selection_score", "indicator_id"],
        ascending=[True, False, True],
    )
    r["up_pool_rank"] = r.groupby("origin_position").cumcount() + 1
    return r.set_index(["origin_position", "indicator_id"])["up_pool_rank"]


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    prod = pd.read_parquet(ACTIVE)
    designs = {n: load_design(n) for n in DESIGNS}

    led = build_ledger(designs, prod)
    ranks = up_ranks(prod)
    led["down_up_pool_rank"] = [
        float(ranks.get((int(o), str(i)), np.nan))
        for o, i in zip(led["origin"], led["down_indicator"])
    ]
    led["victim_up_pool_rank"] = [
        float(ranks.get((int(o), str(i)), np.nan))
        for o, i in zip(led["origin"], led["victim_indicator"])
    ]
    led["outside_up_pool"] = led["down_up_pool_rank"] > led["monthly_cap"]
    led.to_parquet(AUDIT / "replacement_ledger.parquet", index=False)

    per_origin, cls = reconcile(designs)
    per_origin.to_csv(AUDIT / "architecture_reconciliation.csv", index=False)
    cls.to_csv(AUDIT / "set_classification_by_category.csv", index=False)

    tab = window_table(designs)
    tab.to_csv(AUDIT / "corrected_window_table.csv", index=False)

    findings = {
        "n_designs": len(designs),
        "ledger_rows": int(len(led)),
        "per_origin_rows": int(len(per_origin)),
        "nonzero_unexplained": int((per_origin["unexplained_delta"] != 0).sum()),
        "worst_unexplained": float(per_origin["unexplained_delta"].abs().max()),
        "window_table": tab.to_dict(orient="records"),
    }
    (AUDIT / "audit_findings.json").write_text(
        strict_json_dumps(findings, indent=2, default=str), encoding="utf-8"
    )

    pd.set_option("display.width", 220)
    print("=== replacement ledger (by arch / path) ===")
    print(
        led.groupby(["arch", "path"]).agg(
            n=("replacement_delta", "size"),
            down_hits=("down_correct", "sum"),
            victim_hits=("victim_up_correct", "sum"),
            delta=("replacement_delta", "sum"),
        ).to_string()
    )
    print("\n=== corrected window table ===")
    cols = [
        "arch", "window", "calls", "hits_dir_aware", "hit_rate_dir_aware",
        "hits_y_true_reported", "final_up_calls", "final_down_calls",
        "down_hits", "down_precision", "overlay_flips", "external_inserts",
        "prod_down_rows_in_base", "direct_hit_delta_vs_A",
    ]
    print(tab[cols].to_string(index=False))
    print("\n=== unexplained delta per origin (nonzero only) ===")
    bad = per_origin[per_origin["unexplained_delta"] != 0]
    print(f"{len(bad)} nonzero of {len(per_origin)}; max |.|={findings['worst_unexplained']}")
    # Nonzero `unexplained_delta` is a defect only for designs that keep A's Up
    # ranking (B/C/D/E). F and G replace the Up ranking arm, so their whole hit
    # difference is `other_selection_delta`; it is measured separately by the
    # disabled-Down re-run recorded in final_comparison.json["decomposition"].
    keepers = bad[~bad["arch"].isin(["F", "G"])]
    if len(keepers):
        print("DEFECT: nonzero unexplained_delta outside F/G:")
        print(keepers.to_string(index=False))
    else:
        print("all nonzero rows are F/G (Up-ranking arm) -- expected, see AUDIT_REPORT.md §7")


if __name__ == "__main__":
    main()
