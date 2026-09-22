"""P8: Threshold grid + Pareto curve for Down calls.

Uses the walk-forward refit predictions. For each threshold and margin, count
the Down calls that would have fired and their realized precision, per window.
Parameters are SCREENED on Tuning only; Validation is a gate and Confirmation
is descriptive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from json_output import strict_json_dumps  # noqa: E402
REFIT = ROOT / "research/down_v2/artifacts/down_refit_predictions.parquet"
P4 = ROOT / "research/down_v2/artifacts/p4_unified_scores.parquet"
OUT = ROOT / "research/down_v2/metrics/p8_threshold_pareto.json"

WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}


def down_calls(frame: pd.DataFrame, score: str, threshold: float) -> pd.DataFrame:
    return frame[frame[score].ge(threshold)]


def summarize_calls(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"calls": 0, "hits": 0, "precision": float("nan"), "months": 0}
    return {
        "calls": int(len(frame)),
        "hits": int(frame["down_target"].eq(1.0).sum()),
        "precision": float(frame["down_target"].eq(1.0).mean()),
        "months": int(frame["origin_position"].nunique()),
    }


def main() -> None:
    pred = pd.read_parquet(REFIT)
    pred["down_target"] = pred["down_target"].astype(float)
    pred["p_down_blend"] = pred[
        ["p_down_global", "p_down_local", "p_down_pattern", "p_down_indicator_prior"]
    ].mean(axis=1)

    print("=" * 78)
    print("P8  Threshold / Pareto analysis for Down calls")
    print("=" * 78)
    print("Screen on Tuning only. Validation = gate. Confirmation = descriptive.")
    print()

    results = {}
    for score in ["p_down_blend", "p_down_global", "p_down_indicator_prior"]:
        rows = []
        for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            row = {"threshold": threshold}
            for wname, (a, b) in WINDOWS.items():
                w = pred[pred["origin_position"].between(a, b)]
                row[wname] = summarize_calls(down_calls(w, score, threshold))
            rows.append(row)
        results[score] = rows

        print(f"--- {score} ---")
        print(f"{'thresh':>7s} | " + " | ".join(
            f"{n}: calls/prec" for n in WINDOWS))
        for row in rows:
            cells = []
            for wname in WINDOWS:
                d = row[wname]
                prec = f"{d['precision']:.4f}" if d["calls"] else "  --  "
                cells.append(f"{d['calls']:5d}/{prec}")
            print(f"{row['threshold']:7.2f} | " + " | ".join(f"{c:>12s}" for c in cells))
        print()

    print("--- Pareto: Tuning Down calls vs Tuning precision (p_down_blend) ---")
    for row in results["p_down_blend"]:
        d = row["tuning"]
        print(f"  threshold {row['threshold']:.2f}: calls={d['calls']:4d} "
              f"precision={d['precision'] if d['calls'] else float('nan'):.4f}")
    print()

    print("--- Does a high threshold stay stable on Validation? ---")
    print(f"{'threshold':>9s} {'T_prec':>8s} {'V_prec':>8s} {'C_prec':>8s} {'T_calls':>8s} {'V_calls':>8s}")
    for row in results["p_down_blend"]:
        t, v, c = row["tuning"], row["validation"], row["confirmation"]
        print(
            f"{row['threshold']:9.2f} "
            f"{(t['precision'] if t['calls'] else float('nan')):8.4f} "
            f"{(v['precision'] if v['calls'] else float('nan')):8.4f} "
            f"{(c['precision'] if c['calls'] else float('nan')):8.4f} "
            f"{t['calls']:8d} {v['calls']:8d}"
        )
    print()

    print("--- Margin requirement (p_down - p_up) on top of threshold ---")
    if P4.exists():
        u = pd.read_parquet(P4)
        merged = pred.merge(
            u[["origin_position", "indicator_id", "p_up_base"]],
            on=["origin_position", "indicator_id"],
            how="left",
        )
        merged["margin"] = merged["p_down_blend"] - merged["p_up_base"]
        for threshold in (0.55, 0.60, 0.65):
            for margin in (0.0, 0.05, 0.10, 0.15, 0.20):
                row = {"threshold": threshold, "margin": margin}
                for wname, (a, b) in WINDOWS.items():
                    w = merged[merged["origin_position"].between(a, b)]
                    sel = w[w["p_down_blend"].ge(threshold) & w["margin"].ge(margin)]
                    row[wname] = summarize_calls(sel)
                t, v = row["tuning"], row["validation"]
                tp = f"{t['precision']:.4f}" if t["calls"] else "--"
                vp = f"{v['precision']:.4f}" if v["calls"] else "--"
                print(
                    f"  thresh={threshold:.2f} margin={margin:.2f}: "
                    f"T={t['calls']:4d}/{tp} V={v['calls']:4d}/{vp}"
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(strict_json_dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
