"""P5a: Component ablation for the Down model.

Free ablation: the frozen artifact already stores p_down_global, p_down_local,
p_down_pattern, p_down_indicator_prior per origin. Any blend weight vector can
be evaluated without refitting, so we grid the blend space and run
leave-one-component-out tests on Tuning, Validation, Confirmation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PRED = ROOT / "research/down_v2/artifacts/down_refit_predictions.parquet"
WINDOWS = {
    "tuning": (120, 179),
    "validation": (180, 219),
    "confirmation": (220, 266),
}
COMPONENTS = [
    "p_down_global",
    "p_down_local",
    "p_down_pattern",
    "p_down_indicator_prior",
]


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean": float(p.mean()),
        "p95": float(np.quantile(p, 0.95)),
    }


def topn_hit_rate(frame: pd.DataFrame, score: str, n: int) -> float:
    """Monthly top-n by score; fraction whose down_target == 1."""
    top = frame.sort_values(
        ["origin_position", score], ascending=[True, False]
    ).groupby("origin_position").head(n)
    return float(top["down_target"].mean())


def main() -> None:
    pred = pd.read_parquet(PRED)
    pred["down_target"] = pred["down_target"].astype(float)
    print("=" * 78)
    print("P5a  Down component ablation (no refit; blend grid over frozen parts)")
    print("=" * 78)

    # reproduce the production blend: local 0.25, pattern 0.25, global 0.50
    def blend(w_global, w_local, w_pattern):
        return (
            w_global * pred["p_down_global"]
            + w_local * pred["p_down_local"]
            + w_pattern * pred["p_down_pattern"]
        ).clip(1e-6, 1 - 1e-6)

    print("\n--- Blend grid: AUC / Brier / top-5 Down hit rate per window ---")
    print(f"{'w_glob':>7s} {'w_loc':>6s} {'w_pat':>6s} | " + " | ".join(
        f"{n} AUC/Brier/T5" for n in WINDOWS
    ))
    rows = []
    for wg in (1.0, 0.75, 0.5):
        for wl in (0.0, 0.25, 0.5):
            for wp in (0.0, 0.25, 0.5):
                if abs(wg + wl + wp - 1.0) > 1e-9:
                    continue
                s = blend(wg, wl, wp)
                tmp = pred.copy()
                tmp["blend"] = s
                cells = []
                for _, (a, b) in WINDOWS.items():
                    w = tmp[tmp["origin_position"].between(a, b)]
                    m = summarize(w["down_target"].to_numpy(), w["blend"].to_numpy())
                    t5 = topn_hit_rate(w, "blend", 5)
                    cells.append(f"{m['auc']:.4f}/{m['brier']:.4f}/{t5:.4f}")
                print(f"{wg:7.2f} {wl:6.2f} {wp:6.2f} | " + " | ".join(cells))
                rows.append({"w_global": wg, "w_local": wl, "w_pattern": wp,
                             **{f"{k}_{n}": v for n, cell in zip(WINDOWS, cells)
                                for k, v in [("auc", float(cell.split("/")[0])),
                                             ("brier", float(cell.split("/")[1])),
                                             ("top5", float(cell.split("/")[2]))]}})
    print()

    print("--- Leave-one-component-out (equal-weight blend minus one part) ---")
    base = pred[COMPONENTS].mean(axis=1).clip(1e-6, 1 - 1e-6)
    configs = {"all_four_equal": base}
    for drop in COMPONENTS:
        keep = [c for c in COMPONENTS if c != drop]
        configs[f"drop_{drop}"] = pred[keep].mean(axis=1).clip(1e-6, 1 - 1e-6)
    for name in COMPONENTS:
        configs[f"only_{name}"] = pred[name].astype(float)

    print(f"{'config':32s} {'T AUC':>7s} {'T Br':>6s} {'T top5':>7s} {'V AUC':>7s} "
          f"{'V Br':>6s} {'V top5':>7s} {'C AUC':>7s} {'C top5':>7s}")
    for name, s in configs.items():
        tmp = pred.copy()
        tmp["s"] = s.to_numpy() if hasattr(s, "to_numpy") else s
        out = [f"{name:32s}"]
        for _, (a, b) in WINDOWS.items():
            w = tmp[tmp["origin_position"].between(a, b)]
            m = summarize(w["down_target"].to_numpy(), w["s"].to_numpy())
            t5 = topn_hit_rate(w, "s", 5)
            out.append(f"{m['auc']:7.4f} {m['brier']:6.4f} {t5:7.4f}")
        print(" ".join(out))
    print()

    print("--- Indicator prior alone: is it just the base rate? ---")
    print(f"  p_down_indicator_prior std = {pred['p_down_indicator_prior'].std():.4f}")
    print(f"  overall down base rate      = {pred['down_target'].mean():.4f}")
    print()

    print("--- Local model availability ---")
    avail = pred.groupby("origin_position")["local_model_available"].mean()
    print(f"  origins with local models: mean share {avail.mean():.4f}")
    for _, (a, b) in WINDOWS.items():
        w = pred[pred["origin_position"].between(a, b)]
        print(f"  window {a}-{b}: local share {w['local_model_available'].mean():.4f}, "
              f"n indicators with local ~{w.groupby('origin_position')['local_model_available'].sum().mean():.1f}")
    print()
    print("--- Does local actually beat global where available? ---")
    for _, (a, b) in WINDOWS.items():
        w = pred[pred["origin_position"].between(a, b) & pred["local_model_available"]]
        if len(w):
            print(f"  {a}-{b}: AUC local={roc_auc_score(w['down_target'], w['p_down_local']):.4f} "
                  f"global={roc_auc_score(w['down_target'], w['p_down_global']):.4f}")


if __name__ == "__main__":
    main()
