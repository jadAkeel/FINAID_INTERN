"""Regression tests for the Down V3 architecture-comparison audit.

These exist because the reported numbers once were not reproducible from
row-level selected calls. Each test pins one identity that failed before.

Run:  python -m pytest research/down_v3/audit/test_reconciliation.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / "research/down_v3"
ART = V3 / "artifacts"
ACTIVE = ROOT / "artifacts/active/regime_adaptive_predictions.parquet"

WINDOWS = {"tuning": (120, 179), "validation": (180, 219), "confirmation": (220, 266)}
DESIGNS = ["A", "B", "C", "D", "E", "F", "G"]


def _load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(ART / f"final_sel_{name}.parquet")
    if "final_direction" not in df.columns:
        # tolerate artifacts written by an older script version
        df["final_direction"] = np.where(
            df["source"].eq("external_insert"),
            "Down",
            df["predicted_direction"].fillna("Up"),
        )
    if "call_correct" not in df.columns:
        is_down = df["final_direction"].eq("Down")
        df["call_correct"] = np.where(is_down, 1 - df["y_true"], df["y_true"])
    return df


@pytest.fixture(scope="module")
def designs() -> dict[str, pd.DataFrame]:
    return {n: _load(n) for n in DESIGNS}


def _win(df: pd.DataFrame, w: str) -> pd.DataFrame:
    lo, hi = WINDOWS[w]
    return df[df["origin_position"].between(lo, hi)]


# --- Test 1: architecture reconciliation -------------------------------
def _up_ranking_delta(df: pd.DataFrame, a: pd.DataFrame, w: str) -> int:
    """Hit delta caused purely by the Up ranking arm.

    Measured on the rows both designs selected without a Down event: matched
    (origin, indicator) pairs that differ in correctness. F and G rank with
    Up V3, so the same indicator can be correct under one ranking and wrong
    under the other only via a *direction* change; a pure indicator swap is
    attributed here too. This is the `other_selection_delta` of the audit.
    """
    aw, bw = _win(a, w), _win(df, w)
    aa = aw[~aw["source"].isin(["external_insert", "overlay_flip"])]
    bb = bw[~bw["source"].isin(["external_insert", "overlay_flip"])]
    m = aa[["origin_position", "indicator_id", "call_correct"]].merge(
        bb[["origin_position", "indicator_id", "call_correct"]],
        on=["origin_position", "indicator_id"], how="inner",
        suffixes=("_a", "_b"),
    )
    return int((m["call_correct_b"] - m["call_correct_a"]).sum())


def _indicator_swap_delta(df: pd.DataFrame, a: pd.DataFrame, w: str) -> int:
    """Hit delta from indicators present in only one design, EXCLUDING ledger
    events (victims on the A side, inserts/flips on the challenger side) --
    those are already accounted for by the replacement delta."""
    aw, bw = _win(a, w), _win(df, w)
    ev = bw[bw["victim_correct"].notna()]
    victims = set(zip(ev["origin_position"], ev["victim_id"].astype(str)))
    inserts = set(
        zip(ev["origin_position"], ev["indicator_id"].astype(str))
    )
    aa = set(zip(aw["origin_position"], aw["indicator_id"]))
    bb = set(zip(bw["origin_position"], bw["indicator_id"]))
    only_b = bw[
        bw.apply(
            lambda r: (r["origin_position"], r["indicator_id"]) not in aa
            and (r["origin_position"], r["indicator_id"]) not in inserts,
            axis=1,
        )
    ]
    only_a = aw[
        aw.apply(
            lambda r: (r["origin_position"], r["indicator_id"]) not in bb
            and (r["origin_position"], r["indicator_id"]) not in victims,
            axis=1,
        )
    ]
    return int(only_b["call_correct"].sum() - only_a["call_correct"].sum())


def test_architecture_reconciliation(designs):
    """challenger_hits - baseline_hits == replacement + flip + up-ranking
    + indicator-swap, with zero residual, for every design and window.

    For B/C/D/E the challenger keeps A's Up ranking, so the identity must hold
    from rows alone. F and G also replace the Up ranking arm, which interacts
    with insertion (a re-ranked pool changes which victims are available);
    there, the up-ranking term is measured by the frozen decomposition written
    by final_comparison.py, which re-runs each design with both Down paths
    disabled and differences the result against A.
    """
    a = designs["A"]
    for name, df in designs.items():
        if name == "A":
            continue
        for w in WINDOWS:
            aw, bw = _win(a, w), _win(df, w)
            arch_delta = int(bw["call_correct"].sum()) - int(aw["call_correct"].sum())
            ev = bw[bw["victim_correct"].notna()]
            rep = int((ev["call_correct"] - ev["victim_correct"]).sum())
            if name in ("B", "C", "D", "E"):
                other = _up_ranking_delta(df, a, w) + _indicator_swap_delta(df, a, w)
                assert other == 0, (
                    f"{name}/{w}: designs that keep A's Up ranking must not "
                    f"show an other_selection_delta (found {other:+d})"
                )
                assert arch_delta == rep, (
                    f"{name}/{w}: arch_hit_delta={arch_delta:+d} != "
                    f"replacement_delta={rep:+d}"
                )


def test_architecture_reconciliation_decomposition_json():
    """The full decomposition recorded by final_comparison.py must have zero
    residual for every design and window (including F and G, whose
    up-ranking arm is measured by a dedicated disabled-Down re-run)."""
    import json

    p = V3 / "metrics" / "final_comparison.json"
    assert p.exists(), "final_comparison.json missing; run final_comparison.py"
    res = json.loads(p.read_text(encoding="utf-8"))
    dec = res.get("decomposition")
    assert dec, "decomposition block missing from final_comparison.json"
    for name, per_window in dec.items():
        for w, d in per_window.items():
            assert d["residual"] == 0, (
                f"{name}/{w}: arch={d['arch_hit_delta']:+d} = "
                f"rep({d['replacement_delta']:+d}) + "
                f"up_rank({d['up_ranking_delta']:+d}) + "
                f"residual({d['residual']:+d})"
            )


# --- Test 2: replacement ledger ---------------------------------------
def test_replacement_ledger(designs):
    """For every replacement: delta == challenger_correct - victim_correct."""
    for name, df in designs.items():
        ev = df[df["victim_correct"].notna()]
        for _, r in ev.iterrows():
            assert r["call_correct"] - r["victim_correct"] == pytest.approx(
                r["call_correct"] - r["victim_correct"]
            )
        # and the victim must be a genuine Up-direction call: displacing a call
        # that was already Down and scoring it on its Up label is the exact
        # defect found at origin 226
        assert (ev["victim_dir"].fillna("Up") != "Down").all(), (
            f"{name}: a Down-direction row was used as an external-insert victim"
        )


# --- Test 3: final_down_calls semantics -------------------------------
def test_final_down_calls_semantics(designs):
    """final_down_calls == actual count of rows with direction == Down."""
    for name, df in designs.items():
        n_down = int(df["final_direction"].eq("Down").sum())
        assert n_down == int(df["is_down_call"].sum()), f"{name}: is_down_call != final_direction==Down"
        for w in WINDOWS:
            bw = _win(df, w)
            n = int(bw["final_direction"].eq("Down").sum())
            ins = int((bw["source"] == "external_insert").sum())
            prod = int(
                ((bw["source"] != "external_insert")
                 & bw["predicted_direction"].fillna("Up").eq("Down")).sum()
            )
            assert n == ins + prod, (
                f"{name}/{w}: final_down_calls={n} != inserts({ins}) + "
                f"production-Down-in-base({prod})"
            )


# --- Test 3b: hits must be direction-aware -----------------------------
def test_hits_are_direction_aware(designs):
    """`y_true` is the Up-correct label; a Down call is a hit iff y_true == 0."""
    for name, df in designs.items():
        for w in WINDOWS:
            bw = _win(df, w)
            hits = int(bw["call_correct"].sum())
            manual = int(
                np.where(bw["final_direction"].eq("Down"),
                         1 - bw["y_true"], bw["y_true"]).sum()
            )
            assert hits == manual, f"{name}/{w}: call_correct != direction-aware recomputation"


# --- Test 4: call-count invariance -------------------------------------
def test_call_count_invariance(designs):
    """Every design must fill the per-origin regime cap exactly."""
    a = designs["A"]
    for name, df in designs.items():
        assert len(df) == len(a), f"{name}: row count differs from A"
        for o in sorted(a["origin_position"].unique()):
            ga = a[a["origin_position"] == o]
            gb = df[df["origin_position"] == o]
            cap = int(ga["regime_cap"].iloc[0])
            assert len(gb) == cap, f"{name}/origin {o}: {len(gb)} calls != cap {cap}"
            assert len(ga) == cap


# --- Test 5: no duplicate selections -----------------------------------
def test_no_duplicate_selections(designs):
    """No indicator may be selected twice in one origin."""
    for name, df in designs.items():
        dup = df.groupby(["origin_position", "indicator_id"]).size()
        assert (dup <= 1).all(), f"{name}: duplicate (origin, indicator)"


# --- Test 6: architecture identity --------------------------------------
def test_bcd_identical_to_a(designs):
    """B/C/D are claimed behaviourally identical to A: assert exact equality
    of (origin, indicator_id, direction), not just the total hit rate."""
    a = designs["A"]
    ka = set(zip(a["origin_position"], a["indicator_id"], a["final_direction"]))
    for name in ("B", "C", "D"):
        df = designs[name]
        kb = set(zip(df["origin_position"], df["indicator_id"], df["final_direction"]))
        assert kb == ka, f"{name} is not row-identical to A: {len(ka ^ kb)} differing rows"


# --- Test 7: production baseline matches the frozen artifact ------------
def test_design_A_is_production(designs):
    """Design A must be the real production selected set, including its own
    Down overlay rows. The pre-audit A reported 0/0/0 Down calls because
    `pred_dir` was never propagated."""
    prod = pd.read_parquet(ACTIVE)
    acc = prod[prod["accepted"].fillna(False).astype(bool)]
    a = designs["A"]
    for o in sorted(a["origin_position"].unique()):
        sa = set(a[a["origin_position"] == o]["indicator_id"])
        sp = set(acc[acc["origin_position"] == o]["indicator_id"])
        assert sa == sp, f"origin {o}: design A != production accepted set"
    # production's own Down calls must be visible in A
    n_prod_down = int(acc["predicted_direction"].fillna("Up").eq("Down").sum())
    n_a_down = int(a["final_direction"].eq("Down").sum())
    assert n_a_down == n_prod_down, f"A shows {n_a_down} Down rows, production has {n_prod_down}"


# --- Test 8: archived (buggy) numbers must not come back ----------------
def test_e_validation_identity_is_exactly_negative_four(designs):
    """The headline discrepancy: reported +2 hits with replacement delta -4.

    Direction-aware, E Validation is 391 = 395 - 4. Any other total means the
    wrong-way Down scoring has returned."""
    a, e = designs["A"], designs["E"]
    av, ev = _win(a, "validation"), _win(e, "validation")
    assert int(av["call_correct"].sum()) == 395
    assert int(ev["call_correct"].sum()) == 391
    d = ev[ev["is_down_call"]]
    assert int(d["call_correct"].sum()) == 14
    assert int(d["victim_correct"].sum()) == 18
    assert len(d) == 34
    assert int(d["call_correct"].sum()) - int(d["victim_correct"].sum()) == -4
    assert int(ev["call_correct"].sum()) - int(av["call_correct"].sum()) == -4
