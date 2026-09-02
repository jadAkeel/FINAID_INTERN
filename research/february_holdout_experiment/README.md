# February Holdout Experiment — March / April / May 2026 Terminal Holdout

This directory records the bounded challenger work described in the mission.
The holdout (origins 314 = 2026-03, 315 = 2026-04, 316 = 2026-05) is terminal
and was opened only for the frozen baseline vs frozen challenger comparison
after development. Development itself used only origins ≤313 and never read
314-316 for fitting or selection.

See scripts:
- diagnose_regime_adaptive.py — full Phase 1 diagnosis
- candidate_a_hierarchical.py — Candidate A hierarchical EB group overlay
- candidate_b_ranker.py — Candidate B cross-sectional regularized ranker (bounded)
- candidate_c_selection_correction.py — Candidate C selection quality correction
- candidate_d_reliability_gated_group.py — Candidate D causal reliability-gated group score
- final_holdout_report.py — frozen baseline vs frozen challenger from Feb origin 313

Artifacts:
- development_report.json
- march_may_holdout_report.json
- group_ablation.csv
- model_comparison.csv
