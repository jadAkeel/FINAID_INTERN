# Decisions and assumptions

- Repository was empty at intake; no existing code, tests, dependencies, or user changes were present.
- Source copies are byte-for-byte copies of the Downloads inputs; originals remain untouched.
- Workbook dimensions are discovered at runtime; no `300` or `40` constants control the pipeline.
- One conservative availability lag month is used because release timestamps and vintages are absent. This is explicitly pseudo-OOS, not a real-time backtest.
- Feature transformations are causal and computed from observations available at or before the as-of row. Training rows for origin t are strictly earlier than t.
- The official target remains next value strictly greater than current; ties are not converted to Up.
- The first reproducible implementation promotes only a regularized global Logistic as a classical anchor. Equal-weight ensemble output is provisional until evidence is reviewed.
- Pretrained model packages/APIs were not locally verifiable; no external upload or unverified installation was attempted.
- Locked audit is frozen after code and configuration commit; a post-audit redesign would invalidate the audit.
- A production-only eligibility fix was applied after the v1 audit so the May 2026 row can produce an unscored ledger without requiring a missing t+1 label. It does not alter the v1 audit path or its results; the v1 freeze manifest remains the authoritative audit manifest.
