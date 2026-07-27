# Blockers and negative results

| ID | Status | Evidence | Impact |
|---|---|---|---|
| B-001 | open | No indicator release-lag, vintage, revision, unit, or semantic metadata supplied | True point-in-time/real-time claim is blocked; study is revised-data pseudo-OOS |
| B-002 | open | `chronos` Python package is not installed; TiRex-2 and TimesFM packages/checkpoints/interfaces are not locally verified | Pretrained experiments cannot be honestly run or scored |
| B-003 | open | Workbook ends at 2026-05-29 | PDF January-2026 six-month plus five-month persistence milestone is not evaluable |
| B-004 | resolved as negative | Parallel OpenCode planner/architect jobs were started with read-only scope but timed out before producing output; no files changed | Direct orchestrator implementation continued; no worker claim is made |
| B-005 | documented | Production-only ledger eligibility was corrected after the v1 audit to allow an unscored origin with no t+1 label | v1 audit remains valid for its frozen evaluation path; production patch is not a new audit result |
| B-006 | resolved for v2 | Earlier monolithic CatBoost run exceeded the bounded 300-second budget; chunked v2 completed all 19 chunks and assembled a validated full artifact | The prior timeout remains historical; v2 is available for OOF comparison |
| B-007 | open/non-promotion | CatBoost v2 OOF accuracy 52.34%, Brier 0.25848, and log loss 0.71209 did not beat the existing anchors | Keep CatBoost as rejected challenger; do not alter Level-C v2 or production policy |
| B-008 | open/negative | Local preflight found no Chronos-2, TiRex-2, or TimesFM package/API and no compatible checkpoints in `artifacts/pretrained_cache` | Pretrained smoke tests are not run; no download or external API access is allowed |
| B-009 | open | Historical release lags and vintage data are absent | Claims remain revised-data pseudo-OOS, not true real-time |
