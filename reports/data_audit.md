# Data audit

- Source SHA-256: `8f9fc27ae0a33f4a25d1241b7d896b56ba7515d4ad7e984999f0c5fe42b20d29`
- Worksheet: `Sheet1`
- Shape: 316 rows x 50 indicators
- Date range: 2000-02-29 through 2026-05-29
- Dates: sorted, unique, monthly, no missing calendar months (verified).
- Full-history indicators: X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X36, X37, X38, X39, X40, X42, X43
- No negative values were observed; log-return and percentage-change eligibility remains series/fold validated.
- Leading missing history is preserved; no backfill or interpolation is used.
- X16 stale/repeated values are profiled as data, not silently removed.

## Eligibility

The matrix is represented in `data_profile.json`; a row is eligible only when the as-of feature is observed, the official t-to-t+1 target is observed, and at least 24 months of history exist.

## Validation boundaries

Development origins: positions 120-267; locked audit: positions 268-315 (48 origins); production origin: position 316.
