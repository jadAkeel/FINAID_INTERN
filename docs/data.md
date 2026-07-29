# Data contract

The input workbook is `data/monthly_indicators.xlsx`.

- Worksheet: `Sheet1`
- Rows: 316 monthly observations
- Indicators: `X1` through `X50`
- Date range: February 2000 through May 2026
- Source SHA-256:
  `8f9fc27ae0a33f4a25d1241b7d896b56ba7515d4ad7e984999f0c5fe42b20d29`

The indicator names, units, release timestamps, and historical vintages were not
provided. This is therefore a revised-data pseudo-out-of-sample study, not a
real-time backtest.

Leading missing history is preserved. The pipeline does not interpolate,
backfill, or invent observations.

The original requirements and research plan are retained under
`docs/reference/`.
