# Source reconciliation

## Immutable inputs

The original files in `C:\Users\10User\Downloads` are read-only inputs. SHA-256 values were computed before repository work:

| Source | SHA-256 |
|---|---|
| `Internship Opportunity Forecasting Global macro indicators.pdf` | `469EFF44F695C4760997665846F40ADC72BE65968827B24C73A22458C0A6D359` |
| `plan_work (1).md` | `7F86EFB45749DA7E506F42009D52378578C8922EF6C9A2F4C6CC1AA41BE5F143` |
| `FinalList_Extended.xlsx` | `8F9FC27AE0A33F4A25D1241B7D896B56BA7515D4AD7E984999F0C5FE42B20D29` |

Read-only copies are kept under `data/raw/`; the Downloads originals were not modified or replaced.

## Precedence and resolved conflicts

1. The PDF defines the official objective: predict whether each indicator increases or decreases during the following month and reports an overall accuracy milestone.
2. The plan defines the detailed scientific protocol: date-grouped walk-forward validation, selective rejection, a hard maximum of 20 accepted predictions, and a soft target of 15.
3. The workbook schema is authoritative for dimensions. The supplied workbook has one visible worksheet (`Sheet1`), 316 monthly observations, and 50 indicators (`X1` through `X50`), not the approximate 300 x 40 described in the plan. Code derives dimensions dynamically.
4. Verified runtime behavior and local package availability override assumptions about pretrained model APIs. Chronos, TiRex, and TimesFM APIs were not verified locally; these are blocked negative results, not fabricated experiments.

## Required reconciliations

- **Full versus selective tracks:** the PDF's per-indicator objective is retained as a full-coverage research track. The plan's rejection and top-20 policy is a separate selective production track. They are never conflated.
- **65% milestone:** selective accepted accuracy is not interpreted as the PDF's overall accuracy. Both tracks are reported separately, and the milestone is `NOT_YET_EVALUABLE`.
- **Dimensions:** all row/indicator counts, origins, eligibility, and reports are parameterized from the workbook. The supplied 316-row fingerprint is independently checked by `forecast_select audit`.
- **Evaluation window:** the workbook ends in May 2026. It cannot support a complete six-month evaluation beginning January 2026 plus five subsequent months. The milestone is not claimed.
- **Anonymous/revised data:** no release lag, vintage, units, revision policy, or economic meaning were supplied. The repository uses `configs/data/availability.yaml`, a one-month conservative research lag, and labels the study `revised_data_pseudo_out_of_sample`.
- **Ties:** the official target is `1` only for `X(t+1) > X(t)`; exact ties are `0` and are separately recorded as `zero_change`.
- **Duplicate plan items:** the plan repeats the Chronos-2 zero-shot line in its architecture diagram. It is treated as one experiment family and not duplicated.

## Claims intentionally not made

This repository does not claim real-time vintage validity, causal interpretation, production readiness, a 65% milestone, compensation eligibility, or a scored June 2026 outcome. Anonymous IDs are not assigned economic labels.

