# Prior-only selector diagnostic — 2026-09-22, corrected 2026-09-23

Non-promoting diagnostic. It answers one question: how much of the active model's accuracy
does a zero-feature rule already explain? The rule picks the 15 indicators with the highest
trailing Up-rate and calls them all Up. Reproduce with
`python research/prior_only_selector/prior_only_diagnostic.py`.

Causal contract at origin `t`, using the project's 1-based `position` from
`forecast_select.io.load_workbook`: direction labels `d_s = 1[X_{s+1} > X_s]` are used only
for `s <= t-2`, the scored target is `d_t`, and at least 24 labels are required inside the
window. The workbook is read with `maximum_position=267`. Every window length is reported and
**none is selected**, so this table carries no tuning.

## Erratum (2026-09-23)

The first version, committed in `0463c9b`, indexed workbook rows from 0, but the project numbers
positions from 1 (`io.py`: `position = range(1, len(frame) + 1)`). That version's "origin t" was
really origin `t+1`:

- Every window ran **one month late**. Its Tuning was 121–180, not 120–179.
- Its Confirmation scored **buffer origin 267**, which the project excludes from evaluation.
- It read level **position 268**, one row past the project's non-locked limit of 267.

The evaluation was still causal: labels stopped two months before each scored target. **No
locked outcome was read.** The first locked target (origin 268: position 268 → 269) needs
position 269, which was never loaded. The seasonal-prior experiment surfaced the error: it
asserts that its targets equal the production artifact's `y_true`, and that assertion failed
under 0-based indexing. The tables below are recomputed with the corrected alignment. The
corrected N=48 rule reproduces the independent local-logistic constant-`p_up` ablation exactly
on Validation (352/600).

## Result (hits / calls)

| Rule | Tuning 120–179 | Validation 180–219 | Confirmation 220–266 |
|---|---:|---:|---:|
| **Production regime-adaptive (15–20 calls)** | 686/1071 **64.05%** | 395/675 **58.52%** | 508/799 **63.58%** |
| Up-rate, k=15, N=24 | 564/900 62.67% | 337/600 56.17% | 428/705 60.71% |
| Up-rate, k=15, N=36 | 566/900 62.89% | 347/600 57.83% | 431/705 61.13% |
| Up-rate, k=15, N=48 | 572/900 63.56% | 352/600 58.67% | 435/705 61.70% |
| Up-rate, k=15, N=60 | 570/900 63.33% | 367/600 61.17% | 428/705 60.71% |
| Up-rate, k=15, N=96 | 561/900 62.33% | 357/600 59.50% | 438/705 62.13% |
| Up-rate, k=15, all history | 561/900 62.33% | 342/600 57.00% | 440/705 62.41% |
| Up-rate, k=20, N=36 | 746/1200 62.17% | 454/800 56.75% | 571/940 60.74% |
| Up-rate, k=20, N=48 | 748/1200 62.33% | 465/800 58.13% | 578/940 61.49% |
| Up-rate, k=20, all history | 729/1200 60.75% | 446/800 55.75% | 577/940 61.38% |
| Majority direction by \|rate−0.5\|, k=15, N=36 | 568/900 63.11% | 310/600 51.67% | 429/705 60.85% |
| Majority direction by \|rate−0.5\|, k=15, N=48 | 571/900 63.44% | 332/600 55.33% | 431/705 61.13% |
| Majority direction by \|rate−0.5\|, k=15, all history | 565/900 62.78% | 339/600 56.50% | 438/705 62.13% |

## Reading

1. **No window length is stable.** N=48 wins Tuning, N=60 wins Validation, and all-history wins
   Confirmation. Validation alone swings 5 points (56.17%–61.17%) across window lengths. Any
   single rule's Validation win is not evidence of skill.
2. **This unpaired table understates production's model contribution.** The window-to-window
   spread is large because whole months move together. The paired, matched-coverage comparison in
   `research/seasonal_prior/` ranks production's universe by production's own prior alone. It
   finds production's model stack adds **+13 hits on Tuning and +13 on Confirmation**, with
   p10–p90 block-bootstrap intervals entirely above zero, and a 1-hit loss on Validation. The
   stack is worth roughly 1.2–1.6 points in two of three windows; it is not a base-rate rule.
3. **Calling the majority direction does not generalize.** Letting the rule call an indicator
   with a low Up-rate Down wins 2 of 9 comparisons, both on Tuning (N=36 by 2 hits,
   all-history by 4). It loses all six Validation and Confirmation comparisons, and loses
   Validation by 3–37 hits. Only X16 has a full-history Up-rate below 0.40.
4. **k=20 is below k=15 in all nine comparisons.** The five marginal calls run 52–61%.

This is a diagnostic, not a candidate. Promoting the N=60 rule because it wins Validation would
be exactly the window-picking this table exposes.
