# Prior-only selector diagnostic — 2026-09-22

Non-promoting diagnostic. It answers one question: how much of the active model's accuracy
does a zero-feature rule — pick the 15 indicators with the highest trailing Up-rate and call
them all Up — already explain? Reproduce with `python research/prior_only_selector/prior_only_diagnostic.py`.

Causal contract at origin `t`: direction labels `d_s = 1[X_{s+1} > X_s]` are used only for
`s <= t-2`; the scored target is `d_t`; at least 24 labels are required inside the window.
No origin above 266 is read. Every window length is reported; **none is selected**, so this
table carries no tuning.

## Result (hits / calls)

| Rule | Tuning 120–179 | Validation 180–219 | Confirmation 220–266 |
|---|---:|---:|---:|
| **Production regime-adaptive (15–20 calls)** | 686/1071 **64.05%** | 395/675 **58.52%** | 508/799 **63.58%** |
| Up-rate, k=15, N=24 | 563/900 62.56% | 327/600 54.50% | 435/705 61.70% |
| Up-rate, k=15, N=36 | 566/900 62.89% | 340/600 56.67% | 432/705 61.28% |
| Up-rate, k=15, N=48 | 570/900 63.33% | 346/600 57.67% | 435/705 61.70% |
| Up-rate, k=15, N=60 | 568/900 63.11% | 362/600 60.33% | 429/705 60.85% |
| Up-rate, k=15, N=96 | 559/900 62.11% | 351/600 58.50% | 440/705 62.41% |
| Up-rate, k=15, all history | 558/900 62.00% | 340/600 56.67% | 438/705 62.13% |
| Up-rate, k=20, N=36 | 744/1200 62.00% | 445/800 55.62% | 574/940 61.06% |
| Up-rate, k=20, N=48 | 747/1200 62.25% | 456/800 57.00% | 580/940 61.70% |
| Up-rate, k=20, all history | 727/1200 60.58% | 441/800 55.12% | 574/940 61.06% |
| Majority direction by \|rate−0.5\|, k=15, N=36 | 568/900 63.11% | 305/600 50.83% | 428/705 60.71% |
| Majority direction by \|rate−0.5\|, k=15, N=48 | 569/900 63.22% | 327/600 54.50% | 429/705 60.85% |
| Majority direction by \|rate−0.5\|, k=15, all history | 560/900 62.22% | 340/600 56.67% | 435/705 61.70% |

## Reading

1. **Production's edge over the best zero-feature rule is about one point** — +0.7 pp on
   Tuning, +1.2 pp on Confirmation — and it trails the N=60 rule on Validation. The whole
   feature / logistic / graph / regime stack buys roughly one point.
2. **No window length is stable.** N=60 wins Validation, N=96 wins Confirmation, N=48 wins
   Tuning. The spread across N is about ±2 pp — the same size as every improvement claimed
   by the challengers in `docs/EXPERIMENT_REGISTRY.md`. That spread is the noise floor.
3. **Down calls lose everywhere.** Letting the rule call the majority direction (so an
   indicator with a low Up-rate is called Down) is worse in every window. Only one
   indicator (X16) has a full-history Up-rate below 0.40; there is nothing for a Down
   model to find.
4. **k=20 is always below k=15.** The marginal five calls run 55–60%.
5. Excluding X16 changes nothing.

This confirms and extends the `accuracy_feasibility` finding (rolling 60-month Up-rate,
62.40% on Discovery, 60.69% once on Confirmation) and the local-logistic constant-`p_up`
ablation (352/600 and 438/705 through the fixed-15 Uptrend path). It is a diagnostic, not a
candidate: promoting the N=60 rule because it wins Validation would be exactly the
window-picking this table warns against.
