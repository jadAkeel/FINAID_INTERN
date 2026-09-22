### Top-15 selector accuracy by period

| Candidate | Tuning | Validation | Confirmation |
| --- | --- | --- | --- |
| A. Global Logistic (baseline) | 579/900 (64.33%) | 347/600 (57.83%) | 436/705 (61.84%) |
| B. Pure Local Logistic | 577/900 (64.11%) | 354/600 (59.00%) | 436/705 (61.84%) |
| C. Global + Local fixed shrinkage | 578/900 (64.22%) | 355/600 (59.17%) | 435/705 (61.70%) |
| D. Global + Local sample-aware shrinkage | 574/900 (63.78%) | 350/600 (58.33%) | 435/705 (61.70%) |
| E. Global + indicator interactions | 570/900 (63.33%) | 351/600 (58.50%) | 438/705 (62.13%) |

### Top-15 hit delta versus the Global baseline

| Candidate | Tuning | Validation | Confirmation |
| --- | --- | --- | --- |
| B. Pure Local Logistic | -2 hits (-0.22 pp) | +7 hits (+1.17 pp) | +0 hits (+0.00 pp) |
| C. Global + Local fixed shrinkage | -1 hits (-0.11 pp) | +8 hits (+1.33 pp) | -1 hits (-0.14 pp) |
| D. Global + Local sample-aware shrinkage | -5 hits (-0.56 pp) | +3 hits (+0.50 pp) | -1 hits (-0.14 pp) |
| E. Global + indicator interactions | -9 hits (-1.00 pp) | +4 hits (+0.67 pp) | +2 hits (+0.28 pp) |

### Monthly selector accuracy dispersion

| Candidate | Period | Months | Monthly mean | Monthly median | Monthly SD |
| --- | --- | --- | --- | --- | --- |
| A. Global Logistic (baseline) | tuning | 60 | 64.33% | 66.67% | 0.2912 |
| A. Global Logistic (baseline) | validation | 40 | 57.83% | 63.33% | 0.2834 |
| A. Global Logistic (baseline) | confirmation | 47 | 61.84% | 80.00% | 0.3430 |
| B. Pure Local Logistic | tuning | 60 | 64.11% | 70.00% | 0.2813 |
| B. Pure Local Logistic | validation | 40 | 59.00% | 63.33% | 0.3021 |
| B. Pure Local Logistic | confirmation | 47 | 61.84% | 80.00% | 0.3361 |
| C. Global + Local fixed shrinkage | tuning | 60 | 64.22% | 73.33% | 0.2852 |
| C. Global + Local fixed shrinkage | validation | 40 | 59.17% | 66.67% | 0.2913 |
| C. Global + Local fixed shrinkage | confirmation | 47 | 61.70% | 80.00% | 0.3381 |
| D. Global + Local sample-aware shrinkage | tuning | 60 | 63.78% | 66.67% | 0.2926 |
| D. Global + Local sample-aware shrinkage | validation | 40 | 58.33% | 66.67% | 0.2863 |
| D. Global + Local sample-aware shrinkage | confirmation | 47 | 61.70% | 80.00% | 0.3404 |
| E. Global + indicator interactions | tuning | 60 | 63.33% | 70.00% | 0.2890 |
| E. Global + indicator interactions | validation | 40 | 58.50% | 66.67% | 0.2999 |
| E. Global + indicator interactions | confirmation | 47 | 62.13% | 80.00% | 0.3445 |

### Raw `p_up` model quality (before graph and prior blending)

| Candidate | Period | Rows | Accuracy | AUC | Brier | Log loss | Up base rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A. Global Logistic (baseline) | tuning | 2395 | 57.87% | 0.5538 | 0.2557 | 0.7140 | 57.04% |
| A. Global Logistic (baseline) | validation | 1750 | 51.03% | 0.4963 | 0.2591 | 0.7134 | 53.77% |
| A. Global Logistic (baseline) | confirmation | 2204 | 55.17% | 0.5198 | 0.2496 | 0.6933 | 57.35% |
| B. Pure Local Logistic | tuning | 2395 | 57.20% | 0.5264 | 0.2488 | 0.6942 | 57.04% |
| B. Pure Local Logistic | validation | 1750 | 53.71% | 0.5292 | 0.2492 | 0.6918 | 53.77% |
| B. Pure Local Logistic | confirmation | 2204 | 54.76% | 0.5308 | 0.2470 | 0.6870 | 57.35% |
| C. Global + Local fixed shrinkage | tuning | 2395 | 57.83% | 0.5531 | 0.2503 | 0.6986 | 57.04% |
| C. Global + Local fixed shrinkage | validation | 1750 | 51.94% | 0.5117 | 0.2552 | 0.7044 | 53.77% |
| C. Global + Local fixed shrinkage | confirmation | 2204 | 55.17% | 0.5161 | 0.2499 | 0.6934 | 57.35% |
| D. Global + Local sample-aware shrinkage | tuning | 2395 | 57.91% | 0.5529 | 0.2551 | 0.7117 | 57.04% |
| D. Global + Local sample-aware shrinkage | validation | 1750 | 51.20% | 0.4998 | 0.2582 | 0.7110 | 53.77% |
| D. Global + Local sample-aware shrinkage | confirmation | 2204 | 55.26% | 0.5199 | 0.2493 | 0.6925 | 57.35% |
| E. Global + indicator interactions | tuning | 2395 | 57.29% | 0.5150 | 0.2559 | 0.7088 | 57.04% |
| E. Global + indicator interactions | validation | 1750 | 51.83% | 0.4891 | 0.2540 | 0.7025 | 53.77% |
| E. Global + indicator interactions | confirmation | 2204 | 55.99% | 0.5153 | 0.2473 | 0.6882 | 57.35% |

### Paired monthly comparison versus Global (circular block bootstrap, 6-month blocks, 500 replicates)

| Candidate | Period | Mean monthly delta | Better/worse/equal months | 90% interval | Excludes zero |
| --- | --- | --- | --- | --- | --- |
| B. Pure Local Logistic | tuning | -0.22 pp | 18/23/19 | [-2.22, +1.56] pp | no |
| B. Pure Local Logistic | validation | +1.17 pp | 9/4/27 | [+0.00, +2.33] pp | yes |
| B. Pure Local Logistic | confirmation | +0.00 pp | 11/7/29 | [-1.42, +1.42] pp | no |
| C. Global + Local fixed shrinkage | tuning | -0.11 pp | 11/15/34 | [-1.00, +0.89] pp | no |
| C. Global + Local fixed shrinkage | validation | +1.33 pp | 10/3/27 | [+0.50, +2.17] pp | yes |
| C. Global + Local fixed shrinkage | confirmation | -0.14 pp | 7/7/33 | [-1.28, +0.85] pp | no |
| D. Global + Local sample-aware shrinkage | tuning | -0.56 pp | 0/5/55 | [-0.78, -0.22] pp | yes |
| D. Global + Local sample-aware shrinkage | validation | +0.50 pp | 3/0/37 | [+0.00, +1.00] pp | no |
| D. Global + Local sample-aware shrinkage | confirmation | -0.14 pp | 1/2/44 | [-0.57, +0.28] pp | no |
| E. Global + indicator interactions | tuning | -1.00 pp | 8/17/35 | [-1.89, -0.22] pp | yes |
| E. Global + indicator interactions | validation | +0.67 pp | 7/4/29 | [-0.33, +1.67] pp | no |
| E. Global + indicator interactions | confirmation | +0.28 pp | 7/4/36 | [-0.28, +0.85] pp | no |

### Selection overlap with the Global baseline

| Candidate | Period | Mean overlap | Overlap % | Added calls correct | Removed calls correct | Net hits |
| --- | --- | --- | --- | --- | --- | --- |
| B. Pure Local Logistic | tuning | 11.95/15 | 79.67% | 105/183 (57.38%) | 107/183 (58.47%) | -2 |
| B. Pure Local Logistic | validation | 13.68/15 | 91.17% | 31/53 (58.49%) | 24/53 (45.28%) | +7 |
| B. Pure Local Logistic | confirmation | 12.68/15 | 84.54% | 68/109 (62.39%) | 68/109 (62.39%) | +0 |
| C. Global + Local fixed shrinkage | tuning | 13.75/15 | 91.67% | 43/75 (57.33%) | 44/75 (58.67%) | -1 |
| C. Global + Local fixed shrinkage | validation | 14.25/15 | 95.00% | 20/30 (66.67%) | 12/30 (40.00%) | +8 |
| C. Global + Local fixed shrinkage | confirmation | 13.77/15 | 91.77% | 33/58 (56.90%) | 34/58 (58.62%) | -1 |
| D. Global + Local sample-aware shrinkage | tuning | 14.83/15 | 98.89% | 4/10 (40.00%) | 9/10 (90.00%) | -5 |
| D. Global + Local sample-aware shrinkage | validation | 14.82/15 | 98.83% | 6/7 (85.71%) | 3/7 (42.86%) | +3 |
| D. Global + Local sample-aware shrinkage | confirmation | 14.72/15 | 98.16% | 8/13 (61.54%) | 9/13 (69.23%) | -1 |
| E. Global + indicator interactions | tuning | 13.75/15 | 91.67% | 39/75 (52.00%) | 48/75 (64.00%) | -9 |
| E. Global + indicator interactions | validation | 13.88/15 | 92.50% | 28/45 (62.22%) | 24/45 (53.33%) | +4 |
| E. Global + indicator interactions | confirmation | 13.36/15 | 89.08% | 46/77 (59.74%) | 44/77 (57.14%) | +2 |

### Local-model fallback frequency

| Measure | Value |
| --- | --- |
| Pure Local (B) | 13.86% |
| Shrinkage candidates (C, D) | 13.86% |
| Tuning | 20.54% |
| Validation | 11.20% |
| Confirmation | 8.71% |
| Mean local training rows | 131.1 |
| Min / max local training rows | 0 / 240 |
| Mean sample-aware weight | 0.0569 |

### Model size

| Measure | Count |
| --- | --- |
| global coefficients | 114 |
| interaction coefficients | 336 |
| interaction terms | 222 |
| local coefficients  core8 | 8 |

### Local coefficient stability

| Feature | Mean slope | Cross-indicator SD | Cross-indicator range | Within-indicator SD over origins | Within/between variance ratio | Sign consistency | Indicators +/- |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direction_lag_1 | -0.0657 | 0.1997 | [-0.4955, 0.3771] | 0.0696 | 0.12 | 91.91% | 16/30 |
| change_lag_1 | 0.1293 | 0.1861 | [-0.4230, 0.5574] | 0.0817 | 0.19 | 88.17% | 36/10 |
| momentum_3 | 0.0396 | 0.2256 | [-0.3916, 0.5926] | 0.0734 | 0.11 | 89.06% | 30/16 |
| momentum_6 | -0.0175 | 0.1945 | [-0.4013, 0.4436] | 0.0694 | 0.13 | 88.75% | 21/25 |
| rolling_std_12 | 0.0005 | 0.1842 | [-0.3759, 0.4813] | 0.0751 | 0.17 | 86.38% | 25/21 |
| robust_z_12 | 0.0483 | 0.2148 | [-0.3830, 0.5887] | 0.0864 | 0.16 | 88.02% | 25/21 |
| distance_mean_12 | -0.1123 | 0.1370 | [-0.3718, 0.2408] | 0.0699 | 0.26 | 86.91% | 11/35 |
| cross_section_rank | -0.0228 | 0.1524 | [-0.3142, 0.3063] | 0.0752 | 0.24 | 87.70% | 21/25 |

### Per-indicator raw accuracy, out-of-sample (validation + confirmation): five worst and five best for local modelling

| Indicator | Rows | Mean local history | Global | Pure local | Hybrid | Interaction | Local - Global | Selected global/hybrid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X33 | 87 | 195 | 56.41% | 44.65% | 54.97% | 58.54% | -11.76 pp | 32 / 30 |
| X30 | 87 | 195 | 56.04% | 48.59% | 53.91% | 49.60% | -7.45 pp | 0 / 0 |
| X44 | 87 | 134 | 55.35% | 49.47% | 51.78% | 55.53% | -5.88 pp | 0 / 1 |
| X31 | 87 | 195 | 54.41% | 49.79% | 50.85% | 53.16% | -4.63 pp | 26 / 26 |
| X39 | 87 | 195 | 66.36% | 61.91% | 66.36% | 64.04% | -4.44 pp | 87 / 87 |
| X46 | 87 | 142 | 47.34% | 55.03% | 50.21% | 43.59% | +7.69 pp | 0 / 0 |
| X17 | 87 | 142 | 49.47% | 57.47% | 62.29% | 51.78% | +8.01 pp | 0 / 0 |
| X47 | 87 | 148 | 46.41% | 55.85% | 38.22% | 45.16% | +9.44 pp | 0 / 0 |
| X36 | 87 | 195 | 49.10% | 61.04% | 47.66% | 52.66% | +11.94 pp | 19 / 23 |
| X48 | 87 | 142 | 40.59% | 58.16% | 49.28% | 42.15% | +17.58 pp | 0 / 0 |

### Tuning search, best five rows per family

| Family | Feature set | C | Min rows | w | Hits/calls | Accuracy | vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_shrinkage | core8 | 0.25 | 60 | 0.40 | 578/900 | 64.22% | -1 |
| fixed_shrinkage | core8 | 0.10 | 36 | 0.30 | 576/900 | 64.00% | -3 |
| fixed_shrinkage | core8 | 0.25 | 36 | 0.30 | 576/900 | 64.00% | -3 |
| fixed_shrinkage | core8 | 0.25 | 36 | 0.50 | 576/900 | 64.00% | -3 |
| fixed_shrinkage | core8 | 0.25 | 36 | 0.40 | 576/900 | 64.00% | -3 |
| interaction | nan | 0.01 | 0 | n/a | 570/900 | 63.33% | -9 |
| interaction | nan | 0.05 | 0 | n/a | 565/900 | 62.78% | -14 |
| interaction | nan | 0.10 | 0 | n/a | 563/900 | 62.56% | -16 |
| interaction | nan | 0.25 | 0 | n/a | 559/900 | 62.11% | -20 |
| pure_local | core8 | 0.01 | 60 | 1.00 | 577/900 | 64.11% | -2 |
| pure_local | lean5 | 0.01 | 60 | 1.00 | 573/900 | 63.67% | -6 |
| pure_local | core8 | 0.01 | 36 | 1.00 | 573/900 | 63.67% | -6 |
| pure_local | lean5 | 0.01 | 36 | 1.00 | 571/900 | 63.44% | -8 |
| pure_local | core8 | 0.01 | 48 | 1.00 | 571/900 | 63.44% | -8 |

### Frozen configuration

```json
{
  "baseline": {
    "accuracy": 0.6433333333333333,
    "calls": 900,
    "down_calls": 0,
    "hits": 579,
    "monthly_mean": 0.6433333333333332,
    "monthly_median": 0.6666666666666666,
    "monthly_std": 0.29119534153265597,
    "months": 60,
    "up_calls": 900
  },
  "fixed_shrinkage": {
    "feature_set": "core8",
    "local_weight": 0.4,
    "logistic_c": 0.25,
    "minimum_local_rows": 60,
    "tuning_accuracy": 0.6422222222222222
  },
  "interaction": {
    "features": [
      "momentum_3",
      "momentum_6",
      "direction_lag_1",
      "rolling_std_12",
      "distance_mean_12",
      "cross_section_rank"
    ],
    "logistic_c": 0.01,
    "tuning_accuracy": 0.6333333333333333
  },
  "pure_local": {
    "feature_set": "core8",
    "local_weight": 1.0,
    "logistic_c": 0.01,
    "minimum_local_rows": 60,
    "tuning_accuracy": 0.6411111111111111
  },
  "sample_aware_shrinkage": {
    "feature_set": "core8",
    "logistic_c": 0.25,
    "maximum_local_weight": 0.1,
    "minimum_local_rows": 60,
    "reference_rows": 180,
    "tuning_accuracy": 0.6377777777777778
  },
  "tuning_window": [
    120,
    179
  ]
}
```

### Locked-set status

- Locked origins: `[268, 315]`
- Maximum workbook row position read: `267`
- Maximum origin evaluated: `266`
- Locked origins read: `False`

