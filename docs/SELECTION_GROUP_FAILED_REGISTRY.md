# سجل الأفكار الفاشلة - Selection Group & Accuracy & Selection Score
**الهدف:** منع تكرار تجارب أثبتت عدم تعميمها على Regime Adaptive Bidirectional Selector.
**آخر تحديث:** 2026-08-29
**قاعدة:** أي فكرة هنا لا تُعاد كما هي تحت اسم جديد. يجب تغيير فرضية علمية مختلفة كلياً.

## الخلاصة التنفيذية
- كل المحاولات من بداية المشروع حتى `2026-08-29` حسّنت Tuning لكن فشلت في Validation أو حافظت على SelAUC <0.50.
- أفضل نتيجة كانت `Family F: recent-miss + group-stability` بـ Sel Validation `0.4015->0.4979 (+0.096)` و Dev `0.553 (+0.043)` لكن بقيت `<0.50` وفشلت بوابة Accuracy/Dir.
- **لا يوجد Challenger جاهز للتطبيق** - النموذج الفعّال بقي `forward_breadth_dynamic_cap_v3`.

## Baseline المثبت (مرجع كل المقارنات)
- المسار: `research/regime_adaptive_selector/artifacts/predictions.parquet`
- DirAUC (كل المؤهلة بـ p_up_selection): Tuning `0.5795` Validation `0.5094` Development `0.5533` Confirmation `0.5380`
- SelAUC (المختارة بـ selection_score): Tuning `0.5413` Validation `0.4015` Development `0.5092` Confirmation `0.4411`
- Accuracy: Tuning `64.05%` 686/1071 Validation `58.52%` 395/675 Development `61.91%` 1081/1746 Confirmation `63.58%` 508/799
- Cap: 15-20 graduated `0.52->15 / 0.68->20` متوسط 17.3، Down Validation `0`
- Corr `p_up vs p_up_selection =0.946` - تكرار شديد

## جدول الأفكار الفاشلة (ممنوع تكرارها كما هي)

| # | الاسم | التاريخ | الفرضية | المعادلة | Features | Hyperparams | Tuning Sel | Validation Sel | Δ Sel Val | Validation Acc | Dir Val | لماذا فشلت؟ | مسار Artifact |
|---|-------|---------|---------|----------|----------|-------------|------------|---|---|---|---|---|---|
| 1 | **Group Overlay الحالي** | baseline | trailing 12 يتنبأ بالشهر القادم | `logit(p_up_sel)=logit(p_up_graph)+0.25*relative_logit` | `relative_logit = logit(group_12m)-logit(market_12m)` | weight 0.25 fixed | 0.5413 | 0.4015 | - | 58.52% | 0.5094 | lift Validation +0.0013 فقط، لا يعمم، يصف drift لا انعكاس | `configs/regime_adaptive_selector.yaml:47` |
| 2 | **Candidate A Hierarchical EB** | 2026-08 | EB لـ p_group يقلل ضوضاء المجموعات الصغيرة | `p_shrunk=(N*p_group+tau*p_market)/(N+tau)` | `N, p_group, p_market` | tau10 w0.75 (6*4 grid) | 0.541->0.583 (+0.04) | 0.5127 (+0.003) | +0.003 | 58.52%->58.52% | 0.5127 | تحسن صغير غير كاف، Dev Acc 61.91->61.80 -0.11pp | `research/february_holdout_experiment/candidate_a_grid.csv` |
| 3 | **Candidate B Cross-sectional Logistic** | 2026-08 | تعلم ترتيب عالمي لكل الصفوف | `logit(correct)= w*19features` | 19 feature (p_up, p_down, group, regime...) | C 0.05-1.0 | 0.606 (+0.06) | 0.488 (-0.02) | -0.02 | 48.4% (-10pp) | 0.488 | Overfitting واضح، Acc انهار | `family_b_grid.csv` |
| 4 | **Candidate C Selection Correction** | 2026-08 | تصحيح selection_score باحتمال الصحة | `correct~disagreement+stability+margin` | 7 features | ridge 1.0 | 0.578 | 0.439 (<0.5) | +0.038 | 58% | 0.439 | بقي <0.5، corrected 0.394 أسوأ | `candidate_c_report.json` |
| 5 | **Candidate D Reliability-Gated** | 2026-08 | وزن المجموعة حسب موثوقيتها الزمنية | `w_g = w*reliability(N,sign_stability)` | `N, sign_flip, std` | w 0.25+gate | 0.542 | 0.4026 (+0.001) | +0.001 | 57.63% (-0.9pp) | 0.5084 (-0.001) | مكسب هامشي وخسارة Dir/Acc | `family_d_grid.csv` |
| 6 | **Family1 GroupResidual Reliability** | 2026-08-29 | تعلم الباقي بعد p_up مع shrinkage | `new_p=logit(p_graph)+0.4*rel*relative_logit` | `rel = sqrt(N/15)*sqrt(months/24)*(1-flip)*exp(-std)` | weight 0.4 alpha10 | 0.5508 | 0.3996 (-0.002) | -0.002 | 58.52% | 0.4744 (-0.035) | Sel لم يتحسن، Dir انهار | `research/regime_adaptive_selection_group_v2/candidate_comparison.csv:2` |
| 7 | **Family2 Reversal Penalty Hierarchical** | 2026-08-29 | عقوبة انعكاس تطرح فقط | `penalty=(0.03*dis+0.02*breadth+0.02*uncertainty)*factor` | `dis, breadth_change, stress` | 3 weights + regime 0.8/1.0/1.1 | 0.5430 | 0.3964 (-0.005) | -0.005 | 58.22% | 0.4743 | Sel ساء، لا قيمة | `candidate_comparison.csv:3` |
| 8 | **Family3 Pairwise Ranker LogitLead** | 2026-08-29 | ترتيب pairwise داخل الشهر | `new_p=inv_logit(logit(p_graph)-0.5*dis+1.0*lead)` | `dis, lead, p_down` | C0.1 | 0.5578 | 0.4628 (+0.061) | +0.061 | 58.37% (-0.15pp) | 0.4932 (-0.016) | Sel<0.5 و Dir ساء رغم +0.06 | `candidate_comparison.csv:4` |
| 9 | **Family D PDown-Aware** | 2026-08-29 | إضافة p_down لتحسين Sel | `p_up+0.5*p_down` | `p_down` | w 0.5 | 0.539 | 0.479 (+0.07) | +0.07 | 57% | 0.48 | Sel<0.5، Brier تدهور 0.34 | `family_d_grid.csv` |
| 10 | **Family E Recent-Miss** | 2026-08-29 | معاقبة مؤشرات كثيرة الأخطاء 6 أشهر | `threshold 0.45 miss_rate` | `recent accuracy` | window6 thresh0.45 | 0.540 | 0.420 (+0.02) | +0.02 | 58% | 0.51 | مكسب صغير | `family_e_grid.csv` |
| 11 | **Family F Recent-Miss + Group-Stability (أفضل نتيجة)** | 2026-08-29 | معاقبة miss + مكافأة مجموعة مستقرة | `penalty miss>0.45 + bonus group_up>0.5 & low_var` | `miss_rate, group_up, group_var` | window6 bonus0.3 | 0.557 (+0.016) | **0.4979 (+0.096)** | **+0.096** | 58.67% (+0.15pp) | 0.554 (+0.044) | **أفضل Δ لكن بقي 0.4979<0.50** فشل بوابة 1 بـ 0.002 | `research/regime_adaptive_selection_group_v2/family_f_full_report.json` |
| 12 | **Logit Lead/Dis Grid (576 تركيبة استكشافية)** | 2026-08-29 | إضافة lead و dis في logit | `logit(p)+w1*p_down+w2*dis+w3*lead` | `p_down, dis, lead` | 576 grid | 0.5618 | **0.5113 (+0.11)** | **+0.11** | 57.93% (-0.59pp) | 0.520 (+0.01) | **تجاوز 0.5 لكن Acc -0.59pp فشل بوابة 4** | `family_f_grid.csv` |

## الدروس المستفادة (لماذا كلها فشلت؟)
1. **التكرار الشديد:** `corr 0.946` يعني group لا تضيف معلومات مستقلة - أي وزن ثابت يضيف ضوضاء.
2. **المجموعات الصغيرة:** commodity 4 مؤشرات ضوضاء عالية، بدون shrinkage هرمي قوي يخرب Validation.
3. **الانجراف vs الانعكاس:** `relative_logit` يصف ماضي 12 شهر (drift) لا يتنبأ بانعكاس الشهر القادم (مارس 10% Up).
4. **تحيز Up:** 675 Up /0 Down في Validation يمنع تعلم Down أو correctness متوازن.
5. **Overfitting السريع:** أي نموذج >3 params أو C>0.1 يصل Tuning 0.60 لكن Validation 0.48.
6. **Brier vs AUC:** تحسين SelAUC عبر إضافة p_down يحسن ترتيب لكن يدمر معايرة Brier (+0.08).

## ما المسموح وما الممنوع مستقبلاً
- **ممنوع:** إعادة أي صف في الجدول أعلاه كما هو تحت اسم جديد، أو Grid ضخم عشوائي، أو استخدام `314-316` لاختيار threshold.
- **مسموح:** تجربة واحدة جديدة فقط في كل دورة، ≤3 params، مع `temporal folds` حتى `219` فقط، و `cap-matched`، و `assert fit_through <= origin-2`.
- **الفرضية الوحيدة المقترحة حالياً:** `Regime-conditioned Lead Penalty` - استخدم `lead_negative_share` فقط عندما `breadth_change_3<0` و `regime_uncertainty>0.6` مع shrinkage `sqrt(n_regime/47)` و `isotonic` للحفاظ على Dir. (لم تجرب بعد كما هي).

## كيفية استخدام هذا السجل
1. قبل أي تجربة جديدة، اقرأ هذا الملف.
2. إذا كانت فكرتك تطابق أي صف، غيّر فرضية علمية مختلفة كلياً أو لا تنفذها.
3. حدّث هذا السجل بعد كل تجربة فاشلة جديدة (أضف صفاً).

## ???? 2026-08-29 - ????? ?????? 120 ??????
- **???????:** wg 0-0.6 * wl 0-1.0 * wd -0.5-0 * wp 0-0.2 ?? logit
- **???????:** ???? ?????? Val Sel  .4689 (+0.067) ??? <0.50? 0/120 ?????? ?????? 0.50 cap-matched? ???? ?????? Val Acc 58.81% (+0.29pp) ?? ???? +3pp
- **???????:** ??? ????? ????? ?????? ?? ???? - ????? feature ??? ??? ?? regime-conditioned
- **?????:** research/regime_adaptive_selection_group_v2/exhaustive_search_results.csv

## ??????? ??? ?????? ??????? ??????? - 2026-08-29 (????? ???? ??? ???)
- **???????:** lead ??? ??? readth_deterioration>0.05 ? uncertainty>0.6 ?? actor calm0.7/mixed1.0/stressed1.1 ? dis -0.3
- **????????:** 
ew_p=inv_logit(logit(p_graph)+w*lead*factor -0.3*dis) ??? w 0.3-1.3
- **???????:** ???? w=1.3 Val Sel  .4672 (+0.065) ??? <0.50? Tuning  .5199 (-0.02)? Isotonic  .465 - **????** ??? ????? ????
- **???????:** ??? ????? ??? ????? ?? ????   .50 cap-matched - ????? ????

## recent_miss ????? - 2026-08-29 (30 ??????)
- **???????:** th 0.40-0.50 * bonus 0.2-0.4 * penalty 0.3-0.5
- **???????:** ???? ?????? Val Sel  .4720 (+0.07) ??? <0.50? Val Acc 56.59% (-1.9pp)? 0/30 ?????? 0.50 cap-matched
- **???????:** ??? recent_miss ?? ???? 0.50 cap-matched ?? ??????? ?????? - ????? ????
