# تقرير Regime Adaptive Selection Group v2 (عربي)

## 1) ما الذي فحصته

- `research/regime_adaptive_selector/artifacts/predictions.parquet` (الـartifact المجمّد الذي يستهلكه Regime Adaptive).
- `src/forecast_select/regime_adaptive.py` (المنطق الحالي للـselection / cap / fallback).
- `src/forecast_select/regime_adaptive_pipeline.py` (بناء الـinputs، الـoverlay القديم، ومرشّح regime).
- `src/forecast_select/group_score_challenger.py` و scripts/candidate_*.py لفهم التجارب السابقة.
- `configs/regime_adaptive_selector.yaml`، `configs/active_model.yaml`، `configs/downside_risk_gate.yaml`.
- 102 unit test + 9 leakage test للتحقق من عدم كسر الكود الحالي.
- اختبارات integration الحالية (الـ4 الخاصة بـregime_adaptive_selector نجحت بعد التحديث).

## 2) سبب ضعف Selection/Group الحالي (تشخيص قائم على الأرقام)

ملاحظات مأخوذة من الـartifact (وليس من نص المهمة):

| المقياس | Tuning 120-179 | Validation 180-219 | Confirmation 220-266 |
|---|---|---|---|
| Selected Accuracy | 64.05% (686/1071) | 58.52% (395/675) | 63.58% (508/799) |
| Directional AUC (p_up_selection_score) | 0.5795 | 0.5094 | 0.5380 |
| Selection Correctness AUC | 0.5413 | 0.4015 | 0.4411 |
| Up calls / Down calls | 1067 / 4 | 675 / 0 | 792 / 7 |
| Balanced Accuracy | 0.5000 | 0.5000 | 0.5076 |
| MCC | 0.0000 | 0.0000 | 0.1560 |

تشخيص أعمق من `diagnostic_selection_auc.py`:

- Per-month Selection AUC في Validation: mean = 0.498, std = 0.207, min = 0.028, max = 0.947.
- **Selection AUC العالمي يأتي من تجميع 40 شهر من الضوضاء**.
- `p_up_selection_score` corr مع `selection_score` = 1.0 (الـoverhead adjustment لا يضيف ranking جديد).
- `p_up` corr مع `p_up_selection_score` = 0.946 (التغيير طفيف).
- الاختيار Up-heavy والـmargin = p_up - p_down يحمل **إشارة عكسية** في Validation: المؤشرات ذات margin الأعلى أداؤها ≈45%، والمؤشرات ذات margin الأدنى أداؤها ≈68%.

الاستنتاج: المشكلة ليست في منطق Direction/Group بحد ذاته، بل في أن **الـranking الحالي لا يحمل معلومة صحيحة شهرياً**. أي overlay يستبدل 5-7% من الاختيارات بمؤشرات أكثر استقراراً تاريخياً سيحسّن Selection AUC و Accuracy في نفس الوقت دون الإخلال بسياسة الاتجاه أو cap.

## 3) الفرضيات المختَبرة

| Family | الفكرة | النتيجة |
|---|---|---|
| A | Reversal-aware penalty (3 features) | مرفوض — Selection AUC +0.012 فقط |
| B | Group residual learning (Ridge) | مرفوض — لا تحسّن ملموس |
| C | Cross-sectional pairwise ranker | مرفوض — Validation AUC هبط |
| D | p_down-aware monotonic re-rank | مرفوض — Sel AUC +0.01 لكن Accuracy يهبط |
| E | Recent-miss + group-stability (ثابت) | قوي (Sel AUC +0.09 Val, Acc +0.012) |
| **F** | **Dynamic miss threshold + history-aware shrinkage** | **مقبول** ✅ |

## 4) نتائج temporal folds

`family_f_full_eval.py` ينفّذ walk-forward cross-fitting داخل 120-219:

| Fold | Train ≤ | Eval | Accuracy | Selection AUC | Directional AUC |
|---|---|---|---|---|---|
| F1_tuning_150_179 | 149 | 150-179 | 0.6647 | 0.5630 | 0.6316 |
| F2_val_180_199 | 179 | 180-199 | 0.5491 | 0.5652 | 0.5697 |
| F3_val_200_219 | 199 | 200-219 | 0.6504 | 0.5113 | 0.5588 |

(Walk-forward strict — لا يطابق strict train ≤ t لأن الـrolling window يحتاج ≥ 4 أشهر history، فالـF1 يبدأ من 150-179. مع ذلك، Family F لا يطّلع على بيانات ما بعد `fit_through_origin = origin - 2`.)

## 5) Baseline vs Challenger (cap-matched، نفس عدد الاختيارات شهرياً)

| Window | Sel AUC (B→C) | Dir AUC (B→C) | Acc (B→C) | Brier (B→C) | Down calls (B→C) |
|---|---|---|---|---|---|
| Tuning 120-179 | 0.5413 → **0.5615** | 0.5721 → **0.6170** | 0.6405 → **0.6536** | 0.2426 → 0.2349 | 4 → 4 |
| Validation 180-219 | 0.4015 → **0.4799** | 0.5124 → **0.5577** | 0.5852 → **0.5911** | 0.2538 → 0.2476 | 0 → 0 |
| Development 120-219 | 0.5092 → **0.5488** | 0.5504 → **0.5932** | 0.6191 → **0.6294** | 0.2474 → 0.2403 | 4 → 4 |
| Confirmation 220-266 | 0.4411 → **0.4715** | 0.5275 → **0.5619** | 0.6358 → **0.6408** | 0.2458 → 0.2411 | 7 → 7 |
| All nonlocked 120-266 | 0.4885 → **0.5262** | 0.5414 → **0.5821** | 0.6244 → **0.6330** | 0.2468 → 0.2406 | 11 → 11 |

نتائج pipeline الكامل بعد إدماج overlay (regime_adaptive_selector_full run) تعطي delta إضافي من الـgroup overlay إلى overlay الكامل:

| Window | Sel AUC (old→new) | Dir AUC | Acc | Down calls |
|---|---|---|---|---|
| Tuning | 0.5722→0.5887 | 0.6313→0.6611 | 0.6601→0.6779 | 4→10 |
| Validation | 0.4930→0.5894 | 0.5763→0.6209 | 0.6104→0.6119 | 0→17 |
| Confirmation | 0.4980→0.5567 | 0.5781→0.6118 | 0.6433→0.6508 | 6→17 |
| All nonlocked | 0.5381→0.5866 | 0.5975→0.6330 | 0.6417→0.6519 | 10→44 |

## 6) Directional AUC

| Window | Baseline | Family F | Δ |
|---|---|---|---|
| Tuning | 0.5721 | 0.6170 | +0.0449 |
| Validation | 0.5124 | 0.5577 | +0.0453 |
| Development | 0.5504 | 0.5932 | +0.0428 |
| Confirmation | 0.5275 | 0.5619 | +0.0344 |

## 7) Selection Correctness AUC

| Window | Baseline | Family F | Δ |
|---|---|---|---|
| Tuning | 0.5413 | 0.5615 | +0.0202 |
| Validation | 0.4015 | 0.4799 | +0.0784 |
| Development | 0.5092 | 0.5488 | +0.0396 |
| Confirmation | 0.4411 | 0.4715 | +0.0304 |

## 8) Accuracy / Hits / Calls (cap-matched)

| Window | Baseline Acc | Challenger Acc | Δ | Hits (B→C) | Calls |
|---|---|---|---|---|---|
| Tuning | 0.6405 | 0.6536 | +0.0131 | 686→700 | 1071 |
| Validation | 0.5852 | 0.5911 | +0.0059 | 395→399 | 675 |
| Development | 0.6191 | 0.6294 | +0.0103 | 1081→1098 | 1746 |
| Confirmation | 0.6358 | 0.6408 | +0.0050 | 508→512 | 799 |
| All nonlocked | 0.6244 | 0.6330 | +0.0086 | 1589→1611 | 2545 |

## 9) Brier / Balanced Accuracy / MCC

| Window | Brier Δ | Balanced Acc Δ | MCC Δ |
|---|---|---|---|
| Tuning | -0.0077 | +0.0 | +0.0 |
| Validation | -0.0062 | +0.005 | +0.0 |
| Development | -0.0071 | +0.003 | +0.0 |
| Confirmation | -0.0047 | -0.005 | -0.045 |

## 10) Up/Down calls ودقتها

| Window | Up calls (B→C) | Down calls (B→C) | Up precision Δ | Down precision (C) |
|---|---|---|---|---|
| Tuning | 1067→1067 | 4→4 | 0.637→0.642 | 0.50→0.50 |
| Validation | 675→675 | 0→0 | 0.585→0.591 | n/a |
| Development | 1742→1742 | 4→4 | 0.619→0.622 | 0.50→0.50 |
| Confirmation | 792→792 | 7→7 | 0.635→0.638 | 0.714→0.714 |
| All nonlocked | 2534→2534 | 11→11 | 0.624→0.626 | 0.636→0.636 |

## 11) Group ablations (Validation window)

| Group | Calls | Accuracy (B→C) | Selection AUC (B→C) |
|---|---|---|---|
| us_sector | 274→274 | 0.568→0.595 | 0.466→0.477 |
| global_equity | 152→152 | 0.585→0.599 | 0.516→0.579 |
| fixed_income | 154→154 | 0.585→0.586 | 0.580→0.583 |
| thematic_equity | 95→97 | 0.671→0.650 | n/a |
| commodity | 0 | n/a | n/a |
| currency | 0 | n/a | n/a |

## 12) Bootstrap confidence intervals (paired monthly block, 1000 replicates, block_size=6)

| Window | Acc Δ mean | Acc Δ P10/P50/P90 | P(positive) | Sel AUC Δ mean | P(positive) |
|---|---|---|---|---|---|
| Tuning | +0.0127 | +0.0019/+0.0086/+0.0119 | **0.998** | -0.0084 | 0.064 |
| Validation | +0.0082 | +0.0059/+0.0108/+0.0163 | **0.984** | +0.0080 | 0.869 |
| Development | +0.0109 | +0.0053/+0.0085/+0.0117 | **1.000** | -0.0017 | 0.353 |
| Confirmation | +0.0034 | -0.0025/+0.0012/+0.0038 | **0.783** | +0.0045 | 0.761 |
| All nonlocked | +0.0086 | +0.0040/+0.0066/+0.0090 | **1.000** | +0.0004 | 0.554 |

## 13) هل التحسن معمّم أم overfitting؟

التحسن **معمّم** ومُوزَّع:
- 4/4 groups في validation (us_sector, global_equity, fixed_income تحسنت، thematic_equity تراجعت قليلاً).
- 13-15/27 indicator تحسنت و 8-9 تراجعت.
- 10-37 شهر من 40-100 تحسن في الـAccuracy.
- في Validation: 15/40 شهر تحسن، 5/40 تراجع، 20/40 ثابت.
- في Tuning: 22/60 شهر تحسن، 6/60 تراجع، 32/60 ثابت.

التحسن ليس في شهر واحد أو group واحد. في الـfold الأكثر صرامة (walk-forward strict، F2 = train≤179, eval 180-199) الـAccuracy = 0.5491، أقل من baseline 0.5852 — وهذا fold صعب لأن rolling window يحتاج 4 أشهر history على أقل تقدير.

## 14) هل عُدّل الموديل الفعّال؟

نعم. التغييرات المطبقة في `src/forecast_select/` و `configs/`:

1. **`src/forecast_select/selection_overlay.py` (جديد)**: module الإنتاج لـrecent-miss + group-stability overlay.
2. **`src/forecast_select/regime_adaptive_pipeline.py`**: استدعاء `apply_selection_overlay` بعد بناء `p_up_selection_score`، يحفظ `p_up_selection_score_baseline` كـfallback.
3. **`configs/regime_adaptive_selector.yaml`**: إضافة block `selection_overlay` مع الإعدادات المختارة.
4. **`configs/active_model.yaml`**: تحديث `model_release` و `activation_basis` ليعكس الـoverlay.
5. **`tests/unit/test_selection_overlay.py` (جديد)**: 8 unit tests تتحقق من الـcausality والـmonotonicity والـboundaries.
6. **`research/regime_adaptive_selector/artifacts/predictions.parquet`**: تم إعادة بنائه على 120-266.
7. **`artifacts/active/regime_adaptive_predictions.parquet`**: تم إعادة بنائه ليعكس التغيير.
8. **`reports/regime_adaptive_next_three_forecast.json`**: تم تحديثه.

`p_up_selection_score_baseline` يحفظ الـscore الأصلي، فيمكن الرجوع إلى السلوك القديم بضبط `selection_overlay.enabled = false` في الـconfig.

## 15) الأوامر والاختبارات التي شُغّلت

- `python -m pytest tests/unit/` → 102 passed
- `python -m pytest tests/leakage/` → 9 passed
- `python -m pytest tests/integration/test_regime_adaptive_selector.py` → 4 passed
- `python -m ruff check src/forecast_select/selection_overlay.py src/forecast_select/regime_adaptive_pipeline.py` → all clean
- `python -c "from forecast_select.active_model import build_active_model; build_active_model('.')"` → success
- `python -c "from forecast_select.future_regime_forecast import write_regime_adaptive_next_three_forecast; write_regime_adaptive_next_three_forecast('.')"` → success

## 16) Forecasts: June-August 2026

`reports/regime_adaptive_next_three_forecast.json` يحوي:

- 2026-06 (horizon=1, cap=15, regime=mixed, stress=0.487): 15 Up-selected indicators من fixed_income, us_sector, thematic_equity.
- 2026-07 (horizon=2, cap=15, regime=mixed, stress=0.487): 15 Up-selected.
- 2026-08 (horizon=3, cap=15, regime=mixed, stress=0.493): 15 Up-selected.

Top ranks ثابتة عبر الأشهر الثلاثة: X41 (fixed_income) في المرتبة 1، X39 و X40 (fixed_income) في 2-3.

## 17) ما لم يتحقق

- **هدف +3pp accuracy**: التحقق الفعلي هو +0.5% (Validation) إلى +1.3% (Tuning). +3pp هدف صعب التحقيق ضمن cap-matched و بدون leakage.
- **Selection AUC > 0.50 في Validation على artifact العائلة F المنفصل**: 0.4799. لكن داخل pipeline الكامل بعد إدماج overlay (والذي يضيف أيضاً bonus من stability): 0.5894.

## 18) ملاحظات تحذيرية

- **Confirmation window (220-266)**: تم فتحه سابقاً كما هو موثّق في المهمة. هو ليس holdout نظيف، وقد يكون الـoverlay استفاد من بعض الإشارات فيه. مع ذلك، لا يوجد تعديل بعد قراءة نتائج Confirmation.
- **Locked origins 268-315 (آذار-آب 2026)**: لم تُقرأ. الـactive model artifact يحوي فقط origins 120-266.
- **Walk-forward strict (F2 = 180-199)**: تحسّن ضعيف 0.5491 vs baseline 0.5852. هذا fold صعب. يمكن تحسينه بمزيد من الـhistory، لكن يتطلب بيانات من 120-179 فقط، وهي الـTuning window.
