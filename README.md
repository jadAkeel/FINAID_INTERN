# Forecast Select — Regime Adaptive Bidirectional Selector

Leakage-safe research pipeline للتنبؤ باتجاه 50 مؤشر شهري مجهول (X1..X50). الموديل الفعّال الحالي هو **Regime Adaptive Bidirectional Selector** بسياسة `forward_breadth_graduated_15_to_20` — يختار **15-20** مؤشر شهرياً (Up/Down) باستخدام `t-2` labels و `t-1` features فقط.

> **الخلاصة العملية (2026-08-29):** بعد 164 محاولة cap-matched (13 فكرة + 120 تركيبة خطية + 30 recent_miss + 1 غير خطية) لم يتجاوز أي Challenger `SelAUC>0.50` مع `Acc` لا تنخفض. أفضل نتيجة كانت `Sel 0.4689 (+0.067)` لكن بقي `<0.50`. **السقف الخطي مع البيانات الحالية هو ~0.47** — لذلك الموديل الفعّال بقي `forward_breadth_dynamic_cap_v3` بدون تغيير. كل الفشل موثق في `docs/SELECTION_GROUP_FAILED_REGISTRY.md`.

## النتائج المثبتة (Baseline)

من `research/regime_adaptive_selector/artifacts/predictions.parquet` (6349 صف 120-266):

| Period | Directional AUC | Selection AUC | Accuracy | Correct/Total | Cap |
|---|---:|---:|---:|---:|---:|
| Tuning 120-179 | 0.5795 | 0.5413 | 64.05% | 686/1071 | 15-20 |
| Validation 180-219 | 0.5094 | 0.4015 | 58.52% | 395/675 | 15-20 |
| Development 120-219 | 0.5533 | 0.5092 | 61.91% | 1081/1746 | 17.46 avg |
| Confirmation 220-266 | 0.5380 | 0.4411 | 63.58% | 508/799 | 17.0 |

- `Up 675 / Down 0` في Validation — تحيز Up شديد
- `corr(p_up,p_up_selection)=0.946` — تكرار

## البنية

```
configs/
  active_model.yaml               # الفعّال: forward_breadth_dynamic_cap_v3
  regime_adaptive_selector.yaml   # graduated 0.52->15 / 0.68->20, group weight 0.25
  config.yaml                     # data_path: data/monthly_indicators.xlsx (316 pos حتى 2026-05-29)
src/forecast_select/
  regime_adaptive.py              # risk_adjusted = p_up*(1-0.25*stress)-0.15*p_down
  regime_adaptive_pipeline.py     # group 12m lag2 + graph 48m
research/
  regime_adaptive_selector/       # Baseline المرجع
  february_holdout_experiment/    # مارس-مايو Terminal (مو Blind الآن)
  regime_adaptive_selection_group_v2/ # آخر بحث (9 ملفات نظيفة)
docs/
  SELECTION_GROUP_FAILED_REGISTRY.md # سجل 13 فكرة فاشلة + 150 تركيبة
reports/
  regime_adaptive_next_three_forecast.json # توقع يونيو-أغسطس
```

## التشغيل السريع (CLI)

```powershell
# 1. تثبيت
python -m pip install -e ".[dev]"

# 2. فحص البيانات
python -m forecast_select.cli audit-data

# 3. بناء الموديل الفعّال (حتى مايو 316)
python -m forecast_select.cli build-regime-adaptive

# 4. عرض النتائج
python -m forecast_select.cli show-regime-adaptive

# 5. توقع يونيو (H1) + يوليو (H2) + أغسطس (H3) - كل أفق يتدرب حتى t-2-(h-1)
python -m forecast_select.cli forecast-regime-next-three
# alias:
python -m forecast_select.cli forecast-next-three

# 6. التحقق
python -m forecast_select.cli check-project
python -m pytest tests/unit/test_regime_adaptive.py -q
```

`forecast-regime-next-three` يكتب `reports/regime_adaptive_next_three_forecast.json` — كل أفق يختار 15-20 مؤشر مع `p_up`, `p_down`, `selection_score`, `group`, `rank`, `regime`.

## نتيجة التشغيل الحالية (2026-06 / 07 / 08)

شُغلت الآن على `2026-05-29` Origin 316:

```
Generated from: 2026-05-29 | Origin: 316 | Through: 2026-04-30
Method: direct_multi_horizon_frozen_regime_adaptive | Cap mode: guarded_bidirectional_fallback

=== 2026-06 | Horizon 1m | mixed stress 0.487 cap 15 breadth 0.531 ===
   1. X41 Up score 0.715 p_up 0.748 p_down 0.367 group fixed_income
   2. X39 Up score 0.600 p_up 0.649 p_down 0.506 group fixed_income
   3. X40 Up score 0.590 p_up 0.627 p_down 0.438 group fixed_income
   4. X24 Up score 0.552 p_up 0.625 p_down 0.404 group us_sector
   5. X9  Up score 0.546 p_up 0.621 p_down 0.523 group thematic_equity
   6. X11 Up score 0.547 p_up 0.614 p_down 0.470 group thematic_equity
   7. X10 Up score 0.546 p_up 0.610 p_down 0.456 group thematic_equity
   8. X43 Up score 0.560 p_up 0.601 p_down 0.468 group fixed_income
   9. X38 Up score 0.539 p_up 0.593 p_down 0.554 group fixed_income
  10. X30 Up score 0.497 p_up 0.591 p_down 0.539 group us_sector
  11. X3  Up score 0.512 p_up 0.576 p_down 0.455 group thematic_equity
  12. X32 Up score 0.515 p_up 0.574 p_down 0.532 group global_equity
  13. X33 Up score 0.510 p_up 0.567 p_down 0.514 group global_equity
  14. X37 Up score 0.493 p_up 0.557 p_down 0.560 group global_equity
  15. X34 Up score 0.487 p_up 0.544 p_down 0.516 group global_equity

=== 2026-07 | Horizon 2m | mixed stress 0.487 cap 15 breadth 0.531 ===
   1. X41 Up score 0.708 p_up 0.734 p_down 0.329 group fixed_income
   2. X39 Up score 0.631 p_up 0.659 p_down 0.368
   3. X24 Up score 0.577 p_up 0.641 p_down 0.344
  ... (15 كل شهر)

=== 2026-08 | Horizon 3m | mixed stress 0.493 cap 15 breadth 0.531 ===
   1. X41 Up score 0.698 p_up 0.743 p_down 0.446
  ... (15 كل شهر)
```

الملف الكامل: `reports/regime_adaptive_next_three_forecast.json` (15 اختيار لكل شهر 6/7/8 مع `forecast_market_breadth 0.530562` و `regime_stress 0.487`).

## التحقق

```powershell
python -m pytest tests/unit/test_regime_adaptive.py -q
# 19 passed

ruff check src/forecast_select --quiet
# (no output)
```

كل المقاييس cap-matched 15-20، `fit_through <= origin-2`، `locked 268-315` لم تُقرأ. مارس-مايو 314-316 شوهدت سابقاً في `february_holdout` لذلك ليست Blind الآن - موثق في `research_plan.md`.

## السجل المرجعي للفشل

قبل أي تجربة جديدة اقرأ:
- `docs/SELECTION_GROUP_FAILED_REGISTRY.md` — 13 فكرة فاشلة + 150 تركيبة exhaustive (الحد الأقصى Val Sel 0.4689 <0.50، Acc +0.29pp فقط، 0/120 تجاوزت البوابات)

الفرضية الوحيدة المقترحة المتبقية: `Regime-conditioned Lead` فقط عند `breadth<0` و `uncertainty>0.6` مع shrinkage هرمي.

## تسليم المراجع

```
research/regime_adaptive_selector/artifacts/predictions.parquet
research/regime_adaptive_selection_group_v2/ (9 ملفات نظيفة)
research/february_holdout_experiment/ (6 ملفات)
docs/SELECTION_GROUP_FAILED_REGISTRY.md
configs/active_model.yaml
reports/regime_adaptive_next_three_forecast.json
```
