# شرح المشروع

المشروع يتوقع اتجاه 50 مؤشرًا شهريًا مجهول الاسم.

النموذج الفعّال اسمه **Uptrend Selector** لأنه متخصص باختيار المؤشرات التي
يعتقد أنها سترتفع في الشهر القادم.

مساره:

```text
Excel
  -> ميزات تاريخية لا تستخدم المستقبل
  -> Structured Logistic
  -> Correlation Graph
  -> ترتيب المؤشرات
  -> اختيار أفضل 15 مؤشرًا
```

النتيجة المسجلة:

- 100 شهر
- 15 مؤشرًا كل شهر
- 926 إصابة من 1500
- Accuracy تساوي 61.73%
- جميع الاختيارات كانت Up

المشروع منظم حول نموذج فعّال واحد. النماذج الأخرى موجودة فقط كملفات مقارنة
داخل `research/reference_models/`، بنسخة واحدة واضحة من كل عائلة.

الأوامر:

```powershell
python -m forecast_select audit-data
python -m forecast_select build-model
python -m forecast_select show-results
python -m forecast_select check-project
python -m pytest
```

المشروع لا يدّعي الوصول إلى 65%، ولا يدّعي أنه يتنبأ بالهبوط بشكل جيد حاليًا.
