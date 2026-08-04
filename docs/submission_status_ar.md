# حالة المشروع للتسليم وطلب بيانات جديدة

## الملخص التنفيذي

المشروع أصبح نسخة بحثية قابلة لإعادة التشغيل والتدقيق، مع نموذج فعّال وتجارب
دفاعية منفصلة وMeta-Controller موحّد. لم يتم ترقية أي تجربة غير مثبتة إلى
النموذج الفعّال، ولم يتم استخدام مجموعة Locked Evaluation في الاختيار أو
الـtuning.

المشروع لا يثبت حالياً دقة 65% مستقرة. النتيجة المسجلة للنموذج الفعّال هي:

- `926 / 1500` اختياراً صحيحاً.
- `61.73%` على Selection/Discovery.
- اختيار 15 مؤشراً شهرياً.
- النموذج الفعّال هو `Uptrend Selector`، وتوقعاته كلها Up.

## ما تم إنجازه

### البيانات والعقد

- تثبيت عقد ملف `data/monthly_indicators.xlsx`.
- 316 شهراً و50 مؤشراً مجهول الاسم، من شباط 2000 حتى أيار 2026.
- بناء Targets للشهر التالي مع التعامل مع التعادل كـDown.
- عدم استخدام interpolation أو backfill للقيم المفقودة.

### السلامة المنهجية

- Features سببية تستخدم البيانات المتاحة فقط قبل لحظة التنبؤ.
- Walk-forward training.
- منع استخدام Targets غير المتاحة زمنياً.
- اختبارات مخصصة لمنع Data Leakage.
- Locked Evaluation محفوظة byte-for-byte ولا تستخدم للترقية.

### النموذج الأساسي

- Structured Logistic Regression.
- Momentum وvolatility وbreadth وPCA وpeer correlation.
- Frozen signed correlation graph.
- اختيار أفضل 15 مؤشراً كل شهر.
- حفظ predictions بصيغة Parquet مع hashes وprovenance.

### التجارب الدفاعية

| التجربة | النتيجة الحالية | القرار |
|---|---:|---|
| Uptrend Selector | 61.73%، `926/1500` | النموذج الفعّال |
| Downside Risk Gate | 61.99% في Confirmation، تغيير اختيار واحد | غير مروّج |
| Directional Downside Selector | 61.84% في Confirmation؛ Down: `6/12` | غير مروّج |
| Contextual Defensive Selector | 62.27% Discovery، و61.84% Confirmation | غير مروّج |
| Unified Forecast Controller | 61.84% Confirmation، دون تحسن | غير مروّج |

الـUnified Controller يستخدم Directional Downside كأساس، ثم يختبر تأثير Risk
Percentile وContextual Stress/Role Bonus، من دون تغيير النموذج الفعّال. الأوزان
التي اختارها Tuning كانت صفراً، ما يعني أن الدمج الحالي لم يثبت قيمة إضافية.

## ما لم يُحسم بعد

- هل المطلوب هو توقع Up/Down لكل المؤشرات الخمسين، أم اختيار أفضل 15 فقط؟
- هل قياس 65% يُحسب على كل القرارات أم على الاختيارات المقبولة فقط؟
- لا يوجد تقييم نهائي للنموذج الحالي على فترة حديثة كاملة تبدأ من يناير 2026.
- البيانات الحالية لا تحتوي أسماء المؤشرات أو وحداتها أو أوقات الإصدار والمراجعة.
- البيانات شهرية ولا تظهر الحركة داخل الشهر أو الإشارات اليومية السابقة للصدمات.

## البيانات المطلوبة للمرحلة التالية

نحتاج من الجهة المالكة للبيانات:

1. أسماء المؤشرات، الفئة، الوحدة، العملة، والمنطقة الجغرافية لكل `X1..X50`.
2. تاريخ ووقت توفر كل ملاحظة للمستخدم، وليس فقط تاريخ الفترة التي تصفها.
3. Historical vintages أو revision history لمعرفة ما كان متاحاً لحظة التنبؤ.
4. بيانات يومية أو أسبوعية، ويفضل معها volume وbreadth وadvance/decline.
5. مؤشرات leading إضافية مثل VIX وterm structure وcredit spreads وliquidity وfinancial-stress measures.
6. توضيح فترة التقييم الرسمية، وعدد القرارات الشهري، وتعريف النجاح عند 65%.
7. القيم الفعلية حتى نهاية فترة التقييم المطلوبة، حتى يمكن حساب Confirmation وPersistence بشكل صحيح.

## رسالة جاهزة لطلب البيانات

نحتاج نسخة محدثة من بيانات المؤشرات مع metadata كاملة لكل سلسلة، بما يشمل
الاسم والفئة والوحدة والمنطقة وتاريخ الإصدار وrevision history. نحتاج أيضاً
بيانات يومية أو أسبوعية وleading indicators مرتبطة بالمخاطر النظامية، إضافة إلى
توضيح رسمي لتعريف الـaccuracy: هل المطلوب توقع كل المؤشرات Up/Down أم اختيار
عدد محدد منها؟ كما نحتاج actual outcomes للفترة المطلوبة حتى نقيس شرط 65%
وفترة الثبات اللاحقة دون استخدام بيانات مستقبلية.

## أوامر التسليم

```powershell
python -m pip install -e ".[dev]"
python -m forecast_select audit-data
python -m forecast_select build-model
python -m forecast_select build-risk-gate
python -m forecast_select build-directional-downside
python -m forecast_select build-context-selector
python -m forecast_select build-unified-controller
python -m forecast_select check-project
python -m pytest
```
