# Forecast Select — Project Delivery Report

**تاريخ التقرير:** 4 آب 2026

**الإصدار:** 1.0.0

**النموذج الفعّال:** Uptrend Selector

## 1. الملخص التنفيذي

هذا المشروع عبارة عن مسار بحثي بلغة Python لتوقع الاتجاه الشهري التالي لـ50
مؤشراً مجهول الهوية (`X1` إلى `X50`). لا يحاول النظام تقديم توقع متوازن لكل
حالات الصعود والهبوط؛ وظيفته الحالية هي ترتيب المؤشرات واختيار أفضل 15 مؤشراً
يُقدّر أنها سترتفع في الشهر التالي.

النتيجة المسجلة للنموذج الفعّال على فترة Selection هي:

| المقياس | النتيجة |
|---|---:|
| الأشهر | 100 |
| الاختيارات الشهرية | 15 |
| مجموع التوقعات | 1,500 |
| التوقعات الصحيحة | 926 |
| Accuracy | 61.73% |
| توقعات Up | 1,500 |
| توقعات Down | 0 |

المشروع يثبت أن هناك إشارة تاريخية مفيدة لاختيار حالات الصعود، لكنه لا يقدم
دليلاً كافياً للوصول بثبات إلى Accuracy أعلى من 65% مع فرض 15 اختياراً كل شهر.
كما أنه ليس نظام تداول حي، ولا يدّعي Production readiness أو real-time vintage
validity.

## 2. ما الذي تم إنجازه؟

تم إنجاز العناصر التالية:

1. تثبيت عقد واضح لملف البيانات والتحقق من التواريخ والأعمدة والقيم الرقمية.
2. بناء Target شهري سببي: `Up` إذا كانت قيمة الشهر التالي أكبر من الشهر الحالي،
   وإلا `Down`، مع احتساب التعادل كـ`Down`.
3. بناء Features لا تستخدم معلومات مستقبلية، تشمل Momentum وVolatility وBreadth
   وPCA وPeer Correlation وMarket Regime.
4. بناء Logistic Regression عالمي يتعلم من جميع المؤشرات مع الاحتفاظ بهوية كل
   مؤشر عبر One-hot encoding.
5. تطبيق Walk-forward training وإعادة التدريب لكل Forecast origin.
6. بناء Signed Correlation Graph مجمّدة من الفترة التاريخية حتى الموضع 119.
7. دمج احتمال النموذج مع trailing Up-rate لكل مؤشر، ثم اختيار أفضل 15.
8. حفظ التنبؤات بصيغة Parquet مع hashes للبيانات والإعدادات ومعلومات provenance.
9. حفظ Locked Evaluation مستقلة والتحقق من ثبات SHA-256 الخاص بها.
10. مقارنة النموذج مع عدة عائلات مرجعية.
11. تنفيذ دراسة جدوى الوصول إلى 65% ودراسة خاصة بالصدمات والهبوط المفاجئ.
12. تنفيذ تجربتي Downside Risk Gate وContextual Defensive Selector من دون
    ترقيتهما إلى النموذج الفعّال عندما لم تثبتا تحسناً مستقراً.
13. إضافة اختبارات Unit وIntegration وRegression واختبارات خاصة بمنع Leakage.

## 3. البيانات وحدودها

ملف الإدخال هو [`data/monthly_indicators.xlsx`](data/monthly_indicators.xlsx).

| الخاصية | القيمة |
|---|---|
| Worksheet | `Sheet1` |
| الفترة | شباط 2000 — أيار 2026 |
| عدد الأشهر | 316 |
| عدد المؤشرات | 50 |
| أسماء المؤشرات | `X1` إلى `X50` |
| صفوف Target المتاحة | 11,977 |
| Source SHA-256 | `8f9fc27ae0a33f4a25d1241b7d896b56ba7515d4ad7e984999f0c5fe42b20d29` |

يتم الحفاظ على القيم المفقودة في بداية بعض السلاسل، ولا يستخدم المشروع
interpolation أو backfill.

الحدود الأساسية للبيانات:

- أسماء المؤشرات الحقيقية ووحداتها غير متوفرة.
- أوقات إصدار المؤشرات غير متوفرة.
- Historical vintages غير متوفرة.
- البيانات شهرية ولا تُظهر الحركة اليومية داخل الشهر.
- لا توجد بيانات Options أو Volatility forward-looking أو Credit spreads أو
  Liquidity أو Volume أو News.
- لذلك الدراسة pseudo-out-of-sample على revised data وليست real-time backtest.

## 4. آلية عمل النموذج

```text
Excel workbook
      |
      v
Data validation and target construction
      |
      v
Past-only causal features
      |
      v
Walk-forward structured Logistic Regression
      |
      v
Frozen signed correlation propagation
      |
      v
50% model/graph probability + 50% trailing 48-label Up-rate
      |
      v
Rank indicators and select the top 15 Up candidates
```

### 4.1 حد المعلومات الزمني

عند إصدار توقع عند الموضع `t`:

- Features تستخدم مشاهدات متاحة حتى `t-1`.
- أحدث Target مسموح استخدامه في التدريب هو Target عند `t-2`.
- لا تُستخدم قيمة الشهر القادم في بناء Feature أو اختيار النموذج.

هذه القاعدة مغطاة باختبارات تغيّر بيانات المستقبل وتتأكد أن Features والتوقعات
الأقدم لا تتغير.

### 4.2 الـFeatures الحالية

تشمل أهم المتغيرات:

- تغير الشهر الماضي ونسبة التغير.
- اتجاهات وتغيرات متأخرة لعدة أشهر.
- Momentum لفترات 3 و6 و9 و12 شهراً.
- Rolling mean وstandard deviation وMAD وrobust z-score.
- المسافة عن المتوسطات المتحركة.
- Cross-sectional rank وbreadth وdispersion.
- Rolling PCA factors وloadings.
- Peer-correlation strength وdirection consensus.
- Rolling market breadth وregime dispersion وvolatility.
- تاريخ توافر السلسلة وفترات ثبات القيمة.

### 4.3 النموذج والاختيار

- Logistic Regression منتظم بـL2 و`C=0.25`.
- Missing numeric values تُعالج بالـmedian ثم تُقاس بـStandardScaler.
- Indicator identity تدخل عبر One-hot encoding.
- Signed Correlation Graph تستخدم وزن `0.35` ومصفوفة مجمّدة حتى الموضع 119.
- الاحتمال النهائي للترتيب يمزج نتيجة النموذج مع Up-rate آخر 48 Target متاحاً.
- يتم اختيار 15 مؤشراً فريداً كل شهر.

## 5. تصميم التقييم

| الفترة | Origins | الاستخدام |
|---|---:|---|
| Historical warm-up | 1–119 | بناء التاريخ الأولي والـfeatures والgraph |
| Selection / Discovery | 120–219 | النتيجة المسجلة وتطوير الفرضيات |
| Confirmation للتجارب الدفاعية | 220–266 | اختبار واحد بعد تثبيت قواعد التجربة |
| Historical locked evidence | 268–315 | دليل تاريخي محفوظ، غير مستخدم لترقية النموذج |
| آخر صف | 316 | أيار 2026؛ لا يوجد Target تالٍ داخل الملف |

ملف Locked Evaluation محفوظ byte-for-byte، وSHA-256 الخاص به هو:

```text
04ebedf9455051b189486f61deba949299a499915aa33f11b7126efa5a035b39
```

## 6. الأداء والمقارنات

| النموذج أو السياسة | الفترة والنطاق | Accuracy |
|---|---|---:|
| Uptrend Selector | Discovery، أفضل 15 | 61.73% |
| Rolling 60-month Up-rate | Discovery، أفضل 15 | 62.40% |
| Rolling 60-month Up-rate | Confirmation، أفضل 15 | 60.69% |
| Weighted Ensemble | Discovery، أفضل 15 | 61.40% |
| Lead-Lag Logistic | Discovery، أفضل 15 | 60.93% |
| Market Regime Selector | Discovery، أفضل 15 | 60.13% |
| Prequential meta-selector | Discovery، أفضل 15 | 60.87% |
| Bidirectional meta-selector | Discovery، أفضل 15 | 58.60% |

أفضل بديل بسيط على Discovery، وهو Rolling 60-month Up-rate، لم يحافظ على أدائه
عند نقله إلى Confirmation. هذه نتيجة مهمة: تحسين Discovery وحده لا يكفي لإثبات
أن نموذجاً جديداً أفضل.

## 7. لماذا لم نصل إلى 65%؟

### 7.1 الحساب المباشر

للوصول إلى 65% على 1,500 توقع نحتاج:

```text
1,500 × 65% = 975 إصابة
```

النتيجة الحالية 926 إصابة، أي نحتاج **49 إصابة إضافية** من دون خسارة إصابات
حالية.

### 7.2 المشكلة الأساسية: الانعكاسات

التحليل قسّم اختيارات النموذج إلى:

| نوع الحالة | العدد | Accuracy |
|---|---:|---:|
| استمرار السلوك السابق | 777 | 72.97% |
| انعكاس بالنسبة إلى آخر اتجاه متاح | 723 | 49.65% |

النموذج يتعامل جيداً مع استمرار الاتجاه، لكنه يصبح قريباً من التخمين العشوائي
عندما ينقلب الاتجاه. إذا بقي أداء حالات الاستمرار ثابتاً، يجب رفع Accuracy حالات
الانعكاس تقريباً من 49.65% إلى 56.4% للوصول إلى 65% إجمالاً.

### 7.3 الهبوط والصعود المفاجئان

كل الاختيارات الحالية `Up`، لذلك:

- الهبوط المفاجئ لمؤشر مختار يتحول مباشرة إلى توقع خاطئ.
- الصعود المفاجئ لمؤشر غير مختار يمثل فرصة ضائعة لاختيار أفضل، لكنه لا يظهر
  كخطأ مباشر في صفوف المؤشرات المختارة.
- في نموذج متوازن يتوقع الاتجاهين، الصعود المفاجئ بعد هبوط والهبوط المفاجئ بعد
  صعود هما المشكلة نفسها: توقيت الانعكاس.

يمكن تقدير خطر الهبوط، لكن توقيت أول صدمة أصعب بكثير من اكتشاف استمرار موجة
ضغط بدأت فعلاً. الدراسة وجدت:

- معدل الصدمة التالية بعد شهر غير مصدوم: 4.22%.
- معدل الصدمة التالية بعد صدمة: 20.45%، أي أعلى بنحو 4.85 مرات.
- AUC توقع بداية صدمة جديدة مع تأخير البيانات الرسمي: 0.596 تقريباً.
- النطاق الزمني بالـbootstrap كان واسعاً: P05=0.521 وP95=0.670.

هذا يعني أن clustering موجود، لكن توقيت أول انهيار غير ثابت بما يكفي.

### 7.4 الإشارة ضعيفة والصفوف ليست مستقلة

- AUC النموذج الأساسي على الصفوف الكاملة يقارب 0.548.
- ارتباط Selection score بصحة الاختيار ضعيف جداً.
- المؤشرات داخل الشهر تتحرك معاً؛ لذلك 1,500 صف ليست 1,500 تجربة مستقلة.
- التقدير التقريبي لحجم العينة الفعّال كان نحو 285 فقط بعد احتساب الترابط داخل
  الشهر.
- أداء النموذج هبط من 64.80% في النصف الأول من Discovery إلى 58.67% في النصف
  الثاني، ما يشير إلى drift أو winner's curse.

### 7.5 فرض 15 اختياراً

إجبار النظام على اختيار 15 مؤشراً يضيف حالات ضعيفة. نتائج التغطية المنخفضة كانت
أفضل رقمياً:

| التغطية | Discovery | Confirmation | الملاحظة |
|---:|---:|---:|---|
| Top-1 | 66.00% | 75.00% | تحليل post-hoc، وليس دليلاً نهائياً |
| Top-2 | 64.50% | 67.71% | تحليل post-hoc |
| Top-5 | 62.60% | 65.00% | تحليل post-hoc |
| Top-15 | 62.40% | 60.69% | المرشح المجمد |

النتائج توحي بأن رفع Accuracy ممكن عند خفض Coverage، لكن حدود الاختيار اختُبرت
بعد رؤية النتائج، ولذلك تحتاج إلى تقييم مستقبلي مستقل قبل اعتمادها.

## 8. التجارب الدفاعية

### 8.1 Downside Risk Gate

التجربة تبني Logistic specialist للصدمات ثم تخصم Risk score من ترتيب مرشحي
الصعود، من دون قلب القرار إلى `Down`.

| المقياس | النتيجة |
|---|---:|
| Penalty المختار على Discovery | 0.0 |
| Discovery base/gated | 61.73% / 61.73% |
| Confirmation base | 61.84% |
| Confirmation gated | 61.99% |
| الاختيارات المتغيرة في Confirmation | 1 |
| Confirmation shock ROC AUC | 0.564 |

التحسن نقطة واحدة فقط، والـpenalty المختار كان صفراً. لذلك التجربة موثقة كدليل
سلبي ولم تُرقَّ إلى النموذج الفعّال.

### 8.2 Contextual Defensive Selector

التجربة تستبدل أضعف اختيار بأدوار محايدة (`X44` و`X49`) عندما يكون متوسط
Breadth لثلاثة أشهر أقل من أو يساوي 0.45.

| المقياس | النتيجة |
|---|---:|
| Discovery base | 61.73% |
| Discovery contextual | 62.27% |
| Confirmation base | 61.84% |
| Confirmation contextual | 61.84% |

تحسن Discovery لم يتكرر على Confirmation، لذلك لم تتم ترقية التجربة.

## 9. هل يمكن حل المشكلة بالـFeature Engineering؟

الـFeature Engineering يمكن أن يستخرج إشارة موجودة في البيانات، لكنه لا يستطيع
خلق معلومات غير موجودة. مع ذلك، توجد حزمة محدودة تستحق اختباراً منضبطاً:

1. طول سلسلة الارتفاعات والانخفاضات.
2. `P(Up | previous Up)` و`P(Up | previous Down)` مع Bayesian shrinkage.
3. Momentum معدل بالتقلب وتسارع Momentum بين الآجال القصيرة والطويلة.
4. Drawdown depth وdrawdown speed والمسافة عن rolling extrema.
5. تغير Cross-sectional rank بدلاً من الرتبة الحالية فقط.
6. Breadth impulse وDispersion change وvolatility-of-volatility.
7. PCA residual momentum بعد إزالة الحركة العامة.
8. Correlation breakdown وتغير الارتباط، بدلاً من مستوى الارتباط فقط.
9. Features لاستمرار الصدمة بعد بدايتها.

يجب إضافة كل عائلة منفردة، ثم تنفيذ Ablation وWalk-forward evaluation واختيارها
داخل Discovery فقط. إضافة مئات التركيبات أو Polynomial features سترفع خطر
Overfitting بسبب صغر الحجم الزمني الفعّال.

## 10. الحلول المقترحة حسب الأولوية

### الأولوية الأولى: السماح بالامتناع

تغيير السياسة من اختيار 15 دائماً إلى اختيار `0–15` حسب حد ثقة مجمّد. هذا هو
المسار الأكثر واقعية لرفع Accuracy، لكنه يقلل Coverage.

### الأولوية الثانية: بيانات قيادية جديدة

إضافة بيانات يومية أو أسبوعية وPoint-in-time مثل:

- VIX وVIX term structure وSKEW.
- High Yield وInvestment Grade OAS.
- Financial Stress وFunding/Liquidity spreads.
- Volume وAdvance/Decline وdaily breadth.
- Yield curve وeconomic surprises.
- News أو sentiment وrelease calendars.

هذه البيانات قد تساعد في توقيت الانعكاس؛ إعادة تركيب البيانات الشهرية نفسها
لن تخلق إشارة خبر أو سيولة غير موجودة.

### الأولوية الثالثة: نظام قرار من مرحلتين

```text
Uptrend ranker
      |
      v
Downside-risk veto / abstention gate
      |
      v
Accept only sufficiently safe candidates
```

### الأولوية الرابعة: تغيير الـTarget

يمكن فصل `Strong Up` و`Neutral` و`Strong Down` أو تجاهل الحركات الأصغر من حد
محدد نسبة إلى التقلب. هذا يقلل ضجيج الحركات الصغيرة، مع انخفاض عدد القرارات.

### الأولوية الخامسة: تقييم جديد فعلياً

بعد تثبيت كل القرارات، يجب استخدام بيانات مستقبلية لم تُشاهد أثناء البحث. لا
يجوز إعادة استخدام Confirmation أو Locked results لاختيار Features أو
Thresholds جديدة ثم وصف النتيجة بأنها مستقلة.

## 11. Forecasts المخفية: حزيران وتموز وآب 2026

### 11.1 تعريف التوقع

تم توليد هذه التوقعات من ملف البيانات الظاهر الذي ينتهي في 29 أيار 2026. لم
تُقرأ نتائج الأشهر المخفية، ولا توجد أعمدة `Actual` أو `Correct` في النتائج.

| الشهر المستهدف | الاتجاه المتوقع | Horizon | نطاق النموذج |
|---|---|---:|---|
| حزيران 2026 | أيار → حزيران | 1 | Uptrend Selector المسجل |
| تموز 2026 | حزيران → تموز | 2 | Direct-horizon تجريبي |
| آب 2026 | تموز → آب | 3 | Direct-horizon تجريبي |

النموذج المسجل هو One-step-ahead، لذلك توقع حزيران يقع ضمن نطاقه الرسمي. لتوقع
تموز وآب من دون معرفة قيم الأشهر الوسيطة، تم تدريب نموذج مستقل لكل Horizon على
Targets تاريخية مماثلة. لم يتم تنفيذ Recursive forecasting ولم تُخترع قيم
لحزيران أو تموز.

إعداد المشروع يستخدم `availability_lag_months: 1`. لذلك، رغم أن ملف المصدر
ينتهي في أيار، فإن Features الداخلة في Forecast مبنية على المعلومات حتى 30
نيسان 2026. تم الحفاظ على هذا الشرط لأنه جزء من التصميم المسجل، بدلاً من تغييره
خصيصاً عند إصدار التوقع المخفي.

النتائج الكاملة القابلة للقراءة الآلية موجودة في
[`reports/next_three_month_forecast.json`](reports/next_three_month_forecast.json).

### 11.2 توقع حزيران 2026 — Official one-step scope

| Rank | Indicator | Direction | Selection score |
|---:|---|---|---:|
| 1 | X41 | Up | 0.744345 |
| 2 | X39 | Up | 0.646190 |
| 3 | X40 | Up | 0.625797 |
| 4 | X24 | Up | 0.623855 |
| 5 | X9 | Up | 0.618364 |
| 6 | X11 | Up | 0.611217 |
| 7 | X10 | Up | 0.606437 |
| 8 | X43 | Up | 0.598228 |
| 9 | X38 | Up | 0.592190 |
| 10 | X30 | Up | 0.589075 |
| 11 | X3 | Up | 0.575062 |
| 12 | X32 | Up | 0.570954 |
| 13 | X29 | Up | 0.568382 |
| 14 | X33 | Up | 0.564671 |
| 15 | X27 | Up | 0.561647 |

### 11.3 توقع تموز 2026 — Experimental direct horizon-2

| Rank | Indicator | Direction | Selection score |
|---:|---|---|---:|
| 1 | X41 | Up | 0.723804 |
| 2 | X39 | Up | 0.657285 |
| 3 | X24 | Up | 0.639955 |
| 4 | X40 | Up | 0.636543 |
| 5 | X43 | Up | 0.617263 |
| 6 | X9 | Up | 0.616626 |
| 7 | X10 | Up | 0.613821 |
| 8 | X38 | Up | 0.613221 |
| 9 | X30 | Up | 0.612654 |
| 10 | X29 | Up | 0.608631 |
| 11 | X11 | Up | 0.608560 |
| 12 | X32 | Up | 0.607297 |
| 13 | X34 | Up | 0.607236 |
| 14 | X48 | Up | 0.587333 |
| 15 | X37 | Up | 0.587250 |

### 11.4 توقع آب 2026 — Experimental direct horizon-3

| Rank | Indicator | Direction | Selection score |
|---:|---|---|---:|
| 1 | X41 | Up | 0.726538 |
| 2 | X39 | Up | 0.669711 |
| 3 | X24 | Up | 0.654576 |
| 4 | X40 | Up | 0.652232 |
| 5 | X34 | Up | 0.651177 |
| 6 | X36 | Up | 0.648989 |
| 7 | X38 | Up | 0.638993 |
| 8 | X43 | Up | 0.638571 |
| 9 | X30 | Up | 0.631894 |
| 10 | X9 | Up | 0.628918 |
| 11 | X10 | Up | 0.627404 |
| 12 | X11 | Up | 0.620069 |
| 13 | X32 | Up | 0.619257 |
| 14 | X29 | Up | 0.617068 |
| 15 | X37 | Up | 0.609302 |

### 11.5 قراءة النتيجة قبل كشف الإجابات

- كل الاختيارات الـ45 بقيت `Up`، بما يتوافق مع سلوك Uptrend Selector التاريخي.
- هناك 12 مؤشراً مشتركاً بين الأشهر الثلاثة: `X9` و`X10` و`X11` و`X24` و`X29`
  و`X30` و`X32` و`X38` و`X39` و`X40` و`X41` و`X43`.
- `X41` احتل المرتبة الأولى في الأشهر الثلاثة، يليه `X39` ضمن المرتبتين
  الأوليين.
- Selection score أداة ترتيب ممزوجة وليست ضماناً أو احتمال صحة معايراً.
- لا يمكن حساب Accuracy قبل كشف الدكتور للاتجاه الحقيقي لكل مؤشر في الأشهر
  الثلاثة. بعد الكشف يجب تجميد هذا الملف أولاً، ثم إضافة النتائج ومقارنة 45
  توقعاً من دون تعديل النموذج أو الـthresholds.

## 12. التشغيل وإعادة الإنتاج

```powershell
python -m pip install -e ".[dev]"
python -m forecast_select audit-data
python -m forecast_select build-model
python -m forecast_select show-results
python -m forecast_select build-risk-gate
python -m forecast_select show-risk-gate
python -m forecast_select build-context-selector
python -m forecast_select show-context-selector
python -m forecast_select forecast-next-three
python -m forecast_select check-project
python -m pytest
```

الأمر `forecast-next-three` يعيد تدريب Horizons الثلاثة من البيانات الظاهرة ثم
يكتب النتيجة إلى `reports/next_three_month_forecast.json`. ويمكن استدعاء المسار
برمجياً:

```python
from pathlib import Path
from forecast_select.future_forecast import build_direct_monthly_forecasts

predictions = build_direct_monthly_forecasts(
    Path(".").resolve(),
    forecast_origin=316,
    horizons=(1, 2, 3),
)
```

## 13. حالة التحقق عند التسليم

- إطار الاختبارات: pytest.
- CI: GitHub Actions على Python 3.12.
- تم تشغيل `python -m pytest -q` بتاريخ 4 آب 2026: **36 اختباراً نجحت من دون
  أي فشل**.
- الاختبارات تغطي Targets وFeatures والنموذج والاختيار وعقد OOF وسلامة
  Artifacts ومنع تسرب المستقبل.
- تحقق سلامة المشروع: `python -m forecast_select check-project`.
- تمت مقارنة Horizon-1 الجديد مع مخرجات Uptrend Selector التاريخية عند Origin
  315: تطابق 15/15 مؤشراً وتطابقت Selection scores ضمن دقة رقمية `5e-7`.
- تم التحقق من أن Artifact المخفي يحتوي 3 أشهر و45 اختياراً ولا يحتوي أي
  `Actual` أو `y_true`.
- ظهر تحذير Deprecation من إضافة `pytest-asyncio` المثبتة في بيئة التشغيل؛ لا
  يمثل فشلاً في المشروع ولا يؤثر على نتيجة الاختبارات الحالية.

## 14. الخلاصة والقرار

المشروع ناجح كمسار بحثي سببي ومنظم، والنموذج يقدم Ranking مفيداً لحالات الصعود،
لكن النتيجة الحالية لا تبرر ادعاء 65% أو Production forecasting.

المشكلة الجوهرية ليست نقص تعقيد النموذج فقط؛ هي نقص معلومات مستقرة لتوقيت
الانعكاسات مع بيانات شهرية مجهولة وفرض 15 اختياراً. أفضل اتجاه تطوير هو:

1. تجميد النموذج الحالي كـbenchmark.
2. اختبار حزمة صغيرة من Reversal/Persistence features عبر Ablation.
3. السماح بـ`0–15` اختياراً مع Abstention threshold مجمّد.
4. إضافة بيانات يومية وقيادية وPoint-in-time.
5. استخدام Risk Gate كـveto بدلاً من ادعاء توقع Down كامل.
6. تقييم النظام على بيانات مستقبلية جديدة لم تُستخدم في أي قرار تطوير.

الوصول إلى 65% مع 15 اختياراً ثابتاً ومن البيانات الحالية فقط هو **No-Go
حالياً** بحسب الأدلة المتوفرة. الوصول إليها مع Coverage أقل أو بيانات قيادية
جديدة يبقى فرضية قابلة للاختبار، وليس نتيجة مثبتة بعد.

## 15. المراجع داخل المستودع

- [`README.md`](README.md): ملخص المشروع وأوامر التشغيل.
- [`docs/data.md`](docs/data.md): عقد البيانات وحدودها.
- [`docs/methodology.md`](docs/methodology.md): منهجية النموذج والفصل الزمني.
- [`docs/verification.md`](docs/verification.md): قواعد سلامة الـartifacts.
- [`reports/model_performance.json`](reports/model_performance.json): نتيجة النموذج
  الفعّال القابلة للقراءة الآلية.
- [`research/accuracy_feasibility/README.md`](research/accuracy_feasibility/README.md):
  دراسة جدوى الوصول إلى 65%.
- [`research/sudden_drop_study/README.md`](research/sudden_drop_study/README.md):
  تحليل الصدمات والهبوط المفاجئ.
- [`research/downside_risk_gate/README.md`](research/downside_risk_gate/README.md):
  تجربة Downside Risk Gate.
- [`research/contextual_defensive_selector/README.md`](research/contextual_defensive_selector/README.md):
  تجربة Contextual Defensive Selector.
