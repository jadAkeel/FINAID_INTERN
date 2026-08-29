# Regime Adaptive Selection Group v2 - البحث الثاني لتحسين Selection

**الحالة:** REJECTED - لا يوجد Challenger جاهز للتطبيق.

## الملفات المحتفظ بها (بعد التنظيف)
- `research_plan.md` - خطة 3 عائلات
- `diagnosis.json` - تشخيص Baseline
- `candidate_comparison.csv` - مقارنة cap-matched لكل Challenger
- `group_ablation.csv` - تدرج raw->graph->group_old->group_new
- `temporal_fold_metrics.csv` - 4 folds زمنية
- `bootstrap_results.json` (+ `_summary.csv`) - block bootstrap 6m 500
- `finalize.py` - كود التجارب الثلاث النهائي
- `final_report_ar.md` + `final_report_comprehensive_ar.md` - التقارير النهائية
- `README.md` - هذا الملف
- `archive_intermediate/` - كل grids وتقارير وسيطة مؤرشفة (لا تستخدم)

## الملفات المؤرشفة
كل `family_*.py` و `family_*_grid.csv` و `family_*_report.json` نُقلت إلى `archive_intermediate/` لمنع التكرار.

## النتيجة
أفضل Challenger `Family3 Pairwise LogitLead` حسن SelValidation `0.4015->0.4628 (+0.061)` لكن بقي `<0.50` و Dir انخفض `-0.016` و Acc `-0.15pp` - **مرفوض**.
أفضل نتيجة استكشافية `logit-0.5*dis+1.0*lead` وصلت `0.5113 (+0.11)` لكن Acc `-0.59pp` - مرفوضة أيضاً.

## لا تغيير على الموديل الفعّال
`configs/active_model.yaml` و `artifacts/active/` لم تمس.

## السجل المرجعي للفشل
راجع `docs/SELECTION_GROUP_FAILED_REGISTRY.md` و `research/FAILED_SELECTION_GROUP_GUIDE.md` قبل أي تجربة جديدة.
