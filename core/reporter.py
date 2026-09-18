from typing import Dict, Any
import datetime

class EvaluationReporter:
    """
    ماژول تولید گزارش‌های مدیریتی و گواهی‌های ارزش‌آفرینی پروژه‌ها.
    """
    
    @staticmethod
    def generate_ex_ante_report(proposal_data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> str:
        """
        تولید متن گزارش رسمی ارزیابی پیشینی پروپوزال جهت دانلود یا چاپ.
        """
        metrics = evaluation_result.get("quantitative_metrics", {})
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report_content = f"""==================================================
        گزارش رسمی ارزیابی پیشینی پروپوزال (سامانه پژوهشیار)
==================================================
تاریخ صدور: {now_str}
عنوان پروپوزال: {proposal_data.get('title', 'بدون عنوان')}

۱. ورودی‌های مالی و فنی:
--------------------------------------------------
- تخمین هزینه (خوش‌بینانه / محتمل / بدبینانه): {proposal_data.get('cost_p10')} / {proposal_data.get('cost_p50')} / {proposal_data.get('cost_p90')} میلیون تومان
- تخمین منافع (خوش‌بینانه / محتمل / بدبینانه): {proposal_data.get('benefit_p10')} / {proposal_data.get('benefit_p50')} / {proposal_data.get('benefit_p90')} میلیون تومان
- احتمال موفقیت فنی: {int(proposal_data.get('p_success', 0) * 100)}%

۲. نتایج شبیه‌سازی آماری (مونت‌کارلو):
--------------------------------------------------
- میانگین نرخ بازگشت سرمایه (ROI): {metrics.get('mean_roi')}%
- شاخص ارزش در معرض ریسک (VaR 95%): {metrics.get('var_95')}%
- احتمال زیان مالی: {metrics.get('probability_of_loss')}%
- سطح ریسک ارزیابی‌شده: {evaluation_result.get('confidence_level_status')}

۳. خلاصه ارزیابی و توصیه سیستم:
--------------------------------------------------
پروژه از نظر شاخص‌های سودآوری و ریسک فنی مورد تحلیل قرار گرفت. 
توصیه می‌شود مصوبه بر اساس الگوی مدیریت ریسک و تعریف مایلستون‌های فازبندی‌شده ابلاغ گردد.
==================================================
"""
        return report_content