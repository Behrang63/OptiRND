from typing import Dict, Any, List
from core.llm_provider import BaseLLMProvider

class ProcessAgent:
    """
    عامل پایش فرآیندی برای ردیابی انحرافات زمانی و بودجه‌ای مایلستون‌های پروژه.
    """
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    def monitor_process(self, milestone_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تحلیل انحراف بودجه و زمان مایلستون‌ها.
        """
        total_planned_budget = sum(m.get('planned_budget', 0) for m in milestone_data)
        total_actual_budget = sum(m.get('actual_budget', 0) for m in milestone_data)
        
        total_planned_days = sum(m.get('planned_duration_days', 0) for m in milestone_data)
        total_actual_days = sum(m.get('actual_duration_days', 0) for m in milestone_data)

        # محاسبه درصد انحرافات
        cost_variance_pct = ((total_actual_budget - total_planned_budget) / max(total_planned_budget, 1.0)) * 100
        schedule_variance_pct = ((total_actual_days - total_planned_days) / max(total_planned_days, 1)) * 100

        # تعیین وضعیت هشدار
        if cost_variance_pct > 15 or schedule_variance_pct > 20:
            status = "CRITICAL_VARIANCE"
        elif cost_variance_pct > 5 or schedule_variance_pct > 10:
            status = "WARNING"
        else:
            status = "ON_TRACK"

        # تحلیل کیفی از LLM
        prompt = f"Process monitoring alert: Cost Variance = {cost_variance_pct:.1f}%, Schedule Variance = {schedule_variance_pct:.1f}%"
        llm_advice = self.llm_provider.generate_json(prompt, schema_description="Process monitoring schema")

        return {
            "status": status,
            "cost_variance_pct": round(cost_variance_pct, 2),
            "schedule_variance_pct": round(schedule_variance_pct, 2),
            "mitigation_advice": llm_advice.get("mitigation_advice", "اقدام اصلاحی خاصی ثبت نشده است.")
        }