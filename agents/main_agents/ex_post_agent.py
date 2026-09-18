from typing import Dict, Any
from core.llm_provider import BaseLLMProvider

class ExPostAgent:
    """
    عامل ارزیابی پسینی برای سنجش ارزش‌آفرینی واقعی پس از خاتمه پروژه.
    """
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm_provider = llm_provider

    def evaluate_ex_post(self, actual_cost: float, realized_benefit: float) -> Dict[str, Any]:
        """
        محاسبه بازگشت سرمایه واقعی (Realized ROI) و تحلیل کیفی دستاوردها.
        """
        net_value = realized_benefit - actual_cost
        realized_roi = (net_value / max(actual_cost, 1.0)) * 100

        prompt = f"Ex-post value evaluation: Actual Cost = {actual_cost}, Realized Benefit = {realized_benefit}"
        llm_assessment = self.llm_provider.generate_json(prompt, schema_description="Ex-post evaluation schema")

        return {
            "actual_cost": actual_cost,
            "realized_benefit": realized_benefit,
            "net_value": round(net_value, 2),
            "realized_roi": round(realized_roi, 2),
            "value_certificate_status": "APPROVED" if realized_roi > 0 else "UNDER_REVIEW",
            "summary": llm_assessment.get("message", "ارزیابی پسینی با موفقیت تکمیل شد.")
        }