import sys
import re
from pathlib import Path
from typing import Dict, Any, List

# افزودن مسیر ریشه پروژه
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.md_reporter import MarkdownReportGenerator


class MarkdownQualityEvaluator:
    """
    ماژول ارزیابی کیفی و ساختاری فایل‌های گزارش Markdown
    """
    REQUIRED_SECTIONS = [
        "۱. اطلاعات عمومی و شناسه پروژه",
        "۲. ورودی‌های مالی و شرایط اقتصاد کلان",
        "۳. ارزیابی ریسک‌های عملیاتی",
        "۵. خروجی شبیه‌سازی مونت‌کارلو",
        "🤖 راهنمای پرامپت هوش مصنوعی"
    ]

    @classmethod
    def evaluate_content(cls, md_text: str) -> Dict[str, Any]:
        """
        تحلیل خط‌به‌خط و اعتبارسنجی کیفیت فایل Markdown
        """
        issues = []
        checks = {}

        # ۱. بررسی وجود بخش‌های الزامی
        missing_sections = [sec for sec in cls.REQUIRED_SECTIONS if sec not in md_text]
        checks["required_sections_present"] = len(missing_sections) == 0
        if missing_sections:
            issues.append(f"بخش‌های زیر در گزارش یافت نشدند: {missing_sections}")

        # ۲. بررسی داده‌های نامعتبر (None یا NaN)
        invalid_patterns = re.findall(r'\b(None|NaN|undefined|null)\b', md_text, re.IGNORECASE)
        checks["no_null_or_nan"] = len(invalid_patterns) == 0
        if invalid_patterns:
            issues.append(f"مقادیر نامعتبر در متن گزارش شناسایی شد: {set(invalid_patterns)}")

        # ۳. بررسی ساختار جداول Markdown
        table_lines = [line for line in md_text.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
        checks["valid_tables"] = len(table_lines) >= 3
        if len(table_lines) < 3:
            issues.append("تعداد ردیف‌های جداول ساختاریافته کمتر از حد استاندارد است.")

        # ۴. بررسی وجود پرامپت آماده هوش مصنوعی
        has_prompt = "AI System Prompt Instruction" in md_text or "راهنمای پرامپت" in md_text
        checks["ai_prompt_ready"] = has_prompt
        if not has_prompt:
            issues.append("بخش راهنمای پرامپت برای اتصال به مدل زبانی وجود ندارد.")

        # محاسبه نمره کیفی (از 100)
        passed_checks = sum(1 for v in checks.values() if v)
        quality_score = int((passed_checks / len(checks)) * 100)

        return {
            "score": quality_score,
            "status": "PASSED" if quality_score == 100 else "NEEDS_REVIEW",
            "checks": checks,
            "issues": issues,
            "total_lines": len(md_text.splitlines()),
            "total_words": len(md_text.split())
        }


def run_evaluation_test():
    """
    اجرای تست کیفیت روی یک نمونه تولیدشده واقعی
    """
    dummy_input = {
        "title": "سامانه هوشمند عیب‌یابی خط نورد گرم",
        "cost_p10": 3000.0, "cost_p50": 4000.0, "cost_p90": 5500.0,
        "benefit_p10": 6000.0, "benefit_p50": 8000.0, "benefit_p90": 11000.0,
        "p_success": 0.85, "years": 3,
        "inflation_triplet": (0.35, 0.50, 0.70),
        "currency_type": "USD", "annual_fx_savings": 40000.0, "base_fx_rate": 65000.0,
        "enable_energy_risk": True,
        "power_outage_triplet": (10.0, 15.0, 25.0),
        "gas_outage_triplet": (15.0, 20.0, 30.0),
        "enable_reliability_risk": True,
        "mtbf_triplet": (500.0, 900.0, 1200.0),
        "mttr_triplet": (2.0, 4.0, 8.0),
        "enable_iran_tax": True,
        "corporate_tax_rate": 0.20
    }

    dummy_eval = {
        "title": "سامانه هوشمند عیب‌یابی خط نورد گرم",
        "adjusted_net_benefit": 6460.57,
        "total_downtime_hours": 917.1,
        "energy_loss_toman": 1466.94,
        "downtime_loss_toman": 725.01,
        "lead_time_delay_days": 22.7,
        "iran_tax_credit_toman": 652.52,
        "carbon_savings_toman": 0.0,
        "risk_status": "MODERATE_RISK",
        "quantitative_metrics": {"mean_roi": 457.46, "var_95": 15.2, "probability_of_loss": 8.0},
        "qualitative_assessment": {"risk_summary": "پروژه دارای توجیه اقتصادی قوی و ریسک مدیریت‌شده است."}
    }

    # تولید متن گزارش
    md_content = MarkdownReportGenerator.generate_project_md(dummy_input, dummy_eval)

    # ارزیابی کیفی
    eval_result = MarkdownQualityEvaluator.evaluate_content(md_content)

    print("\n--- 📋 کارنامه ارزیابی کیفی فایل گزارش Markdown ---")
    print(f"📊 نمره کیفی: {eval_result['score']} / 100")
    print(f"📌 وضعیت: {eval_result['status']}")
    print(f"📏 تعداد خطوط: {eval_result['total_lines']} خط | تعداد کلمات: {eval_result['total_words']} کلمه")
    
    if eval_result["issues"]:
        print("\n⚠️ موارد نیازمند بازبینی:")
        for issue in eval_result["issues"]:
            print(f"  * {issue}")
    else:
        print("\n✅ فایل گزارش از تمام آزمون‌های ساختاری، سلامت داده و سازگاری با LLM با موفقیت عبور کرد.")


if __name__ == "__main__":
    run_evaluation_test()