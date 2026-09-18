import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class MarkdownReportGenerator:
    """
    ماژول تولید شناسنامه جامع و پویا (Dynamic) پروژه با فرمت Markdown (.md)
    جهت ارزیابی دستی و ارسال به عنوان Context به مدل‌های زبانی (LLM).
    """

    @staticmethod
    def generate_project_md(
        proposal_input: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        portfolio_decision: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        تولید متن کامل گزارش با فرمت Markdown به صورت پویا بر اساس ورودی‌های فعال.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = proposal_input.get("title", "پروژه بدون عنوان")
        metrics = evaluation_result.get("quantitative_metrics", {})
        llm_eval = evaluation_result.get("qualitative_assessment", {})

        # ۱. اطلاعات عمومی
        md_lines = [
            f"# 📄 شناسنامه تحلیلی و گزارش ارزیابی جامع پروژه R&D",
            "",
            "---",
            "## ۱. اطلاعات عمومی و شناسه پروژه",
            f"* **عنوان پروژه:** {title}",
            f"* **تاریخ استخراج گزارش:** `{now_str}`",
            f"* **افق زمانی ارزیابی:** {proposal_input.get('years', 3)} سال",
            f"* **احتمال موفقیت فنی پایه:** {int(proposal_input.get('p_success', 0.85) * 100)}%",
            f"* **وضعیت ریسک پروژه:** `{evaluation_result.get('risk_status', 'MODERATE_RISK')}`",
            "",
            "---",
            "## ۲. ورودی‌های مالی و شرایط اقتصاد کلان",
            "| پارامتر ارزیابی | سناریوی خوش‌بینانه (P10) | سناریوی محتمل (P50) | سناریوی بدبینانه (P90) |",
            "| :--- | :---: | :---: | :---: |",
            f"| **هزینه اولیه R&D (م.ت)** | {proposal_input.get('cost_p10', 0):,.0f} | {proposal_input.get('cost_p50', 0):,.0f} | {proposal_input.get('cost_p90', 0):,.0f} |",
            f"| **منافع ریالی سالانه (م.ت)** | {proposal_input.get('benefit_p10', 0):,.0f} | {proposal_input.get('benefit_p50', 0):,.0f} | {proposal_input.get('benefit_p90', 0):,.0f} |"
        ]

        # افزودن پویا ردیف نرخ تورم در صورت وجود
        inflation = proposal_input.get("inflation_triplet")
        if inflation:
            md_lines.append(
                f"| **نرخ تورم سالانه** | {inflation[0]*100:.0f}% | {inflation[1]*100:.0f}% | {inflation[2]*100:.0f}% |"
            )

        md_lines.extend([
            "",
            f"* **ارز مبنا و صرفه‌جویی:** {proposal_input.get('currency_type', 'USD')} با صرفه‌جویی سالانه {proposal_input.get('annual_fx_savings', 0):,.0f} {proposal_input.get('currency_type', 'USD')}",
            f"* **نرخ پایه تبدیل ارز:** {proposal_input.get('base_fx_rate', 65000):,.0f} تومان",
            "",
            "---",
            "## ۳. ارزیابی پویا و تعدیل‌شده ریسک‌های عملیاتی"
        ])

        # ساخت پویای جدول ریسک‌ها
        risk_rows = []
        
        if proposal_input.get("enable_energy_risk", True):
            pwr = proposal_input.get("power_outage_triplet", (0, 0, 0))
            gas = proposal_input.get("gas_outage_triplet", (0, 0, 0))
            e_loss = evaluation_result.get("energy_loss_toman", 0)
            risk_rows.append(
                f"| **ناترازی انرژی (برق و گاز)** | برق: {pwr[1]} روز \\| گاز: {gas[1]} روز | خسارت سالانه: **{e_loss:,.1f} م.ت** |"
            )

        if proposal_input.get("enable_reliability_risk", True):
            mtbf = proposal_input.get("mtbf_triplet", (0, 0, 0))
            mttr = proposal_input.get("mttr_triplet", (0, 0, 0))
            dt_loss = evaluation_result.get("downtime_loss_toman", 0)
            risk_rows.append(
                f"| **قابلیت اطمینان (MTBF/MTTR)** | MTBF: {mtbf[1]} ساعت \\| MTTR: {mttr[1]} ساعت | خسارت توقف: **{dt_loss:,.1f} م.ت** |"
            )

        if proposal_input.get("enable_supply_chain_risk", True):
            p_lead = proposal_input.get("planned_lead_time", 90)
            a_lead = proposal_input.get("actual_lead_time_triplet", (0, 0, 0))
            delay = evaluation_result.get("lead_time_delay_days", 0)
            risk_rows.append(
                f"| **تأخیر زنجیره تأمین** | تعهد: {p_lead} روز \\| واقعی: {a_lead[1]} روز | میانگین تأخیر: **{delay} روز** |"
            )

        # ردیف تجمیعی توقفات
        total_dt = evaluation_result.get("total_downtime_hours", 0)
        risk_rows.append(
            f"| **مجموع توقفات خط** | تجمیع توقف ناشی از خرابی و قطعی انرژی | کل زمان توقف: **{total_dt} ساعت/سال** |"
        )

        if risk_rows:
            md_lines.append("| نوع ریسک صنعتی | مقدار / سناریو | اثر مالی / زمانی تعدیل‌شده |")
            md_lines.append("| :--- | :--- | :--- |")
            md_lines.extend(risk_rows)
        else:
            md_lines.append("*هیچ ریسک عملیاتی متفاوتی برای این پروژه فعال نشده است.*")

        md_lines.extend([
            "",
            "---",
            "## ۴. مشوق‌ها و معافیت‌های قانونی و مالیاتی"
        ])

        # ساخت پویای لیست مشوق‌ها
        incentive_lines = []
        if proposal_input.get("enable_iran_tax", True):
            tax_cr = evaluation_result.get("iran_tax_credit_toman", 0)
            incentive_lines.append(f"* **اعتبار مالیاتی R&D (قانون جهش تولید ایران):** `{tax_cr:,.1f} میلیون تومان`")

        if proposal_input.get("enable_cbam_tax", False):
            cbam_cr = evaluation_result.get("carbon_savings_toman", 0)
            incentive_lines.append(f"* **معافیت عوارض کربن صادراتی اروپا (CBAM):** `{cbam_cr:,.1f} میلیون تومان`")

        if incentive_lines:
            md_lines.extend(incentive_lines)
        else:
            md_lines.append("*هیچ مشوق مالیاتی برای این پروژه فعال نگردیده است.*")

        md_lines.extend([
            "",
            "---",
            "## ۵. خروجی شبیه‌سازی مونت‌کارلو و شاخص‌های تجمیعی",
            f"* **منفعت خالص تعدیل‌شده سالانه (پس از اثر ریسک‌ها و مشوق‌ها):** **`{evaluation_result.get('adjusted_net_benefit', 0):,.1f} میلیون تومان`**",
            f"* **میانگین نرخ بازگشت سرمایه (Mean ROI):** `{metrics.get('mean_roi', 0)}%`",
            f"* **شاخص ارزش در معرض ریسک (VaR 95%):** `{metrics.get('var_95', 0)}%`",
            f"* **احتمال وقوع زیان کل:** `{metrics.get('probability_of_loss', 0)}%`",
            "",
            "---",
            "## ۶. تحلیل اولیه سیستم و شواهد کیفی",
            f"> 💡 **خلاصه ارزیابی اولیه:** {llm_eval.get('risk_summary', 'پروژه با موفقیت ارزیابی شد.')}"
        ])

        # بخش پویای مایلستون‌ها (در صورت وجود)
        milestones = proposal_input.get("milestones", [])
        if milestones:
            md_lines.extend([
                "",
                "---",
                "## ۷. جدول پویای مایلستون‌های اجرایی",
                "| عنوان فاز / مایلستون | بودجه مصوب (م.ت) | زمان‌بندی (روز) | درصد پیشرفت |",
                "| :--- | :---: | :---: | :---: |"
            ])
            for m in milestones:
                md_lines.append(
                    f"| {m.get('title', '-')} | {m.get('planned_budget', 0):,.0f} | {m.get('planned_duration_days', 0)} | {m.get('completion_percentage', 0)}% |"
                )

        # بخش پویای سبد سرمایه‌گذاری
        if portfolio_decision:
            md_lines.extend([
                "",
                "---",
                "## ۸. جایگاه در سبد بهینه‌سازی سرمایه‌گذاری",
                f"* **وضعیت انتخاب در سبد مصوب:** `{'تصویب‌شده' if portfolio_decision.get('is_selected') else 'رد شده / مازاد بر سقف منابع'}`",
                f"* **الگوریتم بهینه‌سازی استفاده‌شده:** `{portfolio_decision.get('method_label', 'نامشخص')}`",
                f"* **سهم از کل بودجه R&D:** `{portfolio_decision.get('budget_share_pct', 0)}%`"
            ])

        # راهنمای سیستم پرامپت
        md_lines.extend([
            "",
            "---",
            "## 🤖 راهنمای پرامپت هوش مصنوعی (AI System Prompt Instruction)",
            "کاربر گرامی! شما می‌توانید این متن را مستقیماً به عنوان زمینه (Context) به همراه دستور زیر به مدل زبانی ارسال کنید:",
            "```text",
            "شما یک مشاور ارشد ارزیابی ریسک و سرمایه‌گذاری پروژه‌های R&D در صنایع سنگین هستید. بر اساس داده‌های ساختاریافته بالا، یک گزارش مدیریتی تحلیلی شامل:",
            "۱. تحلیل حساسیت سود نسبت به ناترازی انرژی و توقفات خط",
            "۲. ارزیابی ریسک دم (Tail Risk) بر اساس VaR",
            "۳. توصیه نهایی برای تصویب یا رد پروژه با ذکر شروط کنترلی",
            "تدوین نمایید.",
            "```",
            ""
        ])

        return "\n".join(md_lines)

    @staticmethod
    def save_md_file(content: str, filename: str, directory: str = "data/reports") -> str:
        """ذخیره فایل روی دیسک محلی"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(file_path)