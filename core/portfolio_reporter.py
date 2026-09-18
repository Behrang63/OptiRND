import pandas as pd
from typing import List, Dict, Any
from core.portfolio_optimizer import PortfolioOptimizationEngine


class PortfolioReporter:
    """
    ماژول تولید گزارش‌های تحلیلی و مقایسه هم‌زمان مدل‌های بهینه‌سازی سبد پروژه‌ها.
    """
    def __init__(self):
        self.optimizer = PortfolioOptimizationEngine(num_scenarios=1000)

    def compare_all_methods(
        self,
        projects: List[Dict[str, Any]],
        max_budget: float,
        max_downtime_hours: float
    ) -> pd.DataFrame:
        """
        اجرای هم‌زمان هر ۳ مدل بهینه‌سازی و ساخت جدول مقایسه‌ای جهت تصمیم‌گیری مدیران.
        """
        methods = {
            "برنامه‌ریزی تصادفی (Stochastic)": "stochastic",
            "برنامه‌ریزی استوار (Robust)": "robust",
            "بهینه‌سازی ریسک دم (Mean-CVaR)": "cvar"
        }

        comparison_data = []

        for label, method_key in methods.items():
            res = self.optimizer.optimize_portfolio(
                projects=projects,
                max_budget=max_budget,
                max_downtime_hours=max_downtime_hours,
                method=method_key
            )

            if res.get("status") == "OPTIMAL":
                selected_titles = ", ".join([p["title"] for p in res["selected_projects"]])
                comparison_data.append({
                    "رویکرد بهینه‌سازی": label,
                    "تعداد پروژه‌های مصوب": len(res["selected_projects"]),
                    "مجموع سود خالص (NPV - م.ت)": f"{res['total_selected_npv']:,.0f}",
                    "بودجه مصرفی (م.ت)": f"{res['total_selected_cost']:,.0f} ({res['budget_utilization_pct']}%)",
                    "زمان توقف خط (ساعت)": f"{res['total_selected_downtime']} ({res['downtime_utilization_pct']}%)",
                    "عناوین پروژه‌های منتخب": selected_titles
                })
            else:
                comparison_data.append({
                    "رویکرد بهینه‌سازی": label,
                    "تعداد پروژه‌های مصوب": 0,
                    "مجموع سود خالص (NPV - م.ت)": "0",
                    "بودجه مصرفی (م.ت)": "0",
                    "زمان توقف خط (ساعت)": "0",
                    "عناوین پروژه‌های منتخب": "بدون پاسخ موجه"
                })

        return pd.DataFrame(comparison_data)

    def generate_excel_report(self, selected_projects: List[Dict[str, Any]], summary_metrics: Dict[str, Any], file_path: str) -> str:
        """
        تولید فایل اکسل ساختاریافته از سبد مصوب پروژه‌ها.
        """
        df_projects = pd.DataFrame(selected_projects)
        df_summary = pd.DataFrame([summary_metrics])

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name="خلاصه شاخص‌های کلان", index=False)
            df_projects.to_excel(writer, sheet_name="فهرست پروژه‌های مصوب", index=False)

        return file_path