import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.portfolio_reporter import PortfolioReporter


def test_portfolio_comparison():
    reporter = PortfolioReporter()

    sample_projects = [
        {"id": 1, "title": "پروژه عیب‌یابی ورق", "cost_p10": 2000, "cost_p50": 2500, "cost_p90": 3500, "benefit_p10": 4000, "benefit_p50": 5500, "benefit_p90": 7000, "dt_p10": 10, "dt_p50": 15, "dt_p90": 25},
        {"id": 2, "title": "بومی‌سازی کاتالیست", "cost_p10": 3000, "cost_p50": 4000, "cost_p90": 5500, "benefit_p10": 6000, "benefit_p50": 8500, "benefit_p90": 11000, "dt_p10": 15, "dt_p50": 20, "dt_p90": 30},
        {"id": 3, "title": "کاهش مصرف نسوز", "cost_p10": 1000, "cost_p50": 1500, "cost_p90": 2000, "benefit_p10": 2000, "benefit_p50": 3000, "benefit_p90": 4000, "dt_p10": 5, "dt_p50": 8, "dt_p90": 12}
    ]

    df_comparison = reporter.compare_all_methods(
        projects=sample_projects,
        max_budget=7000.0,
        max_downtime_hours=35.0
    )

    assert len(df_comparison) == 3
    assert "رویکرد بهینه‌سازی" in df_comparison.columns
    print("\n✅ [تست ماژول مقایسه هم‌زمان سناریوهای سبد با موفقیت کامل پاس شد]")
    print(df_comparison[["رویکرد بهینه‌سازی", "مجموع سود خالص (NPV - م.ت)", "زمان توقف خط (ساعت)"]])


if __name__ == "__main__":
    test_portfolio_comparison()