import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.portfolio_optimizer import PortfolioOptimizationEngine

def test_all_portfolio_methods():
    engine = PortfolioOptimizationEngine(num_scenarios=500)

    sample_projects = [
        {"id": 1, "title": "پروژه A (بومی‌سازی نسوز)", "cost_p10": 2000, "cost_p50": 2500, "cost_p90": 3500, "benefit_p10": 4000, "benefit_p50": 5500, "benefit_p90": 7000, "dt_p10": 10, "dt_p50": 15, "dt_p90": 25},
        {"id": 2, "title": "پروژه B (کاهش مصرف برق کوره)", "cost_p10": 3000, "cost_p50": 4000, "cost_p90": 5500, "benefit_p10": 6000, "benefit_p50": 8500, "benefit_p90": 11000, "dt_p10": 15, "dt_p50": 20, "dt_p90": 30},
        {"id": 3, "title": "پروژه C (تشخیص آنلاین عیوب)", "cost_p10": 1000, "cost_p50": 1500, "cost_p90": 2000, "benefit_p10": 2000, "benefit_p50": 3000, "benefit_p90": 4000, "dt_p10": 5, "dt_p50": 8, "dt_p90": 12},
    ]

    for method in ["robust", "stochastic", "cvar"]:
        res = engine.optimize_portfolio(
            projects=sample_projects,
            max_budget=6000.0,
            max_downtime_hours=35.0,
            method=method
        )
        assert res["status"] == "OPTIMAL"
        assert res["total_selected_cost"] <= 6000.0
        assert res["total_selected_downtime"] <= 35.0
        print(f"✅ تست الگوریتم {method} موفقیت‌آمیز بود (تعداد پروژه‌ها: {len(res['selected_projects'])})")

if __name__ == "__main__":
    test_all_portfolio_methods()