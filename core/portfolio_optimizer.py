import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import List, Dict, Any

from core.distributions import generate_pert_samples, make_rng


class PortfolioOptimizationEngine:
    """
    موتور بهینه‌سازی پیشرفته سبد پروژه‌های R&D با پشتیبانی از ۳ مدل عدم‌قطعیت:
    ۱. برنامه‌ریزی استوار (Robust Optimization)
    ۲. برنامه‌ریزی تصادفی سناریومحور (Stochastic Programming)
    ۳. بهینه‌سازی ارزش در معرض ریسک شرطی (Mean-CVaR)

    RNG policy: uses an isolated numpy Generator (dependency-injected).
    The global numpy random state is never mutated. Deterministic streams
    are produced ONLY when an explicit seed is passed.
    """

    def __init__(self, num_scenarios: int = 1000, seed: int = None,
                 rng: np.random.Generator = None):
        self.num_scenarios = int(num_scenarios)
        self.rng = rng if rng is not None else make_rng(seed)

    def _generate_pert_matrix(self, triplets: np.ndarray) -> np.ndarray:
        """
        تولید ماتریس سناریوهای تصادفی بر اساس سه‌گانه‌های P10, P50, P90
        triplets shape: (num_projects, 3)
        خروجی: (num_projects, num_scenarios)

        Implemented on top of the shared core.distributions PERT utility;
        each project row is sampled from the injected isolated Generator.
        """
        n_proj = triplets.shape[0]
        scenarios = np.zeros((n_proj, self.num_scenarios))

        for i in range(n_proj):
            scenarios[i, :] = generate_pert_samples(
                triplets[i][0], triplets[i][1], triplets[i][2],
                size=self.num_scenarios, rng=self.rng,
            )

        return scenarios

    def optimize_portfolio(
        self,
        projects: List[Dict[str, Any]],
        max_budget: float,
        max_downtime_hours: float,
        method: str = "stochastic",  # 'robust', 'stochastic', 'cvar'
        cvar_alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        اجرای بهینه‌سازی چیدمان سبد بر اساس الگوریتم انتخابی.
        """
        n = len(projects)
        if n == 0:
            return {"status": "EMPTY", "selected_projects": []}

        # استخراج سه‌گانه‌های هزینه، منفعت و توقفات
        cost_triplets = np.array([[p.get("cost_p10", 0), p.get("cost_p50", 0), p.get("cost_p90", 0)] for p in projects])
        benefit_triplets = np.array([[p.get("benefit_p10", 0), p.get("benefit_p50", 0), p.get("benefit_p90", 0)] for p in projects])
        downtime_triplets = np.array([[p.get("dt_p10", 0), p.get("dt_p50", 0), p.get("dt_p90", 0)] for p in projects])

        # تولید سناریوهای تصادفی (shared utility, isolated rng)
        cost_scenarios = self._generate_pert_matrix(cost_triplets)
        benefit_scenarios = self._generate_pert_matrix(benefit_triplets)
        downtime_scenarios = self._generate_pert_matrix(downtime_triplets)

        # محاسبه سناریوهای ارزش فعلی خالص (NPV)
        npv_scenarios = benefit_scenarios - cost_scenarios

        # ۱. تنظیم بردار هدف (c) و قیود بر اساس متد انتخابی
        if method == "robust":
            # بدترین سناریو: کمترین سود، بیشترین هزینه و بیشترین توقف
            expected_npv = np.min(npv_scenarios, axis=1)
            effective_costs = np.max(cost_scenarios, axis=1)
            effective_downtimes = np.max(downtime_scenarios, axis=1)

        elif method == "cvar":
            # بهینه‌سازی میانگین ۵٪ بدترین سناریوها (CVaR)
            expected_npv = np.zeros(n)
            cutoff_index = int(self.num_scenarios * cvar_alpha)
            for i in range(n):
                sorted_npv = np.sort(npv_scenarios[i, :])
                expected_npv[i] = np.mean(sorted_npv[:max(cutoff_index, 1)])
            effective_costs = np.mean(cost_scenarios, axis=1)
            effective_downtimes = np.mean(downtime_scenarios, axis=1)

        else:  # 'stochastic'
            # بهینه‌سازی امید ریاضی (میانگین تمام سناریوها)
            expected_npv = np.mean(npv_scenarios, axis=1)
            effective_costs = np.mean(cost_scenarios, axis=1)
            effective_downtimes = np.mean(downtime_scenarios, axis=1)

        # ۲. فرموله‌سازی MILP برای بیشینه‌سازی سود (کمینه‌سازی c = -expected_npv)
        c = -expected_npv
        A = np.vstack([effective_costs, effective_downtimes])
        b_u = np.array([max_budget, max_downtime_hours])
        b_l = np.array([0.0, 0.0])

        constraints = LinearConstraint(A, b_l, b_u)
        integrality = np.ones(n)  # متغیرهای باینری (۰ یا ۱)
        bounds = Bounds(0, 1)

        # ۳. حل مسئله با موتور MILP
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

        if not res.success:
            return {"status": "FAILED", "message": "ترکیب بهینه‌ای با این قیود یافت نشد."}

        # ۴. تجمیع نتایج پروژه‌های انتخاب‌شده
        selection_mask = np.round(res.x).astype(bool)
        selected_projects = []
        for i, is_selected in enumerate(selection_mask):
            if is_selected:
                proj_info = dict(projects[i])
                proj_info["calculated_score_npv"] = round(float(expected_npv[i]), 2)
                proj_info["effective_cost"] = round(float(effective_costs[i]), 2)
                proj_info["effective_downtime"] = round(float(effective_downtimes[i]), 2)
                selected_projects.append(proj_info)

        total_cost = sum(p["effective_cost"] for p in selected_projects)
        total_npv = sum(p["calculated_score_npv"] for p in selected_projects)
        total_downtime = sum(p["effective_downtime"] for p in selected_projects)

        return {
            "status": "OPTIMAL",
            "method_used": method,
            "selected_projects": selected_projects,
            "total_selected_npv": round(total_npv, 2),
            "total_selected_cost": round(total_cost, 2),
            "total_selected_downtime": round(total_downtime, 2),
            "budget_utilization_pct": round((total_cost / max(max_budget, 1.0)) * 100.0, 1),
            "downtime_utilization_pct": round((total_downtime / max(max_downtime_hours, 1.0)) * 100.0, 1)
        }
