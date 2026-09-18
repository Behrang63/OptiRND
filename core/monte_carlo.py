import numpy as np
from typing import Dict, Any, Tuple

from core.distributions import generate_pert_samples, make_rng

class MonteCarloEngine:
    """
    موتور شبیه‌سازی مونت‌کارلو با قابلیت محاسبه اثرات تورم ریالی،
    نوسانات ارزی (دلار، یورو، یوان) و صرفه‌جویی ارزی بومی‌سازی.

    RNG policy: uses an isolated numpy Generator (dependency-injected).
    The global numpy random state is never mutated.
    """
    def __init__(self, num_simulations: int = 10000, seed: int = None,
                 rng: np.random.Generator = None):
        self.num_simulations = int(num_simulations)
        # Deterministic stream ONLY when an explicit seed is passed;
        # otherwise a fresh isolated Generator (no global state touched).
        self.rng = rng if rng is not None else make_rng(seed)

    def run_roi_simulation(
        self,
        cost_p10_p50_p90: Tuple[float, float, float],
        benefit_p10_p50_p90: Tuple[float, float, float],
        inflation_p10_p50_p90: Tuple[float, float, float] = (0.35, 0.50, 0.70),
        fx_growth_p10_p50_p90: Tuple[float, float, float] = (0.30, 0.45, 0.65),
        currency_type: str = "USD",
        base_fx_rate: float = 65000.0,       # قیمت اولیه ارز به تومان
        annual_fx_savings: float = 0.0,      # میزان صرفه‌جویی ارزی سالانه
        p_success: float = 0.85,
        real_discount_rate: float = 0.10,
        years: int = 3
    ) -> Dict[str, Any]:
        """
        اجرای شبیه‌سازی جامع با اعمال مدل تورم فیشر و رشد نرخ ارز.
        """
        # ۱. نمونه‌گیری توزیع PERT (shared utility, isolated rng)
        cost_samples = generate_pert_samples(*cost_p10_p50_p90, size=self.num_simulations, rng=self.rng)
        annual_domestic_benefit = generate_pert_samples(*benefit_p10_p50_p90, size=self.num_simulations, rng=self.rng)
        inflation_samples = generate_pert_samples(*inflation_p10_p50_p90, size=self.num_simulations, rng=self.rng)
        fx_growth_samples = generate_pert_samples(*fx_growth_p10_p50_p90, size=self.num_simulations, rng=self.rng)
        
        # ۲. متغیر تصادفی موفقیت فنی پروژه
        p_success_bounded = min(max(float(p_success), 0.0), 1.0)
        success_mask = self.rng.binomial(1, p_success_bounded, self.num_simulations)
        
        # ۳. تنزیل جریان‌های نقدی اسمی بر اساس رابطه فیشر: (1 + i) = (1 + r)(1 + pi)
        total_discounted_benefits = np.zeros(self.num_simulations)
        total_realized_fx_savings_toman = np.zeros(self.num_simulations)
        
        nominal_discount_samples = (1.0 + real_discount_rate) * (1.0 + inflation_samples) - 1.0
        # جلوگیری از مقادیر نامعتبر تنزیل
        nominal_discount_samples = np.maximum(nominal_discount_samples, -0.99)

        for t in range(1, int(years) + 1):
            # الف) رشد منفعت ریالی متناسب با تورم
            escalated_domestic_benefit = annual_domestic_benefit * ((1.0 + inflation_samples) ** t)
            
            # ب) ارزش ریالی صرفه‌جویی ارزی در سال t (میلیون تومان)
            current_fx_rate = base_fx_rate * ((1.0 + fx_growth_samples) ** t)
            fx_savings_in_toman_millions = (annual_fx_savings * current_fx_rate) / 1_000_000.0
            
            # تجمیع منافع سال t
            total_annual_benefit = escalated_domestic_benefit + fx_savings_in_toman_millions
            
            # تنزیل جریان نقدی به زمان مبدأ
            discount_factor = 1.0 / ((1.0 + nominal_discount_samples) ** t)
            total_discounted_benefits += (total_annual_benefit * discount_factor)
            total_realized_fx_savings_toman += fx_savings_in_toman_millions

        # ۴. محاسبه ارزش خالص فعلی (NPV) و نرخ بازگشت سرمایه (ROI)
        npv_samples = (total_discounted_benefits * success_mask) - cost_samples
        roi_samples = (npv_samples / np.maximum(cost_samples, 1.0)) * 100.0

        # ۵. محاسبه شاخص‌های آماری خروجی
        mean_roi = float(np.mean(roi_samples))
        std_roi = float(np.std(roi_samples))
        var_95 = float(np.percentile(roi_samples, 5))
        var_99 = float(np.percentile(roi_samples, 1))

        return {
            "mean_roi": round(mean_roi, 2),
            "std_dev": round(std_roi, 2),
            "var_95": round(var_95, 2),
            "var_99": round(var_99, 2),
            "expected_npv": round(float(np.mean(npv_samples)), 2),
            "probability_of_loss": round(float(np.mean(roi_samples < 0) * 100), 2),
            "expected_total_fx_savings_toman": round(float(np.mean(total_realized_fx_savings_toman)), 2),
            "currency_type": currency_type,
            "raw_roi_samples": roi_samples
        }