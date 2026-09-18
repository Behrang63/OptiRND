import numpy as np
from typing import Dict, Any, Tuple

from core.distributions import generate_pert_samples, make_rng


class ReliabilityEngine:
    """
    موتور شبیه‌سازی مونت‌کارلو برای ارزیابی قابلیت اطمینان تجهیزات،
    محاسبه نرخ خرابی (MTBF)، زمان تعمیر (MTTR) و زیان توقف خط تولید.

    RNG policy: uses an isolated numpy Generator (dependency-injected).
    The global numpy random state is never mutated.
    """

    def __init__(self, num_simulations: int = 10000, seed: int = None,
                 rng: np.random.Generator = None):
        self.num_simulations = int(num_simulations)
        self.rng = rng if rng is not None else make_rng(seed)

    def simulate_downtime_risk(
        self,
        base_annual_benefit: float,
        mtbf_hours_triplet: Tuple[float, float, float],    # ساعت بین خرابی (P10, P50, P90)
        mttr_hours_triplet: Tuple[float, float, float],    # ساعت زمان تعمیر (P10, P50, P90)
        hourly_downtime_loss: float,                       # هزینه خسارت هر ساعت توقف خط (میلیون تومان)
        annual_operating_hours: float = 7200.0,            # کل ساعت کاری خط در سال
        years: int = 3
    ) -> Dict[str, Any]:
        """
        اجرای شبیه‌سازی مونت‌کارلو برای محاسبه توقفات خط و زیان مالی آن.
        """
        # ۱. نمونه‌گیری مقادیر MTBF و MTTR با توزیع PERT (shared utility, isolated rng)
        mtbf_samples = generate_pert_samples(*mtbf_hours_triplet,
                                             size=self.num_simulations, rng=self.rng)
        mttr_samples = generate_pert_samples(*mttr_hours_triplet,
                                             size=self.num_simulations, rng=self.rng)

        # ۲. محاسبه تعداد خرابی‌های سالانه (جلوگیری از تقسیم بر صفر)
        safe_mtbf = np.maximum(mtbf_samples, 10.0)
        expected_failures_per_year = annual_operating_hours / safe_mtbf

        # نمونه‌گیری تعداد وقوع رخداد خرابی با توزیع پواسون
        simulated_failures = self.rng.poisson(expected_failures_per_year)

        # ۳. محاسبه کل ساعات توقف خط در سال
        annual_downtime_hours = simulated_failures * mttr_samples

        # ۴. محاسبه قابلیت دسترسی خط (Availability Percentage)
        availability_pct = (
            (annual_operating_hours - np.minimum(annual_downtime_hours, annual_operating_hours))
            / annual_operating_hours
        ) * 100.0

        # ۵. محاسبه زیان سالانه توقفات (میلیون تومان)
        annual_downtime_loss = annual_downtime_hours * hourly_downtime_loss

        # منفعت خالص تعدیل‌شده سالانه
        adjusted_benefit = np.maximum(base_annual_benefit - annual_downtime_loss, 0.0)

        return {
            "mean_annual_failures": round(float(np.mean(simulated_failures)), 1),
            "mean_downtime_hours": round(float(np.mean(annual_downtime_hours)), 1),
            "mean_availability_pct": round(float(np.mean(availability_pct)), 2),
            "mean_annual_downtime_loss": round(float(np.mean(annual_downtime_loss)), 2),
            "total_lifetime_loss": round(float(np.mean(annual_downtime_loss * years)), 2),
            "adjusted_annual_benefit": round(float(np.mean(adjusted_benefit)), 2),
            "max_downtime_hours_p95": round(float(np.percentile(annual_downtime_hours, 95)), 1),
            "raw_downtime_hours": annual_downtime_hours
        }
