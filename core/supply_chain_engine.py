import numpy as np
from typing import Dict, Any, Tuple

from core.distributions import generate_pert_samples, make_rng


class SupplyChainEngine:
    """
    موتور شبیه‌سازی مونت‌کارلو برای ارزیابی ریسک‌های زنجیره تأمین،
    مدت‌زمان تحویل تجهیزات (Lead-Time) و خسارت‌های ناشی از تأخیر لجستیکی.

    RNG policy: uses an isolated numpy Generator (dependency-injected).
    The global numpy random state is never mutated.
    """

    def __init__(self, num_simulations: int = 10000, seed: int = None,
                 rng: np.random.Generator = None):
        self.num_simulations = int(num_simulations)
        self.rng = rng if rng is not None else make_rng(seed)

    def simulate_lead_time_risk(
        self,
        planned_lead_time_days: float,                       # زمان مصوب تحویل تجهیز (روز)
        actual_lead_time_triplet: Tuple[float, float, float], # سه‌گانه روزهای واقعی تحویل (P10, P50, P90)
        daily_delay_cost: float,                             # هزینه خسارت روزانه تأخیر (میلیون تومان)
        critical_delay_threshold: float = 60.0               # آستانه تأخیر بحرانی (روز)
    ) -> Dict[str, Any]:
        """
        محاسبه توزیع روزهای تأخیر، خسارت مالی و احتمال عبور از آستانه بحرانی.
        """
        # ۱. نمونه‌گیری روزهای واقعی تحویل تجهیز (shared PERT utility, isolated rng)
        simulated_lead_times = generate_pert_samples(*actual_lead_time_triplet,
                                                     size=self.num_simulations, rng=self.rng)

        # ۲. محاسبه روزهای تأخیر مازاد بر برنامه مصوب
        delay_days = np.maximum(simulated_lead_times - planned_lead_time_days, 0.0)

        # ۳. محاسبه خسارت مالی ناشی از تأخیر پروژه (میلیون تومان)
        financial_loss = delay_days * daily_delay_cost

        # ۴. محاسبه احتمال تأخیر حاد و بحرانی
        prob_critical_delay = float(np.mean(delay_days >= critical_delay_threshold) * 100.0)

        # ۵. محاسبه شاخص‌های آماری
        mean_lead_time = float(np.mean(simulated_lead_times))
        mean_delay = float(np.mean(delay_days))
        mean_loss = float(np.mean(financial_loss))
        delay_p95 = float(np.percentile(delay_days, 95))
        loss_p95 = float(np.percentile(financial_loss, 95))

        return {
            "mean_lead_time_days": round(mean_lead_time, 1),
            "mean_delay_days": round(mean_delay, 1),
            "mean_delay_cost": round(mean_loss, 2),
            "max_delay_days_p95": round(delay_p95, 1),
            "max_delay_cost_p95": round(loss_p95, 2),
            "prob_critical_delay_pct": round(prob_critical_delay, 1),
            "raw_delay_days": delay_days
        }
