import numpy as np
from typing import Dict, Any, Tuple

from core.distributions import generate_pert_samples, make_rng


class EnergyRiskEngine:
    """
    موتور شبیه‌سازی مونت‌کارلو برای سنجش ریسک‌های ناشی از ناترازی گاز و برق
    و محاسبه هزینه فرصت ناشی از توقف تولید در پروژه‌های صنعتی.

    RNG policy: uses an isolated numpy Generator (dependency-injected).
    The global numpy random state is never mutated.
    """

    def __init__(self, num_simulations: int = 10000, seed: int = None,
                 rng: np.random.Generator = None):
        self.num_simulations = int(num_simulations)
        # Deterministic stream ONLY when an explicit seed is passed;
        # otherwise a fresh isolated Generator (no global state touched).
        self.rng = rng if rng is not None else make_rng(seed)

    def simulate_energy_impact(
        self,
        base_annual_benefit: float,
        power_outage_days_triplet: Tuple[float, float, float],   # روزهای قطعی برق در تابستان (P10, P50, P90)
        gas_outage_days_triplet: Tuple[float, float, float],     # روزهای قطعی گاز در زمستان (P10, P50, P90)
        daily_downtime_loss: float,                              # خسارت روزانه توقف به میلیون تومان
        years: int = 3
    ) -> Dict[str, Any]:
        """
        محاسبه کاهش منافع پروژه ناشی از قطعی‌های پیش‌بینی‌نشده انرژی.
        """
        # ۱. نمونه‌گیری روزهای قطعی سالانه گاز و برق (shared PERT utility, isolated rng)
        power_days = generate_pert_samples(*power_outage_days_triplet,
                                           size=self.num_simulations, rng=self.rng)
        gas_days = generate_pert_samples(*gas_outage_days_triplet,
                                         size=self.num_simulations, rng=self.rng)

        # مجموع روزهای توقف ناشی از انرژی در هر سال
        total_curtailment_days = power_days + gas_days

        # ۲. محاسبه زیان سالانه عدم تولید (میلیون تومان)
        annual_energy_losses = total_curtailment_days * daily_downtime_loss

        # ۳. محاسبه منفعت تعدیل‌شده پس از کسر خسارت انرژی
        adjusted_annual_benefit = np.maximum(base_annual_benefit - annual_energy_losses, 0.0)

        # تجمیع خسارت کل در طول دوره چندساله
        total_lifetime_losses = annual_energy_losses * years

        return {
            "mean_outage_days_yearly": round(float(np.mean(total_curtailment_days)), 1),
            "mean_annual_loss_toman": round(float(np.mean(annual_energy_losses)), 2),
            "total_expected_energy_loss": round(float(np.mean(total_lifetime_losses)), 2),
            "max_annual_loss_scenario": round(float(np.percentile(annual_energy_losses, 95)), 2),
            "adjusted_annual_benefit": round(float(np.mean(adjusted_annual_benefit)), 2),
            "raw_loss_samples": annual_energy_losses
        }
