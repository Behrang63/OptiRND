import numpy as np
from typing import Dict, Any, Tuple

from core.distributions import generate_pert_samples, make_rng


class CarbonTaxEngine:
    """
    موتور شبیه‌سازی مالیاتی:
    ۱. عوارض و معافیت کربن صادراتی اروپا (CBAM)
    ۲. اعتبار و معافیت مالیاتی پروژه‌های تحقیق و توسعه طبق قوانین ایران (قانون جهش تولید)

    RNG policy: uses an isolated numpy Generator (dependency-injected).
    The global numpy random state is never mutated.
    """

    def __init__(self, num_simulations: int = 10000, seed: int = None,
                 rng: np.random.Generator = None):
        self.num_simulations = int(num_simulations)
        self.rng = rng if rng is not None else make_rng(seed)

    def simulate_eu_cbam_benefit(
        self,
        annual_export_tons_triplet: Tuple[float, float, float],     # تناژ صادرات فولاد به اروپا (تن)
        co2_reduction_per_ton_kg: float,                            # کاهش CO2 به ازای هر تن (کیلوگرم)
        carbon_tax_per_ton_usd_triplet: Tuple[float, float, float], # مالیات کربن اروپا (دلار / تن CO2)
        usd_to_toman_rate: float = 65000.0,
        years: int = 3
    ) -> Dict[str, Any]:
        """
        محاسبه صرفه‌جویی حاصل از معافیت جریمه‌های صادراتی بر مبنای قوانین اروپا (CBAM).
        """
        export_samples = generate_pert_samples(*annual_export_tons_triplet,
                                               size=self.num_simulations, rng=self.rng)
        tax_samples = generate_pert_samples(*carbon_tax_per_ton_usd_triplet,
                                            size=self.num_simulations, rng=self.rng)

        co2_reduced_tons = export_samples * (co2_reduction_per_ton_kg / 1000.0)
        annual_savings_usd = co2_reduced_tons * tax_samples
        annual_savings_toman = (annual_savings_usd * usd_to_toman_rate) / 1_000_000.0

        return {
            "mean_annual_co2_reduced_tons": round(float(np.mean(co2_reduced_tons)), 1),
            "mean_annual_carbon_savings_usd": round(float(np.mean(annual_savings_usd)), 2),
            "mean_annual_carbon_savings_toman": round(float(np.mean(annual_savings_toman)), 2),
            "total_lifetime_savings_toman": round(float(np.mean(annual_savings_toman * years)), 2),
            "raw_savings_toman": annual_savings_toman
        }

    def simulate_iran_tax_credit(
        self,
        rd_cost_triplet: Tuple[float, float, float],          # هزینه R&D پروژه (میلیون تومان)
        approved_cost_pct_triplet: Tuple[float, float, float],# درصد تایید هزینه توسط کارگروه (P10, P50, P90)
        corporate_tax_rate: float = 0.20                      # نرخ مالیات عملکرد صنعتی در ایران (۲۰ یا ۲۵ درصد)
    ) -> Dict[str, Any]:
        """
        محاسبه اعتبار مالیاتی و کاهش مالیات قطعی بر اساس ماده ۱۱ قانون جهش تولید دانش‌بنیان در ایران.
        """
        rd_costs = generate_pert_samples(*rd_cost_triplet,
                                         size=self.num_simulations, rng=self.rng)
        approved_pcts = generate_pert_samples(*approved_cost_pct_triplet,
                                              size=self.num_simulations, rng=self.rng)

        # هزینه‌های مورد تایید اداره مالیات
        approved_rd_expenditures = rd_costs * approved_pcts

        # میزان اعتبار مالیاتی قابل کسر از مالیات عملکرد سالانه
        tax_credit_savings_toman = approved_rd_expenditures * corporate_tax_rate

        return {
            "mean_rd_cost_toman": round(float(np.mean(rd_costs)), 2),
            "mean_approved_rd_cost": round(float(np.mean(approved_rd_expenditures)), 2),
            "mean_iran_tax_credit_savings": round(float(np.mean(tax_credit_savings_toman)), 2),
            "savings_p90_toman": round(float(np.percentile(tax_credit_savings_toman, 90)), 2),
            "raw_tax_savings": tax_credit_savings_toman
        }
