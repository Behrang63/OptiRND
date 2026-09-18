import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.carbon_tax_engine import CarbonTaxEngine

def test_tax_simulations():
    engine = CarbonTaxEngine(num_simulations=5000)
    
    # ۱. تست محاسبات بر مبنای قوانین اروپا
    eu_res = engine.simulate_eu_cbam_benefit(
        annual_export_tons_triplet=(100000.0, 200000.0, 300000.0),
        co2_reduction_per_ton_kg=150.0,
        carbon_tax_per_ton_usd_triplet=(60.0, 85.0, 120.0),
        usd_to_toman_rate=65000.0,
        years=3
    )
    assert eu_res["mean_annual_carbon_savings_usd"] > 0
    print("\n✅ [تست معافیت کربن اروپا (CBAM) با موفقیت انجام شد]")
    print(f"صرفه‌جویی سالانه اروپا: {eu_res['mean_annual_carbon_savings_usd']:,.0f} دلار")

    # ۲. تست محاسبات بر مبنای قوانین ایران
    iran_res = engine.simulate_iran_tax_credit(
        rd_cost_triplet=(8000.0, 10000.0, 14000.0),
        approved_cost_pct_triplet=(0.60, 0.80, 0.95),
        corporate_tax_rate=0.20
    )
    assert iran_res["mean_iran_tax_credit_savings"] > 0
    print("✅ [تست اعتبار مالیاتی قانون جهش تولید ایران با موفقیت انجام شد]")
    print(f"میانگین تخفیف مالیاتی در ایران: {iran_res['mean_iran_tax_credit_savings']:,.0f} میلیون تومان")

if __name__ == "__main__":
    test_tax_simulations()