import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.supply_chain_engine import SupplyChainEngine

def test_supply_chain_simulation():
    engine = SupplyChainEngine(num_simulations=5000)
    
    # تست با فرض زمان مصوب ۹۰ روز و زمان واقعی بین ۶۰ تا ۱۸۰ روز
    results = engine.simulate_lead_time_risk(
        planned_lead_time_days=90.0,
        actual_lead_time_triplet=(60.0, 110.0, 180.0),
        daily_delay_cost=10.0,  # روزی ۱۰ میلیون تومان خسارت تأخیر
        critical_delay_threshold=45.0
    )
    
    assert "mean_delay_days" in results
    assert "mean_delay_cost" in results
    assert results["mean_lead_time_days"] > 0
    
    print("\n✅ [تست موتور شبیه‌سازی زنجیره تأمین با موفقیت انجام شد]")
    print(f"میانگین کل زمان تحویل: {results['mean_lead_time_days']} روز")
    print(f"میانگین روزهای تأخیر: {results['mean_delay_days']} روز")
    print(f"میانگین خسارت مالی تأخیر: {results['mean_delay_cost']:,.0f} میلیون تومان")
    print(f"احتمال وقوع تأخیر بحرانی: {results['prob_critical_delay_pct']}%")

if __name__ == "__main__":
    test_supply_chain_simulation()