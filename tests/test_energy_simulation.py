import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.energy_simulation import EnergyRiskEngine

def test_energy_simulation():
    engine = EnergyRiskEngine(num_simulations=5000)
    
    # تست با فرض ۱۰ الی ۳۰ روز قطعی برق و ۱۵ الی ۴۵ روز قطعی گاز
    results = engine.simulate_energy_impact(
        base_annual_benefit=5000.0,
        power_outage_days_triplet=(10.0, 20.0, 35.0),
        gas_outage_days_triplet=(15.0, 30.0, 45.0),
        daily_downtime_loss=50.0,  # روزی ۵۰ میلیون تومان
        years=3
    )
    
    assert "mean_outage_days_yearly" in results
    assert results["mean_outage_days_yearly"] > 0
    assert results["mean_annual_loss_toman"] > 0
    print("\n✅ [تست موتور شبیه‌سازی انرژی با موفقیت انجام شد]")
    print(f"میانگین روزهای قطعی سالانه: {results['mean_outage_days_yearly']} روز")
    print(f"میانگین خسارت سالانه انرژی: {results['mean_annual_loss_toman']:,.0f} میلیون تومان")

if __name__ == "__main__":
    test_energy_simulation()