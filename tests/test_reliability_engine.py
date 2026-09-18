import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.reliability_engine import ReliabilityEngine

def test_reliability_simulation():
    engine = ReliabilityEngine(num_simulations=5000)
    
    # تست با فرض MTBF بین ۵۰۰ تا ۱۵۰۰ ساعت و MTTR بین ۲ تا ۱۰ ساعت
    result = engine.simulate_downtime_risk(
        base_annual_benefit=6000.0,
        mtbf_hours_triplet=(500.0, 1000.0, 1500.0),
        mttr_hours_triplet=(2.0, 4.0, 8.0),
        hourly_downtime_loss=15.0,  # ساعتی ۱۵ میلیون تومان خسارت توقف
        annual_operating_hours=7200.0,
        years=3
    )
    
    assert "mean_annual_failures" in result
    assert "mean_availability_pct" in result
    assert result["mean_availability_pct"] > 0
    assert result["mean_annual_downtime_loss"] >= 0
    
    print("\n✅ [تست موتور شبیه‌سازی قابلیت اطمینان و توقفات با موفقیت انجام شد]")
    print(f"میانگین تعداد دفعات خرابی سالانه: {result['mean_annual_failures']} بار")
    print(f"میانگین ساعت توقف سالانه: {result['mean_downtime_hours']} ساعت")
    print(f"درصد دسترسی‌پذیری تجهیز: {result['mean_availability_pct']}%")
    print(f"میانگین زیان مالی توقفات: {result['mean_annual_downtime_loss']:,.0f} میلیون تومان")

if __name__ == "__main__":
    test_reliability_simulation()