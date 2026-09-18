from core.monte_carlo import MonteCarloEngine

def test_monte_carlo_with_fx_scenarios():
    engine = MonteCarloEngine(num_simulations=5000)
    
    # تست سناریوی بومی‌سازی با صرفه‌جویی سالانه ۵۰ هزار دلار
    result = engine.run_roi_simulation(
        cost_p10_p50_p90=(8000.0, 10000.0, 14000.0),       # هزینه اولیه (میلیون تومان)
        benefit_p10_p50_p90=(2000.0, 3000.0, 5000.0),      # منفعت ریالی سالانه (میلیون تومان)
        inflation_p10_p50_p90=(0.35, 0.50, 0.65),          # سه‌گانه تورم ریالی
        fx_growth_p10_p50_p90=(0.30, 0.40, 0.60),          # سه‌گانه رشد نرخ دلار
        currency_type="USD",
        base_fx_rate=65000.0,                              # نرخ پایه دلار به تومان
        annual_fx_savings=50000.0,                         # صرفه‌جویی سالانه ۵۰ هزار دلار
        p_success=0.90,
        years=3
    )
    
    print("\n--- نتایج شبیه‌سازی سناریوی ارزی و بومی‌سازی ---")
    print(f"نوع ارز انتخابی: {result['currency_type']}")
    print(f"میانگین بازگشت سرمایه (ROI): {result['mean_roi']}%")
    print(f"شاخص ریسک VaR 95%: {result['var_95']}%")
    print(f"ارزش کل صرفه‌جویی ارزی (معادل میلیون تومان): {result['expected_total_fx_savings_toman']:,.0f}")
    
    assert "mean_roi" in result
    assert result["expected_total_fx_savings_toman"] > 0

if __name__ == "__main__":
    test_monte_carlo_with_fx_scenarios()