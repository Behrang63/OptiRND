import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.llm_provider import LLMFactory
from agents.main_agents.ex_ante_agent import ExAnteAgent


def test_comprehensive_ex_ante_agent():
    llm = LLMFactory.get_provider("mock")
    agent = ExAnteAgent(llm_provider=llm)

    sample_proposal = {
        "title": "سامانه هوشمند کنترل خط نورد",
        "cost_p10": 2000.0, "cost_p50": 2500.0, "cost_p90": 3500.0,
        "benefit_p10": 4000.0, "benefit_p50": 5500.0, "benefit_p90": 7000.0,
        "p_success": 0.85,
        "years": 3,
        "enable_energy_risk": True,
        "power_outage_triplet": (5.0, 10.0, 20.0),
        "gas_outage_triplet": (10.0, 15.0, 25.0),
        "energy_daily_loss": 30.0,
        "enable_reliability_risk": True,
        "mtbf_triplet": (600.0, 1000.0, 1400.0),
        "mttr_triplet": (2.0, 3.0, 6.0),
        "hourly_downtime_loss": 15.0,
        "enable_iran_tax": True,
        "corporate_tax_rate": 0.20
    }

    result = agent.evaluate_comprehensive_proposal(sample_proposal)

    assert "adjusted_net_benefit" in result
    assert "total_downtime_hours" in result
    assert result["total_downtime_hours"] > 0
    assert result["adjusted_net_benefit"] > 0

    print("\n✅ [گام دوم: تست جامع عامل ارزیابی پیشینی با موفقیت کامل انجام شد]")
    print(f"منفعت اولیه پایه: {sample_proposal['benefit_p50']} م.ت")
    print(f"منفعت خالص تعدیل‌شده: {result['adjusted_net_benefit']:,.1f} م.ت")
    print(f"مجموع ساعات توقف خط: {result['total_downtime_hours']} ساعت")
    print(f"اعتبار مالیاتی ذخیره‌شده: {result['iran_tax_credit_toman']:,.1f} م.ت")


if __name__ == "__main__":
    test_comprehensive_ex_ante_agent()