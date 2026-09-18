import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from agents.sub_agents.test_stress_agent import StressTestSubAgent


def test_stress_subagent_execution():
    """تست اجرای سناریوهای تستی خودکار و ثبت لاگ خطاها"""
    tester = StressTestSubAgent()
    
    success_list, errors = tester.execute_and_evaluate_all()
    
    assert len(success_list) + len(errors) == 4
    assert len(success_list) >= 3
    
    print("\n--- 📊 کارنامه نتایج اجرای سناریوهای خودکار Sub-Agent ---")
    for item in success_list:
        print(f"✅ سناریو: {item['scenario_name']} | سود تعدیل‌شده: {item['adjusted_benefit']:,.1f} م.ت | توقف کل: {item['total_downtime_hours']} ساعت | وضعیت: {item['risk_status']}")
        
    if errors:
        print(f"\n⚠️ تعداد خطاهای رخ داده: {len(errors)} (جزئیات در logs/test_agent_errors.log ثبت شد)")
    else:
        print("\n🎯 تمامی سناریوها بدون کرش و با پایداری کامل توسط ماشین ارزیابی شدند.")


if __name__ == "__main__":
    test_stress_subagent_execution()