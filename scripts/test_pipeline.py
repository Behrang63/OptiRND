import sys
from pathlib import Path

# افزودن مسیر ریشه پروژه به پایتون
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.database import init_db, Proposal
from core.llm_provider import LLMFactory
from agents.main_agents.ex_ante_agent import ExAnteAgent

def run_pipeline_test():
    """
    اجرای یکپارچه جریان کاری: دیتابیس -> عامل هوشمند -> مونت کارلو -> خروجی
    """
    print("--- 🚀 در حال اجرای تست یکپارچه خط لوله پژوهشیار ---")
    
    # ۱. راه‌اندازی دیتابیس در حافظه
    Session = init_db("sqlite:///:memory:")
    session = Session()
    
    # ۲. ساخت داده نمونه پروپوزال
    sample_proposal = {
        "title": "سامانه تشخیص عیوب ورق فولاد مبارکه",
        "cost_p10": 8000.0, "cost_p50": 10000.0, "cost_p90": 14000.0,
        "benefit_p10": 3000.0, "benefit_p50": 5000.0, "benefit_p90": 8000.0,
        "p_success": 0.85
    }

    # ۳. راه‌اندازی عامل پیشینی با هوش مصنوعی آفلاین
    llm = LLMFactory.get_provider("mock")
    agent = ExAnteAgent(llm_provider=llm)

    # ۴. ارزیابی پروپوزال
    result = agent.evaluate_proposal(sample_proposal)
    
    print(f"📌 عنوان پروپوزال: {result['proposal_title']}")
    print(f"📊 میانگین ROI: {result['quantitative_metrics']['mean_roi']}%")
    print(f"🛡️ شاخص VaR 95%: {result['quantitative_metrics']['var_95']}%")
    print(f"📉 احتمال زیان: {result['quantitative_metrics']['probability_of_loss']}%")
    print(f"🏷️ وضعیت سطح اطمینان: {result['confidence_level_status']}")
    print("--- ✅ تست خط لوله با موفقیت کامل انجام شد ---")

if __name__ == "__main__":
    run_pipeline_test()