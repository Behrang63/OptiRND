import sys
import tempfile
import shutil
from pathlib import Path

# افزودن مسیر ریشه پروژه به پایتون
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.llm_provider import LLMFactory
from core.database import (
    init_db,
    save_comprehensive_proposal,
    get_all_proposals,
    get_proposals_for_portfolio,
    delete_proposal_by_id
)
from core.reporter import EvaluationReporter
from agents.main_agents.ex_ante_agent import ExAnteAgent
from core.rag_engine import create_rag_engine


def test_full_comprehensive_workflow():
    """
    تست جامع و سرتاسری جریان کاری سامانه:
    ورودی داده -> ارزیابی زنجیره‌ای عامل -> ذخیره در دیتابیس -> استخراج آماده سبد -> گزارش
    """
    print("\n--- 🚀 آغاز تست جامع یکپارچه‌سازی جریان داده پلتفرم پژوهشیار ---")

    # Create temp directory for RAG vector store to avoid dimension mismatch
    temp_dir = tempfile.mkdtemp()
    
    try:
        # ۱. راه‌اندازی پایگاه داده موقت در حافظه (in-memory) جهت تست ایمن
        session = init_db("sqlite:///:memory:")  # now a session_factory

        # ۲. راه‌اندازی ارائه‌دهنده هوش مصنوعی آفلاین و عامل پیشینی
        llm = LLMFactory.get_provider("mock")
        
        # Inject RAG engine with temp directory to avoid dimension mismatch with existing data
        rag_engine = create_rag_engine(provider_type="mock", store_dir=temp_dir)
        agent = ExAnteAgent(llm_provider=llm, rag_engine=rag_engine)

        # ۳. تعریف ورودی جامع یک پروژه صنعتی با تمام ابعاد ریسک
        sample_proposal_input = {
            "title": "سامانه هوشمند عیب‌یابی خط نورد گرم",
            "cost_p10": 3000.0, "cost_p50": 4000.0, "cost_p90": 5500.0,
            "benefit_p10": 6000.0, "benefit_p50": 8000.0, "benefit_p90": 11000.0,
            "p_success": 0.85,
            "years": 3,
            
            # ریسک انرژی (ناترازی برق و گاز)
            "enable_energy_risk": True,
            "power_outage_triplet": (10.0, 15.0, 25.0),
            "gas_outage_triplet": (15.0, 20.0, 30.0),
            "energy_daily_loss": 40.0,
            
            # ریسک قابلیت اطمینان و توقف خط (MTBF)
            "enable_reliability_risk": True,
            "mtbf_triplet": (500.0, 900.0, 1200.0),
            "mttr_triplet": (2.0, 4.0, 8.0),
            "hourly_downtime_loss": 20.0,
            "annual_operating_hours": 7200.0,
            
            # ریسک لجستیک و زنجیره تأمین
            "enable_supply_chain_risk": True,
            "planned_lead_time": 90.0,
            "actual_lead_time_triplet": (75.0, 110.0, 160.0),
            "daily_delay_cost": 10.0,
            
            # مشوق‌های مالیاتی ایران و کربن اروپا
            "enable_iran_tax": True,
            "corporate_tax_rate": 0.20,
            "enable_cbam_tax": True,
            "export_tons_triplet": (50000.0, 100000.0, 150000.0),
            "co2_reduction_kg": 120.0,
            "carbon_tax_usd_triplet": (50.0, 80.0, 100.0),
            "base_fx_rate": 65000.0
        }

        # درج کلیدواژه‌های ریسک و توقف در متن سند نمونه
        sample_doc_text = "پروژه بومی‌سازی شامل تحلیل ریسک توقف فنی خط نورد و کاهش مصرف گاز کوره پیش‌گرم است."

        # ۴. اجرای ارزیابی جامع با عامل هوشمند
        evaluation_result = agent.evaluate_comprehensive_proposal(
            proposal_data=sample_proposal_input,
            raw_text=sample_doc_text
        )

        # اعتبارسنجی خروجی‌های محاسباتی عامل
        assert evaluation_result["adjusted_net_benefit"] > 0
        assert evaluation_result["total_downtime_hours"] > 0
        assert "mean_roi" in evaluation_result
        assert len(evaluation_result["rag_evidence"]) > 0

        # ۵. ذخیره‌سازی داده‌های تجمیعی در پایگاه داده
        saved_record = save_comprehensive_proposal(session, evaluation_result)

        assert saved_record.id is not None
        assert saved_record.title == "سامانه هوشمند عیب‌یابی خط نورد گرم"
        assert saved_record.adjusted_net_benefit == evaluation_result["adjusted_net_benefit"]

        # ۶. استعلام و تبدیل خودکار داده‌ها برای جدول چیدمان سبد پروژه‌ها
        portfolio_rows = get_proposals_for_portfolio(session)
        assert len(portfolio_rows) == 1
        assert portfolio_rows[0]["عنوان پروژه"] == "سامانه هوشمند عیب‌یابی خط نورد گرم"
        assert portfolio_rows[0]["هزینه P50"] == 4000.0
        assert portfolio_rows[0]["منافع P50"] == evaluation_result["adjusted_net_benefit"]

        # ۷. تولید گزارش متنی ارزیابی
        report_text = EvaluationReporter.generate_ex_ante_report(sample_proposal_input, evaluation_result)
        assert "سامانه هوشمند عیب‌یابی خط نورد گرم" in report_text

        # ۸. تست حذف رکورد از دیتابیس
        delete_status = delete_proposal_by_id(session, saved_record.id)
        assert delete_status is True
        assert len(get_all_proposals(session)) == 0

        print("\n✅ [گام ۳: تست جامع جریان کاری و یکپارچگی خط لوله با موفقیت کامل پاس شد]")
        print(f"📦 منفعت خالص تعدیل‌شده در سبد: {portfolio_rows[0]['منافع P50']:,.1f} م.ت")
        print(f"⏱️ مجموع توقف سالانه در سبد: {portfolio_rows[0]['توقف P50']} ساعت")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_full_comprehensive_workflow()