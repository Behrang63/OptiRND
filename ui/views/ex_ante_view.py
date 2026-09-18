import streamlit as st
import pandas as pd
from pathlib import Path

from core.parser import (
    DocumentParser,
    DocumentParserError,
    ExtractionIncompleteError,
    FileTooLargeError,
    InvalidDocumentError,
)
from core.repository import ProposalRepository
from core.services import EvaluationService
from core.database import DatabaseTransactionError
from core.md_reporter import MarkdownReportGenerator
from agents.main_agents.ex_ante_agent import ExAnteAgent
from core.task_manager import AsyncTaskManager
from core.contracts import TaskStatus


def reset_proposal_form():
    """
    بازنشاری مقادیر ورودی فرم برای ثبت پروپوزال جدید
    """
    st.session_state["proposal_form_version"] = st.session_state.get("proposal_form_version", 0) + 1
    st.session_state["last_evaluated_result"] = None
    st.session_state["last_proposal_input"] = None


def render_ex_ante_view(llm_provider, db_session_factory):
    """
    نمایش صفحه ثبت و ارزیابی جامع پروپوزال با فرم چندتبی یکپارچه،
    ذخیره‌سازی در پایگاه داده و ذخیره خودکار گزارش Markdown در پوشه data/reports/.
    """
    if "proposal_form_version" not in st.session_state:
        st.session_state["proposal_form_version"] = 0
    if "last_evaluated_result" not in st.session_state:
        st.session_state["last_evaluated_result"] = None
    if "last_proposal_input" not in st.session_state:
        st.session_state["last_proposal_input"] = None
    if "async_task_manager" not in st.session_state:
        st.session_state["async_task_manager"] = AsyncTaskManager(max_workers=2)
    if "current_task_id" not in st.session_state:
        st.session_state["current_task_id"] = None

    form_v = st.session_state["proposal_form_version"]

    st.header("📋 ۱. ثبت و ارزیابی جامع پروژه (Project Intake)")

    with st.expander("📖 **راهنمای جامع ارزیابی یکپارچه (کلیک کنید)**", expanded=False):
        st.markdown("""
        این بخش نقطه شروع_flowchart_flowchart_flowchart
        * تمام ابعاد مالی، ریسک‌های صنعتی خط تولید و مشوق‌های قانونی را در تب‌های زیر وارد کنید.
        * سیستم تمام محاسبات را به‌صورت زنجیره‌ای شبیه‌سازی کرده و سود خالص تعدیل‌شده و کل ساعات توقف خط را محاسبه می‌کند.
        * پس از ثبت، پروژه وارد **بانک اطلاعاتی و جدول چیدمان سبد پروژه‌ها** خواهد شد.
        """)

    # ۱. بارگذاری سند PDF پروپوزال
    st.subheader("📁 بارگذاری سند پروپوزال (اختیاری)")
    uploaded_file = st.file_uploader(
        "فایل PDF پروپوزال را انتخاب کنید", 
        type=["pdf"], 
        key=f"uploader_pdf_{form_v}"
    )
    
    # Neutral UI defaults - the operator ALWAYS confirms/edits these before an
    # evaluation runs. The parser no longer fabricates financial values.
    parsed_defaults = {
        "cost_p10": 0.0, "cost_p50": 0.0, "cost_p90": 0.0,
        "benefit_p10": 0.0, "benefit_p50": 0.0, "benefit_p90": 0.0,
        "p_success": 0.85
    }
    extracted_pdf_text = ""

    if uploaded_file is not None:
        temp_path = Path("data") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            parser = DocumentParser(str(temp_path))
            extraction = parser.parse_financial_triplets()
            extracted_pdf_text = parser.extract_text()

            if extraction.is_complete:
                parsed_defaults.update(extraction.extracted_fields)
                st.success(f"✅ فایل '{uploaded_file.name}' با موفقیت پردازش شد.")
            else:
                st.warning(
                    "⚠️ استخراج سه‌گانه مالی ناقص بود؛ لطفاً مقادیر زیر را دستی وارد کنید. "
                    "هیچ مقدار پیش‌فرض جعلی به مدل ریسک تزریق نمی‌شود. "
                    f"(فیلدهای ناموجود: {', '.join(extraction.missing_fields)})"
                )
        except FileTooLargeError as e:
            st.error(f"⛔ حجم فایل بیش از حد مجاز است: {e}")
        except InvalidDocumentError as e:
            st.error(f"⛔ فایل معتبر PDF نیست یا ساختار آن مخدوش است: {e}")
        except ExtractionIncompleteError as e:
            st.error(f"⚠️ استخراج ناقص: {e}")
        except DocumentParserError as e:
            st.error(f"خطا در پردازش فایل: {e}")

    st.markdown("---")
    title = st.text_input(
        "عنوان پروژه تحقیق و توسعه:", 
        value="سامانه هوشمند عیب‌یابی و کنترل خط نورد", 
        key=f"title_{form_v}"
    )
    
    # ۲. تب‌های ۵ گانه ثبت ورودی‌ها
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 ۱. مالی و تورم", 
        "⚡ ۲. ریسک انرژی", 
        "⚙️ ۳. توقفات و MTBF", 
        "🚚 ۴. لجستیک و ارز", 
        "⚖️ ۵. مشوق‌های مالیاتی"
    ])

    # -------------------------------------------------------------
    # تب ۱: مالی، فنی و تورم
    # -------------------------------------------------------------
    with tab1:
        st.markdown("#### پارامترهای پایه مالی و نرخ تورم ریالی")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("**تخمین costo اولیه پروژه (میلیون تومان):**")
            cost_p10 = st.number_input("هزینه خوش‌بینانه (P10)", value=float(parsed_defaults["cost_p10"]), step=500.0, key=f"c10_{form_v}")
            cost_p50 = st.number_input("هزینه محتمل (P50)", value=float(parsed_defaults["cost_p50"]), step=500.0, key=f"c50_{form_v}")
            cost_p90 = st.number_input("هزینه بدبینانه (P90)", value=float(parsed_defaults["cost_p90"]), step=500.0, key=f"c90_{form_v}")
            
        with c_col2:
            st.markdown("**تخمین منافع ریالی سالانه پایه (میلیون تومان):**")
            benefit_p10 = st.number_input("منافع خوش‌بینانه (P10)", value=float(parsed_defaults["benefit_p10"]), step=500.0, key=f"b10_{form_v}")
            benefit_p50 = st.number_input("منافع محتمل (P50)", value=float(parsed_defaults["benefit_p50"]), step=500.0, key=f"b50_{form_v}")
            benefit_p90 = st.number_input("منافع بدبینانه (P90)", value=float(parsed_defaults["benefit_p90"]), step=500.0, key=f"b90_{form_v}")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            p_success = st.slider("احتمال موفقیت فنی پروژه:", 0.0, 1.0, float(parsed_defaults["p_success"]), step=0.05, key=f"psucc_{form_v}")
        with p_col2:
            project_years = st.slider("افق زمانی بهره‌برداری (سال):", 1, 10, 3, key=f"years_{form_v}")

        st.markdown("**سه‌گانه درصدTorمول سالانه ریالی:**")
        inf_col1, inf_col2, inf_col3 = st.columns(3)
        with inf_col1:
            inf_p10 = st.number_input("حداقلTorمول (P10 - درصد)", value=35.0, step=5.0, key=f"inf10_{form_v}")
        with inf_col2:
            inf_p50 = st.number_input("محتمل‌ترینTorمول (P50 - درصد)", value=50.0, step=5.0, key=f"inf50_{form_v}")
        with inf_col3:
            inf_p90 = st.number_input("حداکثرTorمول (P90 - درصد)", value=70.0, step=5.0, key=f"inf90_{form_v}")

    # -------------------------------------------------------------
    # تب ۲: ریسک ناترازی انرژی
    # -------------------------------------------------------------
    with tab2:
        st.markdown("#### شبیه‌سازی اثرات قطعی برق و گاز بر عملکرد proyecto")
        enable_energy = st.checkbox("فعال‌سازی ارزیابی ریسک انرژی برای این پروژه", value=True, key=f"en_chk_{form_v}")
        en_col1, en_col2 = st.columns(2)
        with en_col1:
            pwr_p10 = st.number_input("خوش‌بینانه برق (P10 - روز)", value=10.0, step=1.0, key=f"pwr10_{form_v}")
            pwr_p50 = st.number_input("محتمل برق (P50 - روز)", value=15.0, step=1.0, key=f"pwr50_{form_v}")
            pwr_p90 = st.number_input("بدبینانه برق (P90 - روز)", value=25.0, step=1.0, key=f"pwr90_{form_v}")
        with en_col2:
            gas_p10 = st.number_input("خوش‌بینانه گاز (P10 - روز)", value=15.0, step=1.0, key=f"gas10_{form_v}")
            gas_p50 = st.number_input("محتمل گاز (P50 - روز)", value=20.0, step=1.0, key=f"gas50_{form_v}")
            gas_p90 = st.number_input("بدبینانه گاز (P90 - روز)", value=30.0, step=1.0, key=f"gas90_{form_v}")
        en_loss_daily = st.number_input("زیان روزانه توقف فرآیند ناشی از قطعی انرژی (میلیون تومان):", value=40.0, step=5.0, key=f"enloss_{form_v}")

    # -------------------------------------------------------------
    # تب ۳: قابلیت اطمینان و توقفات خط (MTBF)
    # -------------------------------------------------------------
    with tab3:
        st.markdown("#### ارزیابی نقص فنی و توقفات خط (MTBF)")
        enable_reliability = st.checkbox("فعال‌سازی ارزیابی MTBF و توقفات خط", value=True, key=f"rel_chk_{form_v}")
        rel_col1, rel_col2 = st.columns(2)
        with rel_col1:
            mtbf_p10 = st.number_input("خوش‌بینانه (P10 - ساعت)", value=1200.0, step=100.0, key=f"mtbf10_{form_v}")
            mtbf_p50 = st.number_input("محتمل (P50 - ساعت)", value=900.0, step=100.0, key=f"mtbf50_{form_v}")
            mtbf_p90 = st.number_input("بدبینانه (P90 - ساعت)", value=500.0, step=100.0, key=f"mtbf90_{form_v}")
        with rel_col2:
            mttr_p10 = st.number_input("خوش‌بینانه (P10 - ساعت)", value=2.0, step=0.5, key=f"mttr10_{form_v}")
            mttr_p50 = st.number_input("محتمل (P50 - ساعت)", value=4.0, step=0.5, key=f"mttr50_{form_v}")
            mttr_p90 = st.number_input("بدبینانه (P90 - ساعت)", value=8.0, step=0.5, key=f"mttr90_{form_v}")
        op_col1, op_col2 = st.columns(2)
        with op_col1:
            annual_hours = st.number_input("کل ساعات کاری سالانه خط (ساعت):", value=7200.0, step=200.0, key=f"ann_hrs_{form_v}")
        with op_col2:
            hourly_downtime_loss = st.number_input("خسارت هر ساعت توقف خط (میلیون تومان):", value=20.0, step=2.0, key=f"hr_loss_{form_v}")

    # -------------------------------------------------------------
    # تب ۴: لجستیک و ارز
    # -------------------------------------------------------------
    with tab4:
        st.markdown("#### تأخیرات زنجیره تأمین و صرفه‌جویی ارزی بومی‌سازی")
        cur_col1, cur_col2 = st.columns(2)
        with cur_col1:
            currency_choice = st.selectbox("ارز مرجع صرفه‌جویی:", ["دلار آمریکا (USD)", "یورو (EUR)", "یوان چین (CNY)"], key=f"cur_choice_{form_v}")
            currency_code = "USD" if "USD" in currency_choice else ("EUR" if "EUR" in currency_choice else "CNY")
            base_fx_rate = st.number_input(f"نرخ فعلی {currency_choice} به تومان:", value=65000.0, step=1000.0, key=f"base_fx_{form_v}")
        with cur_col2:
            annual_fx_savings = st.number_input(f"صرفه‌جویی ارزی سالانه ({currency_code}):", value=40000.0, step=5000.0, key=f"fx_sav_{form_v}")

        enable_supply_chain = st.checkbox("فعال‌سازی ارزیابی تأخیر در ترخیص و تحویل تجهیزات", value=True, key=f"sc_chk_{form_v}")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            planned_lead_time = st.number_input("زمان مصوب تحویل در قرارداد (روز):", value=90.0, step=5.0, key=f"plt_{form_v}")
        with sc_col2:
            daily_delay_cost = st.number_input("خسارت روزانه دیرکرد تحویل (میلیون تومان):", value=10.0, step=1.0, key=f"ddc_{form_v}")

        lt_col1, lt_col2, lt_col3 = st.columns(3)
        with lt_col1:
            lt_p10 = st.number_input("سریع‌ترینTorMoldel (P10)", value=75.0, step=5.0, key=f"lt10_{form_v}")
        with lt_col2:
            lt_p50 = st.number_input("TorMoldel عادی (P50)", value=110.0, step=5.0, key=f"lt50_{form_v}")
        with lt_col3:
            lt_p90 = st.number_input("TorMoldel با تأخیر حاد (P90)", value=160.0, step=5.0, key=f"lt90_{form_v}")

    # -------------------------------------------------------------
    # تب ۵: مشوق‌ها و معافیت‌های مالیاتی
    # -------------------------------------------------------------
    with tab5:
        st.markdown("#### اعتبار مالیاتی در ایران و معافیت کربن اروپا")
        enable_iran_tax = st.checkbox("اعتبار مالیاتی تحقیق و توسعه طبق قانون جهش تولید در ایران", value=True, key=f"irtax_chk_{form_v}")
        tax_col1, tax_col2 = st.columns(2)
        with tax_col1:
            corporate_tax_pct = st.number_input("نرخ مالیات عملکرد شرکت (درصد):", value=20.0, step=1.0, key=f"corptax_{form_v}")
        with tax_col2:
            approved_cost_pct = st.slider("درصد پیش‌بینی تایید هزینه‌های R&D توسط کارگروه:", 0.0, 1.0, 0.80, step=0.05, key=f"apppct_{form_v}")

        enable_cbam = st.checkbox("معافیت از مالیات کربن صادراتی به اروپا (CBAM)", value=False, key=f"cbam_chk_{form_v}")
        if enable_cbam:
            cbam_col1, cbam_col2 = st.columns(2)
            with cbam_col1:
                export_tons = st.number_input("تناژ صادرات سالانه مشمول عوارض (تن):", value=100000.0, step=10000.0, key=f"exptons_{form_v}")
            with cbam_col2:
                co2_reduction_kg = st.number_input("کاهش CO2 به‌ازای هر تن تولید (کیلوگرم):", value=120.0, step=10.0, key=f"co2red_{form_v}")
        else:
            export_tons = 0.0
            co2_reduction_kg = 0.0

    st.markdown("---")
    
    # ۳. دکمه‌های اقدام
    btn_col1, btn_col2 = st.columns([3, 2])
    with btn_col1:
        eval_btn = st.button("🚀 اجرای شبیه‌سازی جامع و ذخیره مستقیم در بانک پروژه‌ها", type="primary")
    with btn_col2:
        st.button("➕ ثبت اطلاعات از پروپوزال جدید", on_click=reset_proposal_form)

    # ۴. اجرای محاسبات و ذخیره‌سازی خودکار در دیتابیس و پوشه data/reports
    if eval_btn:
        # Data-integrity guard: refuse to evaluate an unedited/zeroed form.
        # Every financial input must be a positive, operator-confirmed value.
        if min(float(cost_p10), float(cost_p50), float(cost_p90),
               float(benefit_p10), float(benefit_p50), float(benefit_p90)) <= 0:
            st.error(
                "⛔ مقادیر مالی باید مثبت و تأییدشده باشند. در صورت استخراج ناقص از سند، "
                "لطفاً سه‌گانه‌های هزینه و منافع را دستی وارد کنید."
            )
            st.stop()

        agent = ExAnteAgent(llm_provider=llm_provider)

        costs_sorted = sorted([float(cost_p10), float(cost_p50), float(cost_p90)])
        benefits_sorted = sorted([float(benefit_p10), float(benefit_p50), float(benefit_p90)])
        inf_sorted = sorted([float(inf_p10)/100.0, float(inf_p50)/100.0, float(inf_p90)/100.0])
        pwr_sorted = sorted([float(pwr_p10), float(pwr_p50), float(pwr_p90)])
        gas_sorted = sorted([float(gas_p10), float(gas_p50), float(gas_p90)])
        mtbf_sorted = sorted([float(mtbf_p10), float(mtbf_p50), float(mtbf_p90)])
        mttr_sorted = sorted([float(mttr_p10), float(mttr_p50), float(mttr_p90)])
        lt_sorted = sorted([float(lt_p10), float(lt_p50), float(lt_p90)])

        proposal_input = {
            "title": title,
            "cost_p10": costs_sorted[0], "cost_p50": costs_sorted[1], "cost_p90": costs_sorted[2],
            "benefit_p10": benefits_sorted[0], "benefit_p50": benefits_sorted[1], "benefit_p90": benefits_sorted[2],
            "p_success": float(p_success),
            "years": int(project_years),
            "inflation_triplet": tuple(inf_sorted),
            "currency_type": currency_code,
            "base_fx_rate": float(base_fx_rate),
            "annual_fx_savings": float(annual_fx_savings),
            
            "enable_energy_risk": enable_energy,
            "power_outage_triplet": tuple(pwr_sorted),
            "gas_outage_triplet": tuple(gas_sorted),
            "energy_daily_loss": float(en_loss_daily),
            
            "enable_reliability_risk": enable_reliability,
            "mtbf_triplet": tuple(mtbf_sorted),
            "mttr_triplet": tuple(mttr_sorted),
            "hourly_downtime_loss": float(hourly_downtime_loss),
            "annual_operating_hours": float(annual_hours),
            
            "enable_supply_chain_risk": enable_supply_chain,
            "planned_lead_time": float(planned_lead_time),
            "actual_lead_time_triplet": tuple(lt_sorted),
            "daily_delay_cost": float(daily_delay_cost),
            
            "enable_iran_tax": enable_iran_tax,
            "corporate_tax_rate": float(corporate_tax_pct) / 100.0,
            "approved_pct_triplet": (approved_cost_pct * 0.8, approved_cost_pct, min(approved_cost_pct * 1.2, 1.0)),
            "enable_cbam_tax": enable_cbam,
            "export_tons_triplet": (export_tons * 0.7, export_tons, export_tons * 1.3) if enable_cbam else (0.0, 0.0, 0.0),
            "co2_reduction_kg": float(co2_reduction_kg) if enable_cbam else 0.0,
            "carbon_tax_usd_triplet": (60.0, 85.0, 120.0)
        }

        tm: AsyncTaskManager = st.session_state.get("async_task_manager")
        
        with st.spinner("در حال ارسال tarea به	ThreadPool..."):
            task_id = tm.submit_evaluation(proposal_input, EvaluationService(
                repository=ProposalRepository(db_session_factory),
                agent=agent,
            ))
        
        st.session_state["current_task_id"] = task_id
        st.rerun()

    # ── Non-blocking async polling section ────────────────────────
    # If a task is in progress, show progress and poll without blocking
    current_task_id = st.session_state.get("current_task_id")
    if current_task_id:
        tm: AsyncTaskManager = st.session_state.get("async_task_manager")
        task_result = tm.get_task_status(current_task_id)
        
        # Display progress while task is active
        if task_result.status in (TaskStatus.RUNNING, TaskStatus.PENDING):
            st.info(f"🔄 وضعیت任务: {task_result.stage}")
            st.progress(task_result.progress)
            # Non-blocking poll: rerun to check status again
            st.rerun()
        
        # Task completed - show results and clean up
        if task_result.status == TaskStatus.COMPLETED:
            # Find the EvaluationResult from the task result
            result = task_result.result
            if result:
                st.session_state["last_evaluated_result"] = result
            # Preserve proposal input if available
            if "last_proposal_input" not in st.session_state:
                st.session_state["last_proposal_input"] = st.session_state.get("last_proposal_input")
            st.success(f"✅ tarea tamam: {result.get('title', 'Project') if result else 'Project'}")
            # Clean up the task ID from session
            del st.session_state["current_task_id"]
            st.rerun()
        
        # Task failed - show error without UI collapse
        if task_result.status == TaskStatus.FAILED:
            st.error(f"❌ tarea zadeh: {task_result.error_message or 'Unknown error'}")
            # Allow subsequent tasks to run normally
            del st.session_state["current_task_id"]
            st.rerun()

    # ۵. نمایش کارنامه نتایج
    last_res = st.session_state.get("last_evaluated_result")
    last_inp = st.session_state.get("last_proposal_input")
    last_path = st.session_state.get("last_saved_md_path")
    
    if last_res and last_inp:
        st.subheader("🎯 کارنامه نتایج تجمیعی و تعدیل‌شده پروژه ثبت‌شده")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("منفعت خالص تعدیل‌شده سالانه", f"{last_res['adjusted_net_benefit']:,.0f} م.ت")
        m2.metric("مجموع ساعات توقف خط در سال", f"{last_res['total_downtime_hours']} ساعت")
        m3.metric("میانگین بازگشت سرمایه (ROI)", f"{last_res['mean_roi']}%")
        m4.metric("شاخص ریسک VaR 95%", f"{last_res['var_95']}%")

        llm_info = last_res.get("qualitative_assessment", {})
        if llm_info:
            st.info(f"🤖 **توصیه استراتژیک سیستم:** {llm_info.get('risk_summary', 'ارزیابی ریسک با موفقیت انجام شد.')}")

        md_text = MarkdownReportGenerator.generate_project_md(last_inp, last_res)

        st.markdown("---")
        st.subheader("📄 دانلود خروجی گزارش ساختاریافته Markdown")
        if last_path:
            st.caption(f"📁 نسخه محلی ذخیره‌شده: `{last_path}`")

        st.download_button(
            label="📥 دانلود مستقیم فایل (.md)",
            data=md_text,
            file_name=f"Report_{last_res.get('title', 'Project')}.md",
            mime="text/markdown"
        )