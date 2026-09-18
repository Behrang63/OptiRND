import streamlit as st
import pandas as pd
from core.database import (
    get_all_proposals,
    delete_proposal_by_id,
    DatabaseTransactionError,
    RecordNotFoundError,
)
from core.reporter import EvaluationReporter


def render_history_view(db_session_factory):
    """
    نمایش تاریخچه جامع پروپوزال‌های ثبت‌شده و مدیریت سوابق در دیتابیس.
    """
    st.header("📜 ۵. بانک اطلاعاتی و سوابق ارزیابی پروژه‌ها")
    
    with st.expander("📖 **راهنمای مدیریت سوابق (کلیک کنید)**", expanded=False):
        st.markdown("""
        در این بخش تمامی پروپوزال‌های ارزیابی‌شده همراه با شاخص‌های تعدیل‌شده ذخیره شده‌اند:
        * می‌توانید وضعیت ریسک، سود خالص تعدیل‌شده و ساعات توقف پیش‌بینی‌شده هر پروژه را مشاهده کنید.
        * امکان حذف رکوردهای قدیمی یا دریافت مجدد گزارش شناسنامه پروژه وجود دارد.
        """)

    try:
        proposals = get_all_proposals(db_session_factory)
        if not proposals:
            st.info("هنوز هیچ پروپوزالی در بانک اطلاعاتی ثبت نشده است. ابتدا از بخش «۱. ثبت و ارزیابی جامع پروژه» یک پروژه ثبت کنید.")
            return
        else:
            table_data = []
            for prop in proposals:
                table_data.append({
                    "شناسه": prop.id,
                    "عنوان پروژه": prop.title,
                    "هزینه P50": f"{getattr(prop, 'cost_p50', 0):,.0f} م.ت",
                    "منافع پایه P50": f"{getattr(prop, 'benefit_p50', 0):,.0f} م.ت",
                    "سود تعدیل‌شده": f"{getattr(prop, 'adjusted_net_benefit', 0):,.0f} م.ت",
                    "توقف سالانه": f"{getattr(prop, 'total_downtime_hours', 0)} ساعت",
                    "احتمال موفقیت": f"{int(getattr(prop, 'p_success', 0) * 100)}%",
                    "میانگین ROI": f"{prop.mean_roi}%" if prop.mean_roi is not None else "-",
                    "شاخص VaR 95%": f"{prop.var_95}%" if prop.var_95 is not None else "-",
                    "سطح ریسک": getattr(prop, 'risk_status', "-") or "-",
                    "تاریخ ثبت": prop.created_at.strftime("%Y-%m-%d %H:%M") if getattr(prop, 'created_at', None) else "-"
                })
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            st.markdown("---")
            col_act1, col_act2 = st.columns(2)
            
            # بخش حذف رکورد
            with col_act1:
                st.subheader("🗑️ حذف پروژه از بانک اطلاعاتی")
                delete_id = st.number_input("شناسه پروژه جهت حذف:", min_value=1, step=1, key="del_prop_id")
                if st.button("حذف قطعی رکورد"):
                    try:
                        delete_proposal_by_id(db_session_factory, int(delete_id))
                        st.success(f"پروژه شماره {delete_id} با موفقیت از دیتابیس حذف شد.")
                        st.rerun()
                    except RecordNotFoundError:
                        st.error(f"پروژه‌ای با شناسه {delete_id} یافت نشد.")
                    except DatabaseTransactionError as e:
                        st.error(f"⛔ خطای تراکنش در حذف رکورد: {e}")

            # بخش دانلود شناسنامه متنی
            with col_act2:
                st.subheader("📥 دریافت شناسنامه پروژه")
                selected_id = st.number_input("شناسه پروژه جهت دریافت گزارش:", min_value=1, step=1, key="rep_prop_id")
                target_prop = next((p for p in proposals if p.id == selected_id), None)
                if target_prop:
                    dummy_input = {
                        "title": target_prop.title,
                        "cost_p10": target_prop.cost_p10, "cost_p50": target_prop.cost_p50, "cost_p90": target_prop.cost_p90,
                        "benefit_p10": target_prop.benefit_p10, "benefit_p50": target_prop.benefit_p50, "benefit_p90": target_prop.benefit_p90,
                        "p_success": target_prop.p_success
                    }
                    dummy_result = {
                        "quantitative_metrics": {"mean_roi": target_prop.mean_roi, "var_95": target_prop.var_95, "probability_of_loss": target_prop.probability_of_loss},
                        "confidence_level_status": target_prop.risk_status
                    }
                    report_txt = EvaluationReporter.generate_ex_ante_report(dummy_input, dummy_result)
                    st.download_button(
                        label="📄 دانلود فایل گزارش متنی",
                        data=report_txt,
                        file_name=f"Report_Project_{target_prop.id}.txt",
                        mime="text/plain"
                    )
    except DatabaseTransactionError as e:
        st.error(f"⛔ خطای تراکنش پایگاه داده: {e}")