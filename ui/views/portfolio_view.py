import streamlit as st
import pandas as pd
from core.database import (
    get_proposals_for_portfolio,
    delete_proposal_by_id,
    init_db,
    DatabaseTransactionError,
    RecordNotFoundError,
)
from core.portfolio_optimizer import PortfolioOptimizationEngine
from core.portfolio_reporter import PortfolioReporter


def render_portfolio_view(db_session_factory=None):
    """
    نمایش صفحه بهینه‌سازی، مقایسه هم‌زمان سناریوها و مدیریت و حذف پروژه‌های سبد R&D.
    """
    st.header("📊 ۲. بهینه‌سازی و چیدمان سبد پروژه‌های R&D")
    
    with st.expander("📖 **راهنمای ویرایش، حذف و انتخاب الگوریتم (کلیک کنید)**", expanded=False):
        st.markdown("""
        در این بخش می‌توانید پروژه‌ها را مدیریت کرده و سبد مصوب را استخراج نمایید:
        * **حذف موقت از تحلیل جاری:** روی شماره سمت راست ردیف در جدول کلیک کرده و دکمه **Delete** کیبورد را بزنید.
        * **حذف دائمی از بانک اطلاعاتی:** از بخش «حذف قطعی پروژه» در پایین جدول استفاده کنید.
        * **رویکردهای بهینه‌سازی:**
          1. **تصادفی (Stochastic):** حداکثرسازی سود مورد انتظار در شرایط نوسانات ملایم.
          2. **استوار (Robust):** مصون‌سازی سبد در برابر بدترین سناریوهای احتمالی هزینه و توقف.
          3. **ارزش در معرض ریسک شرطی (Mean-CVaR):** کنترل زیان‌های حاد و سناریوهای بحرانی دم توزیع.
        """)

    # ۱. بارگذاری پروژه‌ها از دیتابیس
    if db_session_factory is None:
        db_session_factory = init_db()

    try:
        db_projects = get_proposals_for_portfolio(db_session_factory)
    except DatabaseTransactionError as e:
        st.error(f"⛔ خطای تراکنش پایگاه داده: {e}")
        db_projects = []

    if not db_projects:
        default_data = [
            {"شناسه": 1, "عنوان پروژه": "سامانه هوشمند عیب‌یابی خط نورد گرم", "هزینه P10": 3000.0, "هزینه P50": 4000.0, "هزینه P90": 5500.0, "منافع P10": 5000.0, "منافع P50": 7200.0, "منافع P90": 9800.0, "توقف P50": 18.0},
            {"شناسه": 2, "عنوان پروژه": "بومی‌سازی کاتالیست احیا مستقیم", "هزینه P10": 3500.0, "هزینه P50": 4500.0, "هزینه P90": 6000.0, "منافع P10": 7000.0, "منافع P50": 9500.0, "منافع P90": 12000.0, "توقف P50": 22.0},
            {"شناسه": 3, "عنوان پروژه": "کاهش مصرف نسوز کوره قوس", "هزینه P10": 2000.0, "هزینه P50": 2800.0, "هزینه P90": 3800.0, "منافع P10": 3800.0, "منافع P50": 5200.0, "منافع P90": 6800.0, "توقف P50": 15.0}
        ]
    else:
        default_data = db_projects

    st.subheader("⚙️ ۱. تنظیم قیود کلان سازمان")
    col1, col2 = st.columns(2)
    with col1:
        max_budget = st.number_input("سقف کل بودجه سالانه مصوب R&D (میلیون تومان):", value=10000.0, step=1000.0)
    with col2:
        max_downtime = st.number_input("حداکثر زمان مجاز توقف خط برای اجرای پروژه‌ها (ساعت):", value=50.0, step=5.0)

    st.markdown("---")
    st.subheader("📋 ۲. جدول پروژه‌های نامزد (با قابلیت حذف و ویرایش مستقیم)")
    
    # نمایش جدول با قابلیت اضافه/حذف ردیف (num_rows="dynamic")
    edited_df = st.data_editor(
        pd.DataFrame(default_data),
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor_table"
    )

    # بخش ابزار حذف دائمی از دیتابیس
    with st.expander("🗑️ **ابزار حذف قطعی یک پروژه از بانک اطلاعاتی**", expanded=False):
        col_del1, col_del2 = st.columns([3, 1])
        with col_del1:
            del_id = st.number_input("شناسه (ID) پروژه‌ای که قصد حذف دائمی آن را دارید وارد کنید:", min_value=1, step=1, key="portfolio_del_id_input")
        with col_del2:
            st.write("")
            st.write("")
            if st.button("حذف دائمی از دیتابیس", type="secondary"):
                try:
                    delete_proposal_by_id(db_session_factory, int(del_id))
                    st.success(f"✅ پروژه با شناسه {del_id} با موفقیت از پایگاه داده حذف شد.")
                    st.rerun()
                except RecordNotFoundError:
                    st.error(f"❌ پروژه‌ای با شناسه {del_id} در پایگاه داده یافت نشد.")
                except DatabaseTransactionError as e:
                    st.error(f"⛔ خطای تراکنش در حذف رکورد: {e}")

    # تبدیل داده‌های جدول ویرایش‌شده به فرمت ورودی الگوریتم
    projects_input = []
    for idx, row in edited_df.iterrows():
        c_p50 = float(row.get("هزینه P50", 0.0))
        b_p50 = float(row.get("منافع P50", 0.0))
        dt_p50 = float(row.get("توقف P50", 0.0))
        
        projects_input.append({
            "id": idx + 1,
            "title": str(row.get("عنوان پروژه", f"پروژه {idx+1}")),
            "cost_p10": float(row.get("هزینه P10", c_p50 * 0.8)),
            "cost_p50": c_p50,
            "cost_p90": float(row.get("هزینه P90", c_p50 * 1.3)),
            "benefit_p10": float(row.get("منافع P10", b_p50 * 0.7)),
            "benefit_p50": b_p50,
            "benefit_p90": float(row.get("منافع P90", b_p50 * 1.3)),
            "dt_p10": dt_p50 * 0.8,
            "dt_p50": dt_p50,
            "dt_p90": dt_p50 * 1.3
        })

    st.markdown("---")
    tab_single, tab_compare = st.tabs([
        "🎯 اجرای چیدمان بر اساس یک رویکرد خاص",
        "📊 مقایسه هم‌زمان هر ۳ رویکرد بهینه‌سازی"
    ])

    # تب ۱: بهینه‌سازی تکی
    with tab_single:
        method_choice = st.selectbox(
            "رویکرد بهینه‌سازی در شرایط عدم‌قطعیت:",
            ["برنامه‌ریزی تصادفی (Stochastic)", "برنامه‌ریزی استوار (Robust - بدترین سناریو)", "بهینه‌سازی ریسک دم (Mean-CVaR)"]
        )
        method_map = {
            "برنامه‌ریزی تصادفی (Stochastic)": "stochastic",
            "برنامه‌ریزی استوار (Robust - بدترین سناریو)": "robust",
            "بهینه‌سازی ریسک دم (Mean-CVaR)": "cvar"
        }

        if st.button("🚀 اجرای بهینه‌سازی و استخراج سبد مصوب"):
            if not projects_input:
                st.warning("⚠️ هیچ پروژه‌ای در جدول برای بهینه‌سازی وجود ندارد.")
            else:
                engine = PortfolioOptimizationEngine(num_scenarios=1000)
                res = engine.optimize_portfolio(
                    projects=projects_input,
                    max_budget=float(max_budget),
                    max_downtime_hours=float(max_downtime),
                    method=method_map[method_choice]
                )

                if res.get("status") == "OPTIMAL" and res.get("selected_projects"):
                    st.success(f"✅ ترکیب بهینه بر اساس **{method_choice}** تعیین شد.")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("مجموع سود خالص (NPV)", f"{res['total_selected_npv']:,.0f} م.ت")
                    m2.metric("بودجه مصرف‌شده", f"{res['total_selected_cost']:,.0f} م.ت ({res['budget_utilization_pct']}%)")
                    m3.metric("مجموع ساعات توقف خط", f"{res['total_selected_downtime']} ساعت ({res['downtime_utilization_pct']}%)")

                    st.subheader("🎯 پروژه‌های منتخب برای تصویب:")
                    
                    table_rows = []
                    for p in res["selected_projects"]:
                        table_rows.append({
                            "عنوان پروژه تصویب‌شده": p.get("title", "بدون عنوان"),
                            "هزینه مؤثر (م.ت)": f"{p.get('effective_cost', p.get('cost_p50', 0.0)):,.1f}",
                            "سودآوری پیش‌بینی‌شده (م.ت)": f"{p.get('calculated_score_npv', p.get('benefit_p50', 0.0)):,.1f}",
                            "زمان توقف خط (ساعت)": f"{p.get('effective_downtime', p.get('dt_p50', 0.0)):,.1f}"
                        })
                    
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
                else:
                    st.error("⚠️ پاسخی که هم‌زمان شرایط بودجه و توقفات را تأمین کند یافت نشد یا هیچ پروژه‌ای انتخاب نگردید.")

    # تب ۲: مقایسه سه‌گانه مدل‌ها
    with tab_compare:
        st.info("💡 **راهنما:** با فشردن دکمه زیر، سیستم هر ۳ مدل را هم‌زمان اجرا کرده و نتایج آن‌ها را در یک ماتریس مقایسه‌ای قرار می‌دهد.")
        if st.button("⚖️ اجرای مقایسه هم‌زمان ۳ رویکرد بهینه‌سازی"):
            if not projects_input:
                st.warning("⚠️ هیچ پروژه‌ای در جدول برای مقایسه وجود ندارد.")
            else:
                reporter = PortfolioReporter()
                with st.spinner("در حال اجرای سناریوهای مقایسه‌ای..."):
                    df_comp = reporter.compare_all_methods(
                        projects=projects_input,
                        max_budget=float(max_budget),
                        max_downtime_hours=float(max_downtime)
                    )
                st.dataframe(df_comp, use_container_width=True)