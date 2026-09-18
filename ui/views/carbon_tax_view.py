import streamlit as st
import numpy as np
import pandas as pd
from core.carbon_tax_engine import CarbonTaxEngine

def render_carbon_tax_view():
    """
    نمایش صفحه تحلیل مالیاتی تفکیک‌شده بر اساس قوانین اروپا و قوانین ایران.
    """
    st.header("⚖️ تحلیل ریسک و معافیت‌های مالیاتی (قوانین اروپا و قوانین ایران)")
    
    tab_eu, tab_iran = st.tabs([
        "🇪🇺 ۱. معافیت و عوارض کربن صادراتی (بر مبنای قوانین اروپا - CBAM)",
        "🇮🇷 ۲. اعتبار مالیاتی تحقیق و توسعه (بر مبنای قوانین ایران - جهش تولید)"
    ])

    engine = CarbonTaxEngine(num_simulations=10000)

    # ---------------------------------------------------------
    # بخش اول: قوانین اروپا (CBAM)
    # ---------------------------------------------------------
    with tab_eu:
        with st.expander("📖 **راهنمای مقررات CBAM اتحادیه اروپا (کلیک کنید)**", expanded=False):
            st.markdown("""
            طبق مقررات کربن اتحادیه اروپا (**CBAM**)، واردات فولاد به اروپا مشمول پرداخت مالیات به‌ازای هر تن دی‌اکسید کربن مازاد است:
            * اجرای پروژه‌های R&D که انتشار کربن را کاهش دهند، موجب **معافیت از پرداخت این جریمه‌ها** در مبادی گمرکی صادرات می‌شود.
            """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🚢 تناژ صادرات سالانه به بازارهای بین‌المللی (تن)")
            exp_p10 = st.number_input("خوش‌بینانه (P10 - تن)", value=100000.0, step=10000.0, key="eu_exp_p10")
            exp_p50 = st.number_input("محتمل (P50 - تن)", value=200000.0, step=10000.0, key="eu_exp_p50")
            exp_p90 = st.number_input("بدبینانه (P90 - تن)", value=300000.0, step=10000.0, key="eu_exp_p90")
            
        with col2:
            st.subheader("💵 نرخ جهانی مالیات کربن (دلار / تن $CO_2$)")
            tax_p10 = st.number_input("خوش‌بینانه (P10 - دلار)", value=60.0, step=5.0, key="eu_tax_p10")
            tax_p50 = st.number_input("محتمل (P50 - دلار)", value=85.0, step=5.0, key="eu_tax_p50")
            tax_p90 = st.number_input("بدبینانه (P90 - دلار)", value=120.0, step=5.0, key="eu_tax_p90")

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            co2_kg = st.number_input("کاهش $CO_2$ به ازای هر تن فولاد (کیلوگرم):", value=150.0, step=10.0, key="eu_co2")
        with f_col2:
            usd_rate = st.number_input("نرخ دلار به تومان:", value=65000.0, step=500.0, key="eu_usd")
        with f_col3:
            years_eu = st.slider("دوره ارزیابی صادرات (سال):", 1, 5, 3, key="eu_years")

        if st.button("🚀 محاسبه معافیت مالیاتی کربن اروپا (CBAM)"):
            with st.spinner("در حال محاسبه معافیت‌های صادراتی اروپا..."):
                eu_results = engine.simulate_eu_cbam_benefit(
                    annual_export_tons_triplet=(float(exp_p10), float(exp_p50), float(exp_p90)),
                    co2_reduction_per_ton_kg=float(co2_kg),
                    carbon_tax_per_ton_usd_triplet=(float(tax_p10), float(tax_p50), float(tax_p90)),
                    usd_to_toman_rate=float(usd_rate),
                    years=int(years_eu)
                )

            st.success("✅ محاسبات معافیت کربن اروپا تکمیل شد.")
            m1, m2, m3 = st.columns(3)
            m1.metric("کاهش سالانه کربن", f"{eu_results['mean_annual_co2_reduced_tons']:,.0f} تن")
            m2.metric("صرفه‌جویی سالانه ارزی", f"${eu_results['mean_annual_carbon_savings_usd']:,.0f}")
            m3.metric("معادل ریالی سالانه", f"{eu_results['mean_annual_carbon_savings_toman']:,.0f} م.ت")

            st.info(f"💡 **مجموع معافیت جریمه صادراتی در طول {years_eu} سال:** **{eu_results['total_lifetime_savings_toman']:,.0f} میلیون تومان**")

    # ---------------------------------------------------------
    # بخش دوم: قوانین ایران (قانون جهش تولید دانش‌بنیان)
    # ---------------------------------------------------------
    with tab_iran:
        with st.expander("📖 **راهنمای اعتبار مالیاتی در قوانین ایران (ماده ۱۱ جهش تولید)**", expanded=False):
            st.markdown("""
            طبق **ماده ۱۱ قانون جهش تولید دانش‌بنیان در ایران**:
            * معادل هزینه‌های انجام‌شده در پروژه‌های تحقیق و توسعه (R&D) مورد تأیید، به‌عنوان **اعتبار مالیاتی** منظور می‌شود.
            * این اعتبار مستقیماً از **مالیات بر درآمد عملکرد سالانه شرکت** کسر می‌گردد.
            """)

        ir_col1, ir_col2 = st.columns(2)
        with ir_col1:
            st.subheader("💰 سه‌گانه کل هزینه R&D پروژه (میلیون تومان)")
            rd_p10 = st.number_input("خوش‌بینانه (P10)", value=8000.0, step=500.0, key="ir_rd_p10")
            rd_p50 = st.number_input("محتمل (P50)", value=10000.0, step=500.0, key="ir_rd_p50")
            rd_p90 = st.number_input("بدبینانه (P90)", value=14000.0, step=500.0, key="ir_rd_p90")

        with ir_col2:
            st.subheader("📋 سه‌گانه درصد تایید هزینه توسط مراجع قانونی")
            app_p10 = st.number_input("حداقل درصد تایید (P10 - درصد)", value=60.0, step=5.0, key="ir_app_p10")
            app_p50 = st.number_input("محتمل‌ترین درصد تایید (P50 - درصد)", value=80.0, step=5.0, key="ir_app_p50")
            app_p90 = st.number_input("حداکثر درصد تایید (P90 - درصد)", value=95.0, step=5.0, key="ir_app_p90")

        st.markdown("---")
        tax_rate_col1, tax_rate_col2 = st.columns(2)
        with tax_rate_col1:
            corporate_tax_pct = st.number_input("نرخ مالیات عملکرد شرکت (درصد):", value=20.0, step=1.0, help="نرخ مصوب مالیات عملکرد صنعتی در ایران.")

        if st.button("🚀 محاسبه اعتبار مالیاتی پروژه در ایران"):
            with st.spinner("در حال محاسبه اعتبار مالیاتی طبق قوانین ایران..."):
                iran_results = engine.simulate_iran_tax_credit(
                    rd_cost_triplet=(float(rd_p10), float(rd_p50), float(rd_p90)),
                    approved_cost_pct_triplet=(float(app_p10)/100.0, float(app_p50)/100.0, float(app_p90)/100.0),
                    corporate_tax_rate=float(corporate_tax_pct)/100.0
                )

            st.success("✅ محاسبات اعتبار مالیاتی طبق قوانین ایران تکمیل شد.")
            im1, im2, im3 = st.columns(3)
            im1.metric("میانگین هزینه R&D", f"{iran_results['mean_rd_cost_toman']:,.0f} م.ت")
            im2.metric("هزینه مورد تایید مراجع", f"{iran_results['mean_approved_rd_cost']:,.0f} م.ت")
            im3.metric("تخفیف در مالیات قطعی", f"{iran_results['mean_iran_tax_credit_savings']:,.0f} م.ت")

            st.info(f"💡 **میزان صرفه‌جویی نقدی در پرداخت مالیات عملکرد:** **{iran_results['mean_iran_tax_credit_savings']:,.0f} میلیون تومان** از مالیات پایان سال شرکت کسر خواهد شد.")

            # نمودار توزیع تخفیف مالیاتی ایران
            st.subheader("📈 توزیع احتمالاتی تخفیف مالیاتی در ایران (میلیون تومان)")
            raw_savings = iran_results['raw_tax_savings']
            counts, bin_edges = np.histogram(raw_savings, bins=25)
            labels = [f"{int(bin_edges[i]):,} م.ت" for i in range(len(counts))]
            chart_data = pd.DataFrame({"میزان تخفیف مالیاتی": labels, "تعداد سناریوها": counts}).set_index("میزان تخفیف مالیاتی")
            st.bar_chart(chart_data)