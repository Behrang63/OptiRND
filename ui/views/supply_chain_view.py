import streamlit as st
import numpy as np
import pandas as pd
from core.supply_chain_engine import SupplyChainEngine

def render_supply_chain_view():
    """
    نمایش صفحه شبیه‌سازی تأخیرات زنجیره تأمین همراه با راهنماهای شفاف و تحلیلی.
    """
    st.header("🚚 شبیه‌سازی ریسک زنجیره تأمین و تأخیر در تحویل تجهیزات (Lead-Time)")
    
    with st.expander("📖 **راهنمای شبیه‌سازی زنجیره تأمین (کلیک کنید)**", expanded=False):
        st.markdown("""
        این بخش برای ارزیابی ریسک‌های ناشی از تأخیر در تأمین قطعات حساس، مواد اولیه وارداتی و تجهیزات های‌تک است:
        * **زمان مصوب تحویل (Planned Lead-Time):** زمان‌بندی تعهدشده در برنامه زمان‌بندی پروژه (روز).
        * **سه‌گانه روزهای تحویل واقعی (P10/P50/P90):** حداقل، محتمل‌ترین و حداکثر زمانی که فرآیند خرید، ترخیص گمرکی و حمل طول می‌کشد.
        * **خسارت روزانه تأخیر:** هزینه راکد ماندن نیروها یا عدم اجرای به‌موقع پروژه به ازای هر روز تأخیر.
        * **آستانه تأخیر بحرانی:** حداکثر روزهای تأخیری که فراتر از آن، کل پروژه با ریسک شکست مواجه می‌شود.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 زمان‌بندی تحویل تجهیز (روز)")
        planned_time = st.number_input("زمان مصوب و برنامه‌ریزی‌شده تحویل (روز):", value=90.0, step=5.0)
        critical_threshold = st.number_input("آستانه تأخیر بحرانی (روز):", value=45.0, step=5.0, help="تأخیری که پروژه را بحرانی می‌کند.")

    with col2:
        st.subheader("⏱️ سه‌گانه پیش‌بینی زمان تحویل واقعی (روز)")
        lead_p10 = st.number_input("خوش‌بینانه (P10 - ترخیص و حمل بسیار سریع):", value=75.0, step=5.0)
        lead_p50 = st.number_input("محتمل (P50 - شرایط عادی بازار):", value=110.0, step=5.0)
        lead_p90 = st.number_input("بدبینانه (P90 - چالش‌های گمرکی و تحریم):", value=180.0, step=5.0)

    st.markdown("---")
    st.subheader("💰 هزینه روزانه تأخیر")
    daily_cost = st.number_input("خسارت یا هزینه بالاسری هر روز تأخیر (میلیون تومان):", value=12.0, step=1.0)

    if st.button("🚀 اجرای شبیه‌سازی مونت‌کارلو زنجیره تأمین"):
        engine = SupplyChainEngine(num_simulations=10000)
        
        with st.spinner("در حال شبیه‌سازی سناریوهای زنجیره تأمین..."):
            results = engine.simulate_lead_time_risk(
                planned_lead_time_days=float(planned_time),
                actual_lead_time_triplet=(float(lead_p10), float(lead_p50), float(lead_p90)),
                daily_delay_cost=float(daily_cost),
                critical_delay_threshold=float(critical_threshold)
            )

        st.success("✅ شبیه‌سازی زنجیره تأمین با موفقیت انجام شد.")
        
        # کارت‌های نتایج
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("میانگین کل زمان تحویل", f"{results['mean_lead_time_days']} روز")
        m2.metric("میانگین روزهای تأخیر", f"{results['mean_delay_days']} روز")
        m3.metric("میانگین خسارت تأخیر", f"{results['mean_delay_cost']:,.0f} م.ت")
        m4.metric("احتمال تأخیر بحرانی", f"{results['prob_critical_delay_pct']}%")

        st.info(f"💡 **در بدترین سناریوها (صدک ۹۵٪):** پروژه ممکن است تا **{results['max_delay_days_p95']} روز تأخیر** و **{results['max_delay_cost_p95']:,.0f} میلیون تومان خسارت** را تجربه کند.")

        # نمودار توزیع تأخیرات
        st.subheader("📈 توزیع احتمالاتی روزهای تأخیر پروژه (روز)")
        raw_delays = results['raw_delay_days']
        counts, bin_edges = np.histogram(raw_delays, bins=25)
        labels = [f"{int(bin_edges[i])} روز" for i in range(len(counts))]
        
        chart_data = pd.DataFrame({"روزهای تأخیر": labels, "تعداد سناریوها": counts}).set_index("روزهای تأخیر")
        st.bar_chart(chart_data)