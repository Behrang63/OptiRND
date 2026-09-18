import streamlit as st
import numpy as np
import pandas as pd
from core.energy_simulation import EnergyRiskEngine

def render_energy_risk_view():
    """
    نمایش صفحه تخصصی شبیه‌سازی ریسک ناترازی انرژی به همراه راهنماهای جامع.
    """
    st.header("⚡ شبیه‌سازی ریسک ناترازی و محدودیت حامل‌های انرژی (برق و گاز)")
    
    with st.expander("📖 **راهنمای شبیه‌سازی ریسک انرژی (کلیک کنید)**", expanded=False):
        st.markdown("""
        در صنایع انرژی‌بر، عدم قطعیت در تأمین پایدار برق و گاز مستقیماً بر تولید اثر می‌گذارد:
        * **سناریوهای روزهای قطعی برق (تابستان):** روزهایی که خط به دلیل محدودیت دیسپاچینگ برق متوقف یا با ظرفیت کاهش‌یافته کار می‌کند.
        * **سناریوهای روزهای قطعی گاز (زمستان):** روزهای ناترازی گاز در فصل سرد سال.
        * **هزینه توقف روزانه:** برآورد ارزش تولید از دست رفته یا هزینه‌های ثابت بر زمین مانده به‌ازای هر روز توقف.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("☀️ سناریوی ناترازی برق تابستان (روز)")
        power_p10 = st.number_input("خوش‌بینانه (P10 - روز)", value=10.0, step=1.0)
        power_p50 = st.number_input("محتمل (P50 - روز)", value=20.0, step=1.0)
        power_p90 = st.number_input("بدبینانه (P90 - روز)", value=35.0, step=1.0)
        
    with col2:
        st.subheader("❄️ سناریوی ناترازی گاز زمستان (روز)")
        gas_p10 = st.number_input("خوش‌بینانه (P10 - روز)", value=15.0, step=1.0)
        gas_p50 = st.number_input("محتمل (P50 - روز)", value=30.0, step=1.0)
        gas_p90 = st.number_input("بدبینانه (P90 - روز)", value=45.0, step=1.0)

    st.markdown("---")
    st.subheader("💰 ابعاد مالی و زیان توقف خط")
    
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        base_benefit = st.number_input("منفعت سالانه پیش‌بینی‌شده پروژه (میلیون تومان):", value=5000.0, step=500.0)
    with f_col2:
        daily_loss = st.number_input("زیان روزانه توقف فرآیند ناشی از انرژی (میلیون تومان):", value=50.0, step=5.0)

    years = st.slider("دوره ارزیابی (سال):", 1, 5, 3)

    if st.button("🚀 اجرای شبیه‌سازی مونت‌کارلو ناترازی انرژی"):
        engine = EnergyRiskEngine(num_simulations=10000)
        
        with st.spinner("در حال شبیه‌سازی اثرات ناترازی انرژی..."):
            results = engine.simulate_energy_impact(
                base_annual_benefit=float(base_benefit),
                power_outage_days_triplet=(float(power_p10), float(power_p50), float(power_p90)),
                gas_outage_days_triplet=(float(gas_p10), float(gas_p50), float(gas_p90)),
                daily_downtime_loss=float(daily_loss),
                years=int(years)
            )

        st.success("✅ محاسبات شبیه‌سازی ریسک انرژی تکمیل شد.")
        
        # نمایش کارت‌های شاخص
        m1, m2, m3 = st.columns(3)
        m1.metric("میانگین روزهای قطعی سالانه", f"{results['mean_outage_days_yearly']} روز")
        m2.metric("خسارت سالانه پیش‌بینی‌شده", f"{results['mean_annual_loss_toman']:,.0f} م.ت")
        m3.metric("منفعت خالص تعدیل‌شده سالانه", f"{results['adjusted_annual_benefit']:,.0f} م.ت")

        # نمودار توزیع خسارت
        st.subheader("📈 توزیع احتمالاتی خسارت سالانه انرژی (میلیون تومان)")
        raw_losses = results['raw_loss_samples']
        counts, bin_edges = np.histogram(raw_losses, bins=25)
        labels = [f"{int(bin_edges[i]):,} م.ت" for i in range(len(counts))]
        
        chart_data = pd.DataFrame({"میزان خسارت": labels, "تعداد سناریوها": counts}).set_index("میزان خسارت")
        st.bar_chart(chart_data)