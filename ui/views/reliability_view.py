import streamlit as st
import numpy as np
import pandas as pd
from core.reliability_engine import ReliabilityEngine

def render_reliability_view():
    """
    نمایش صفحه شبیه‌سازی قابلیت اطمینان و توقفات خط تولید با راهنماهای تحلیلی.
    """
    st.header("⚙️ شبیه‌سازی قابلیت اطمینان تجهیزات و ریسک توقف خط تولید (MTBF / Downtime)")
    
    with st.expander("📖 **راهنمای شبیه‌سازی قابلیت اطمینان (کلیک کنید)**", expanded=False):
        st.markdown("""
        این بخش برای ارزیابی ریسک‌های عملیاتی ناشی از نقص فنی تجهیزات جدید در خط تولید است:
        * **شاخص MTBF (فاصله زمانی بین خرابی‌ها):** میانگین ساعت کارکرد پایدار تجهیز قبل از بروز اشکال بعدی (هرچه بیشتر باشد، بهتر است).
        * **شاخص MTTR (زمان لازم برای تعمیر):** میانگین ساعتی که تیم نگهداری و تعمیرات برای رفع خرابی نیاز دارد.
        * **ضریب در دسترس بودن (Availability):** درصدی از زمان مفید کاری سالانه که خط بدون توقف ناشی از این تجهیز به کار خود ادامه می‌دهد.
        * **زیان توقف ساعتی:** ارزش اقتصادی عدم تولید به ازای هر ساعت توقف کامل خط.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⏱️ تخمین MTBF (فاصله بین خرابی‌ها - ساعت)")
        mtbf_p10 = st.number_input("خوش‌بینانه (P10 - ساعت)", value=1500.0, step=100.0, help="حداکثر کارکرد مداوم بدون خرابی")
        mtbf_p50 = st.number_input("محتمل (P50 - ساعت)", value=1000.0, step=100.0)
        mtbf_p90 = st.number_input("بدبینانه (P90 - ساعت)", value=500.0, step=100.0, help="حداقل کارکرد بدون خرابی")
        
    with col2:
        st.subheader("🛠️ تخمین MTTR (زمان تعمیر - ساعت)")
        mttr_p10 = st.number_input("خوش‌بینانه (P10 - ساعت)", value=2.0, step=0.5, help="سریع‌ترین زمان تعمیر")
        mttr_p50 = st.number_input("محتمل (P50 - ساعت)", value=4.0, step=0.5)
        mttr_p90 = st.number_input("بدبینانه (P90 - ساعت)", value=8.0, step=0.5, help="طولانی‌ترین زمان تعمیر")

    st.markdown("---")
    st.subheader("🏭 شرایط عملیاتی و هزینه توقف خط")
    
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        annual_hours = st.number_input("کل ساعات کاری سالانه خط (ساعت):", value=7200.0, step=200.0)
    with f_col2:
        hourly_loss = st.number_input("خسارت هر ساعت توقف خط (میلیون تومان):", value=20.0, step=2.0)
    with f_col3:
        base_benefit = st.number_input("منفعت سالانه پیش‌بینی‌شده پروژه (میلیون تومان):", value=6000.0, step=500.0)

    years = st.slider("افق زمانی بررسی (سال):", 1, 5, 3)

    if st.button("🚀 اجرای شبیه‌سازی قابلیت اطمینان و توقفات"):
        engine = ReliabilityEngine(num_simulations=10000)
        
        with st.spinner("در حال شبیه‌سازی نرخ خرابی و اثر توقف بر خط تولید..."):
            results = engine.simulate_downtime_risk(
                base_annual_benefit=float(base_benefit),
                mtbf_hours_triplet=(float(mtbf_p10), float(mtbf_p50), float(mtbf_p90)),
                mttr_hours_triplet=(float(mttr_p10), float(mttr_p50), float(mttr_p90)),
                hourly_downtime_loss=float(hourly_loss),
                annual_operating_hours=float(annual_hours),
                years=int(years)
            )

        st.success("✅ شبیه‌سازی قابلیت اطمینان با موفقیت محاسبه شد.")
        
        # نمایش کارت‌های متریک
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("دفعات خرابی سالانه", f"{results['mean_annual_failures']} بار")
        m2.metric("کل ساعت توقف سالانه", f"{results['mean_downtime_hours']} ساعت")
        m3.metric("درصد دسترسی‌پذیری تجهیز", f"{results['mean_availability_pct']}%")
        m4.metric("زیان مالی توقف سالانه", f"{results['mean_annual_downtime_loss']:,.0f} م.ت")

        st.info(f"💡 **منفعت سالانه تعدیل‌شده پروژه پس از کسر خسارت توقفات:** **{results['adjusted_annual_benefit']:,.0f} میلیون تومان**")

        # نمودار توزیع احتمال ساعات توقف
        st.subheader("📈 توزیع احتمالاتی ساعات توقف سالانه خط")
        raw_downtime = results['raw_downtime_hours']
        counts, bin_edges = np.histogram(raw_downtime, bins=25)
        labels = [f"{int(bin_edges[i])} ساعت" for i in range(len(counts))]
        
        chart_data = pd.DataFrame({"ساعات توقف": labels, "تعداد سناریوها": counts}).set_index("ساعات توقف")
        st.bar_chart(chart_data)