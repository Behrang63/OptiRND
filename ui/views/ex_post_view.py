import streamlit as st
from agents.main_agents.ex_post_agent import ExPostAgent

def render_ex_post_view(llm_provider):
    """
    نمایش صفحه ارزیابی پسینی و گواهی ارزش‌آفرینی به همراه متون راهنما.
    """
    st.header("🎯 ارزیابی پسینی و صدور گواهی ارزش‌آفرینی")
    
    with st.expander("📖 **راهنمای ارزیابی پسینی (کلیک کنید)**", expanded=False):
        st.markdown("""
        پس از اتمام کامل پروژه R&D و ورود به فاز بهره‌برداری، دستاورد واقعی با هزینه‌های نهایی مقایسه می‌شود:
        * **ارزش خالص ایجادشده (Net Value):** تفاضل منافع محقق‌شده از کل هزینه صرف‌شده.
        * **بازگشت سرمایه واقعی (Realized ROI):** نرخ سودآوری قطعی حاصل از اجرای پروژه.
        """)

    col1, col2 = st.columns(2)
    with col1:
        actual_cost = st.number_input("کل هزینه واقعی صرف‌شده (میلیون تومان)", value=10000.0, step=500.0)
    with col2:
        realized_benefit = st.number_input("کل منافع مالی محقق‌شده (میلیون تومان)", value=15000.0, step=500.0)

    if st.button("محاسبه ارزش‌آفرینی نهایی"):
        agent = ExPostAgent(llm_provider=llm_provider)
        res = agent.evaluate_ex_post(actual_cost=actual_cost, realized_benefit=realized_benefit)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("ارزش خالص ایجادشده", f"{res.get('net_value', 0.0)} میلیون تومان")
        m2.metric("بازگشت سرمایه واقعی (ROI)", f"{res.get('realized_roi', 0.0)}%")
        m3.metric("وضعیت گواهی", res.get('value_certificate_status', 'PENDING'))