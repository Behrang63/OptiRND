import streamlit as st
from agents.main_agents.process_agent import ProcessAgent

def render_process_view(llm_provider):
    """
    نمایش صفحه پایش فرآیندی و سنجش انحراف مایلستون‌ها همراه با راهنما.
    """
    st.header("📊 پایش فرآیندی و ردیابی انحرافات مایلستون‌ها")
    
    with st.expander("📖 **راهنمای پایش فرآیندی (کلیک کنید)**", expanded=False):
        st.markdown("""
        این بخش برای ارزیابی عملکرد واقعی در حین اجرای پروژه است:
        * **انحراف هزینه (Cost Variance):** تفاوت بودجه مصوب فاز جاری با هزینه واقعی مصرف‌شده.
        * **انحراف زمان (Schedule Variance):** تفاوت زمان مصوب با زمان واقعی صرف‌شده.
        * **سطح هشدار:** سیستم به‌صورت خودکار بر اساس آستانه‌های انحراف وضعیت را در یکی از سطوح *عادی (ON_TRACK)*، *هشدار (WARNING)* یا *بحرانی (CRITICAL)* قرار می‌دهد.
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("بودجه مایلستون (میلیون تومان)")
        planned_budget = st.number_input("بودجه مصوب (برنامه‌ریزی‌شده)", value=1000.0, step=100.0)
        actual_budget = st.number_input("بودجه مصرف‌شده (واقعی)", value=1200.0, step=100.0)
        
    with col2:
        st.subheader("زمان‌بندی مایلستون (روز)")
        planned_days = st.number_input("زمان مصوب (برنامه‌ریزی‌شده)", value=30, step=5)
        actual_days = st.number_input("زمان صرف‌شده (واقعی)", value=38, step=5)

    if st.button("تحلیل انحرافات و وضعیت مایلستون"):
        agent = ProcessAgent(llm_provider=llm_provider)
        milestones = [{
            "title": "فاز جاری",
            "planned_budget": planned_budget,
            "actual_budget": actual_budget,
            "planned_duration_days": planned_days,
            "actual_duration_days": actual_days
        }]
        res = agent.monitor_process(milestones)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("انحراف هزینه", f"{res.get('cost_variance_pct', 0.0)}%")
        m2.metric("انحراف زمان", f"{res.get('schedule_variance_pct', 0.0)}%")
        m3.metric("وضعیت هشدار", res.get('status', 'NORMAL'))
        st.info(f"💡 **توصیه سیستم:** {res.get('mitigation_advice', 'اقدامی ثبت نشده است.')}")