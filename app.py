import streamlit as st
from core.database import init_db
from core.llm_provider import LLMFactory
from ui.state_manager import StateManager

# وارد کردن توابع رندر نماهای یکپارچه سامانه
from ui.views.ex_ante_view import render_ex_ante_view
from ui.views.portfolio_view import render_portfolio_view
from ui.views.process_view import render_process_view
from ui.views.ex_post_view import render_ex_post_view
from ui.views.history_view import render_history_view


@st.cache_resource
def get_db_factory():
    return init_db()


def apply_custom_css():
    st.markdown("""
        <style>
            html, body, [class*="css"], div, p, span, h1, h2, h3, h4, h5, h6 {
                direction: rtl !important;
                text-align: right !important;
            }
            section[data-testid="stSidebar"] {
                direction: rtl !important;
                text-align: right !important;
            }
            .stTextInput > label, .stNumberInput > label, .stSlider > label, 
            .stFileUploader > label, .stSelectbox > label {
                text-align: right !important;
                display: block !important;
                width: 100% !important;
            }
            [data-testid="stMetric"] {
                text-align: right !important;
                direction: rtl !important;
            }
            [data-testid="stMetricLabel"] {
                justify-content: flex-end !important;
            }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="سامانه پژوهشیار",
        page_icon="🔬",
        layout="wide"
    )

    apply_custom_css()
    StateManager.initialize_state()
    db_session_factory = get_db_factory()
    llm_provider = LLMFactory.get_provider("mock")

    st.title("🔬 سامانه هوشمند ارزیابی و بهینه‌سازی سبد پروژه‌های R&D (پژوهشیار)")
    st.subheader("پلتفرم ارزیابی اقتصادی، تورمی، ناترازی انرژی، MTBF و چیدمان بهینه سبد")

    st.sidebar.header("گردش کار ارزیابی و مدیریت")
    pages = [
        "۱. ثبت و ارزیابی جامع پروژه (Project Intake)", 
        "۲. بهینه‌سازی سبد پروژه‌ها (Portfolio Selection)",
        "۳. پایش فرآیندی مایلستون‌ها (Process Monitoring)", 
        "۴. ارزیابی پسینی ارزش‌آفرینی (Ex-Post Evaluation)",
        "۵. بانک اطلاعاتی و سوابق (Project History)"
    ]
    
    selected_page = st.sidebar.radio("مرحله مورد نظر را انتخاب کنید:", pages)

    if selected_page == "۱. ثبت و ارزیابی جامع پروژه (Project Intake)":
        render_ex_ante_view(llm_provider, db_session_factory)

    elif selected_page == "۲. بهینه‌سازی سبد پروژه‌ها (Portfolio Selection)":
        render_portfolio_view(db_session_factory)

    elif selected_page == "۳. پایش فرآیندی مایلستون‌ها (Process Monitoring)":
        render_process_view(llm_provider)

    elif selected_page == "۴. ارزیابی پسینی ارزش‌آفرینی (Ex-Post Evaluation)":
        render_ex_post_view(llm_provider)

    elif selected_page == "۵. بانک اطلاعاتی و سوابق (Project History)":
        render_history_view(db_session_factory)


if __name__ == "__main__":
    main()