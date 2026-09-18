import streamlit as st
import pandas as pd
import httpx
from core.contracts import (
    PortfolioOptimizationRequest,
    PortfolioProjectRequest,
    PortfolioDecisionResponse,
    ErrorResponse,
    Triplet,
)

API_BASE_URL = "http://127.0.0.1:8001"
OPTIMIZE_ENDPOINT = f"{API_BASE_URL}/api/v1/portfolio/optimize"
REQUEST_TIMEOUT = 10.0


def _build_projects_payload(edited_df: pd.DataFrame) -> list[PortfolioProjectRequest]:
    projects = []
    for idx, row in edited_df.iterrows():
        c_p50 = float(row.get("هزینه P50", 0.0))
        b_p50 = float(row.get("منافع P50", 0.0))
        dt_p50 = float(row.get("توقف P50", 0.0))

        projects.append(
            PortfolioProjectRequest(
                title=str(row.get("عنوان پروژه", f"پروژه {idx+1}")),
                cost=Triplet(
                    low=float(row.get("هزینه P10", c_p50 * 0.8)),
                    likely=c_p50,
                    high=float(row.get("هزینه P90", c_p50 * 1.3)),
                ),
                benefit=Triplet(
                    low=float(row.get("منافع P10", b_p50 * 0.7)),
                    likely=b_p50,
                    high=float(row.get("منافع P90", b_p50 * 1.3)),
                ),
                downtime=Triplet(
                    low=dt_p50 * 0.8,
                    likely=dt_p50,
                    high=dt_p50 * 1.3,
                ),
            )
        )
    return projects


def _call_optimize_api(
    projects: list[PortfolioProjectRequest],
    budget_limit: float,
    max_downtime_hours: float,
) -> PortfolioDecisionResponse | ErrorResponse:
    request_dto = PortfolioOptimizationRequest(
        projects=projects,
        budget_limit=budget_limit,
        max_downtime_hours=max_downtime_hours,
    )

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                OPTIMIZE_ENDPOINT,
                json=request_dto.model_dump(mode="json"),
            )
            response.raise_for_status()
            return PortfolioDecisionResponse.model_validate(response.json())
    except httpx.TimeoutException:
        return ErrorResponse(
            error_code="TIMEOUT",
            message=f"Request to {OPTIMIZE_ENDPOINT} timed out after {REQUEST_TIMEOUT}s",
        )
    except httpx.HTTPStatusError as e:
        try:
            return ErrorResponse.model_validate(e.response.json())
        except Exception:
            return ErrorResponse(
                error_code="HTTP_ERROR",
                message=f"Server returned {e.response.status_code}: {e.response.text[:200]}",
            )
    except httpx.RequestError as e:
        return ErrorResponse(
            error_code="CONNECTION_ERROR",
            message=f"Failed to connect to optimization service: {e}",
        )


def _render_decision_response(response: PortfolioDecisionResponse) -> None:
    if response.status == "OPTIMAL" and response.selected_titles:
        st.success(f"✅ ترکیب بهینه بر اساس **{response.method_used}** تعیین شد.")
        m1, m2, m3 = st.columns(3)
        m1.metric("مجموع سود خالص (NPV)", f"{response.total_selected_npv:,.0f} م.ت")
        m2.metric(
            "بودجه مصرف‌شده",
            f"{response.total_selected_cost:,.0f} م.ت ({response.budget_utilization_pct}%)",
        )
        m3.metric(
            "مجموع ساعات توقف خط",
            f"{response.total_selected_downtime} ساعت ({response.downtime_utilization_pct}%)",
        )

        st.subheader("🎯 پروژه‌های منتخب برای تصویب:")
        table_rows = []
        for title in response.selected_titles:
            table_rows.append({"عنوان پروژه تصویب‌شده": title})
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.error(
            "⚠️ پاسخی که هم‌زمان شرایط بودجه و توقفات را تأمین کند یافت نشد یا هیچ پروژه‌ای انتخاب نگردید."
        )


def _render_error_response(error: ErrorResponse) -> None:
    st.error(f"⛔ {error.error_code}: {error.message}")
    if error.details:
        with st.expander("جزئیات خطا"):
            st.json(error.details)


def render_portfolio_view() -> None:
    """
    نمایش صفحه بهینه‌سازی، مقایسه هم‌زمان سناریوها و مدیریت و حذف پروژه‌های سبد R&D.
    Thin Client: تمام محاسبه‌ها از طریق HTTP به FastAPI Server (پورت 8001) ارسال می‌شوند.
    """
    st.header("📊 ۲. بهینه‌سازی و چیدمان سبد پروژه‌های R&D")

    with st.expander("📖 **راهنمای ویرایش، حذف و انتخاب الگوریتم (کلیک کنید)**", expanded=False):
        st.markdown("""
        در این بخش می‌توانید پروژه‌ها را مدیریت کرده و سبد مصوب را استخراج نمایید:
        * **حذف موقت از تحلیل جاری:** روی شماره سمت راست ردیف در جدول کلیک کرده و دکمه **Delete** کیبورد را بزنید.
        * **رویکردهای بهینه‌سازی:**
          1. **تصادفی (Stochastic):** حداکثرسازی سود مورد انتظار در شرایط نوسانات ملایم.
          2. **استوار (Robust):** مصون‌سازی سبد در برابر بدترین سناریوهای احتمالی هزینه و توقف.
          3. **ارزش در معرض ریسک شرطی (Mean-CVaR):** کنترل زیان‌های حاد و سناریوهای بحرانی دم توزیع.
        """)

    default_data = [
        {
            "شناسه": 1,
            "عنوان پروژه": "سامانه هوشمند عیب‌یابی خط نورد گرم",
            "هزینه P10": 3000.0,
            "هزینه P50": 4000.0,
            "هزینه P90": 5500.0,
            "منافع P10": 5000.0,
            "منافع P50": 7200.0,
            "منافع P90": 9800.0,
            "توقف P50": 18.0,
        },
        {
            "شناسه": 2,
            "عنوان پروژه": "بومی‌سازی کاتالیست احیا مستقیم",
            "هزینه P10": 3500.0,
            "هزینه P50": 4500.0,
            "هزینه P90": 6000.0,
            "منافع P10": 7000.0,
            "منافع P50": 9500.0,
            "منافع P90": 12000.0,
            "توقف P50": 22.0,
        },
        {
            "شناسه": 3,
            "عنوان پروژه": "کاهش مصرف نسوز کوره قوس",
            "هزینه P10": 2000.0,
            "هزینه P50": 2800.0,
            "هزینه P90": 3800.0,
            "منافع P10": 3800.0,
            "منافع P50": 5200.0,
            "منافع P90": 6800.0,
            "توقف P50": 15.0,
        },
    ]

    st.subheader("⚙️ ۱. تنظیم قیود کلان سازمان")
    col1, col2 = st.columns(2)
    with col1:
        max_budget = st.number_input(
            "سقف کل بودجه سالانه مصوب R&D (میلیون تومان):", value=10000.0, step=1000.0
        )
    with col2:
        max_downtime = st.number_input(
            "حداکثر زمان مجاز توقف خط برای اجرای پروژه‌ها (ساعت):", value=50.0, step=5.0
        )

    st.markdown("---")
    st.subheader("📋 ۲. جدول پروژه‌های نامزد (با قابلیت حذف و ویرایش مستقیم)")

    edited_df = st.data_editor(
        pd.DataFrame(default_data),
        num_rows="dynamic",
        use_container_width=True,
        key="portfolio_editor_table",
    )

    projects_input = _build_projects_payload(edited_df)

    st.markdown("---")
    tab_single, tab_compare = st.tabs([
        "🎯 اجرای چیدمان بر اساس یک رویکرد خاص",
        "📊 مقایسه هم‌زمان هر ۳ رویکرد بهینه‌سازی",
    ])

    with tab_single:
        method_choice = st.selectbox(
            "رویکرد بهینه‌سازی در شرایط عدم‌قطعیت:",
            [
                "برنامه‌ریزی تصادفی (Stochastic)",
                "برنامه‌ریزی استوار (Robust - بدترین سناریو)",
                "بهینه‌سازی ریسک دم (Mean-CVaR)",
            ],
        )
        method_map = {
            "برنامه‌ریزی تصادفی (Stochastic)": "stochastic",
            "برنامه‌ریزی استوار (Robust - بدترین سناریو)": "robust",
            "بهینه‌سازی ریسک دم (Mean-CVaR)": "cvar",
        }

        if st.button("🚀 اجرای بهینه‌سازی و استخراج سبد مصوب"):
            if not projects_input:
                st.warning("⚠️ هیچ پروژه‌ای در جدول برای بهینه‌سازی وجود ندارد.")
            else:
                with st.spinner("در حال ارسال درخواست به سرور بهینه‌سازی..."):
                    result = _call_optimize_api(
                        projects=projects_input,
                        budget_limit=float(max_budget),
                        max_downtime_hours=float(max_downtime),
                    )

                if isinstance(result, ErrorResponse):
                    _render_error_response(result)
                else:
                    _render_decision_response(result)

    with tab_compare:
        st.info(
            "💡 **راهنما:** با فشردن دکمه زیر، سیستم هر ۳ مدل را هم‌زمان اجرا کرده و نتایج آن‌ها را در یک ماتریس مقایسه‌ای قرار می‌دهد."
        )
        if st.button("⚖️ اجرای مقایسه هم‌زمان ۳ رویکرد بهینه‌سازی"):
            if not projects_input:
                st.warning("⚠️ هیچ پروژه‌ای در جدول برای مقایسه وجود ندارد.")
            else:
                st.warning("⚠️ مقایسه هم‌زمان در نسخه Thin Client فعلاً پشتیبانی نمی‌شود. لطفاً از تب «اجرا» برای هر رویکرد جداگانه استفاده کنید.")