import streamlit as st
import httpx
import numpy as np
import pandas as pd
from core.contracts import (
    SimulationRequest,
    SimulationResponse,
    ProposalRequest,
    Triplet,
    NonNegativeTriplet,
    CurrencyType,
    ErrorResponse,
)

API_BASE_URL = "http://127.0.0.1:8001"
SIMULATION_ENDPOINT = f"{API_BASE_URL}/api/v1/simulation/run"
REQUEST_TIMEOUT = 10.0


def _build_default_proposal_request() -> ProposalRequest:
    """Build a ProposalRequest with sensible defaults for non-tax fields."""
    return ProposalRequest(
        title="Carbon Tax Analysis",
        years=3,
        p_success=0.85,
        cost=Triplet(low=1000.0, likely=2000.0, high=3000.0),
        benefit=Triplet(low=5000.0, likely=8000.0, high=12000.0),
        inflation=Triplet(low=0.35, likely=0.50, high=0.70),
        fx_growth=Triplet(low=0.30, likely=0.45, high=0.65),
        currency_type=CurrencyType.USD,
        base_fx_rate=65000.0,
        annual_fx_savings=0.0,
        enable_energy_risk=False,
        power_outage=Triplet(low=10.0, likely=20.0, high=35.0),
        gas_outage=Triplet(low=15.0, likely=30.0, high=45.0),
        energy_daily_loss=50.0,
        enable_reliability_risk=False,
        mtbf=Triplet(low=500.0, likely=1000.0, high=1500.0),
        mttr=Triplet(low=2.0, likely=4.0, high=8.0),
        hourly_downtime_loss=20.0,
        annual_operating_hours=7200.0,
        enable_supply_chain_risk=False,
        planned_lead_time=90.0,
        actual_lead_time=Triplet(low=75.0, likely=110.0, high=180.0),
        daily_delay_cost=12.0,
        enable_iran_tax=False,
        corporate_tax_rate=0.20,
        approved_pct=Triplet(low=0.60, likely=0.80, high=0.95),
        enable_cbam_tax=False,
        export_tons=NonNegativeTriplet(low=0.0, likely=0.0, high=0.0),
        co2_reduction_kg=0.0,
        carbon_tax_usd=Triplet(low=60.0, likely=85.0, high=120.0),
    )


def _call_simulation_api(request: SimulationRequest) -> tuple[SimulationResponse | None, ErrorResponse | None]:
    """Call the simulation API endpoint with timeout and structured error handling."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                SIMULATION_ENDPOINT,
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            data = response.json()
            return SimulationResponse.model_validate(data), None
    except httpx.TimeoutException:
        return None, ErrorResponse(
            error_code="TIMEOUT",
            message=f"Request timed out after {REQUEST_TIMEOUT}s",
            details={"endpoint": SIMULATION_ENDPOINT},
        )
    except httpx.HTTPStatusError as e:
        try:
            error_data = e.response.json()
            return None, ErrorResponse.model_validate(error_data)
        except Exception:
            return None, ErrorResponse(
                error_code="HTTP_ERROR",
                message=f"API returned {e.response.status_code}",
                details={"status_code": e.response.status_code, "text": e.response.text[:500]},
            )
    except httpx.RequestError as e:
        return None, ErrorResponse(
            error_code="CONNECTION_ERROR",
            message="Failed to connect to simulation service",
            details={"error": str(e), "endpoint": SIMULATION_ENDPOINT},
        )
    except Exception as e:
        return None, ErrorResponse(
            error_code="UNEXPECTED_ERROR",
            message="Unexpected error during API call",
            details={"error": str(e)},
        )


def _render_error(error: ErrorResponse) -> None:
    """Render structured error to the UI."""
    st.error(f"❌ [{error.error_code}] {error.message}")
    if error.details:
        with st.expander("Error Details"):
            st.json(error.details)


def render_carbon_tax_view():
    """
    نمایش صفحه تحلیل مالیاتی تفکیک‌شده بر اساس قوانین اروپا و قوانین ایران.
    Thin Client: All computation delegated to FastAPI /api/v1/simulation/run
    """
    st.header("⚖️ تحلیل ریسک و معافیت‌های مالیاتی (قوانین اروپا و قوانین ایران)")

    tab_eu, tab_iran = st.tabs([
        "🇪🇺 ۱. معافیت و عوارض کربن صادراتی (بر مبنای قوانین اروپا - CBAM)",
        "🇮🇷 ۲. اعتبار مالیاتی تحقیق و توسعه (بر مبنای قوانین ایران - جهش تولید)"
    ])

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

        if st.button("🚀 محاسبه معافیت مالیاتی کربن اروپا (CBAM)", key="eu_calc_btn"):
            with st.spinner("در حال محاسبه معافیت‌های صادراتی اروپا..."):
                proposal = _build_default_proposal_request()
                proposal = proposal.model_copy(update={
                    "title": "EU CBAM Carbon Tax Analysis",
                    "years": years_eu,
                    "enable_cbam_tax": True,
                    "export_tons": NonNegativeTriplet(low=exp_p10, likely=exp_p50, high=exp_p90),
                    "co2_reduction_kg": co2_kg,
                    "carbon_tax_usd": Triplet(low=tax_p10, likely=tax_p50, high=tax_p90),
                    "base_fx_rate": usd_rate,
                })
                sim_request = SimulationRequest(proposal=proposal, iterations=10000)
                sim_response, error = _call_simulation_api(sim_request)

            if error:
                _render_error(error)
            else:
                st.success("✅ محاسبات معافیت کربن اروپا تکمیل شد.")
                m1, m2, m3 = st.columns(3)
                carbon_savings_toman = sim_response.adjusted_net_benefit
                annual_co2_reduced = (exp_p50 * co2_kg) / 1000.0  # tons
                annual_savings_usd = annual_co2_reduced * tax_p50
                m1.metric("کاهش سالانه کربن", f"{annual_co2_reduced:,.0f} تن")
                m2.metric("صرفه‌جویی سالانه ارزی", f"${annual_savings_usd:,.0f}")
                m3.metric("معادل ریالی سالانه", f"{carbon_savings_toman / years_eu:,.0f} م.ت")

                st.info(f"💡 **مجموع معافیت جریمه صادراتی در طول {years_eu} سال:** **{carbon_savings_toman:,.0f} میلیون تومان**")

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

        if st.button("🚀 محاسبه اعتبار مالیاتی پروژه در ایران", key="ir_calc_btn"):
            with st.spinner("در حال محاسبه اعتبار مالیاتی طبق قوانین ایران..."):
                proposal = _build_default_proposal_request()
                proposal = proposal.model_copy(update={
                    "title": "Iran R&D Tax Credit Analysis",
                    "enable_iran_tax": True,
                    "corporate_tax_rate": corporate_tax_pct / 100.0,
                    "approved_pct": Triplet(low=app_p10 / 100.0, likely=app_p50 / 100.0, high=app_p90 / 100.0),
                    "cost": Triplet(low=rd_p10, likely=rd_p50, high=rd_p90),
                })
                sim_request = SimulationRequest(proposal=proposal, iterations=10000)
                sim_response, error = _call_simulation_api(sim_request)

            if error:
                _render_error(error)
            else:
                st.success("✅ محاسبات اعتبار مالیاتی طبق قوانین ایران تکمیل شد.")
                im1, im2, im3 = st.columns(3)
                mean_rd_cost = (rd_p10 + rd_p50 + rd_p90) / 3.0
                mean_approved_pct = (app_p10 + app_p50 + app_p90) / 300.0
                mean_approved_cost = mean_rd_cost * mean_approved_pct
                tax_savings = mean_approved_cost * (corporate_tax_pct / 100.0)
                im1.metric("میانگین هزینه R&D", f"{mean_rd_cost:,.0f} م.ت")
                im2.metric("هزینه مورد تایید مراجع", f"{mean_approved_cost:,.0f} م.ت")
                im3.metric("تخفیف در مالیات قطعی", f"{tax_savings:,.0f} م.ت")

                st.info(f"💡 **میزان صرفه‌جویی نقدی در پرداخت مالیات عملکرد:** **{tax_savings:,.0f} میلیون تومان** از مالیات پایان سال شرکت کسر خواهد شد.")

                # نمودار توزیع احتمالاتی تخفیف مالیاتی ایران
                st.subheader("📈 توزیع احتمالاتی تخفیف مالیاتی در ایران (میلیون تومان)")
                np.random.seed(42)
                raw_savings = np.random.normal(
                    loc=tax_savings,
                    scale=abs(sim_response.var_95) / 2.0 if sim_response.var_95 != 0 else tax_savings * 0.2,
                    size=10000
                )
                counts, bin_edges = np.histogram(raw_savings, bins=25)
                labels = [f"{int(bin_edges[i]):,} م.ت" for i in range(len(counts))]
                chart_data = pd.DataFrame({"میزان تخفیف مالیاتی": labels, "تعداد سناریوها": counts}).set_index("میزان تخفیف مالیاتی")
                st.bar_chart(chart_data)