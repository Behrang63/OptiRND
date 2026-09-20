import streamlit as st
import httpx
from core.contracts import (
    SimulationRequest,
    SimulationResponse,
    ErrorResponse,
    ProposalRequest,
    Triplet,
    CurrencyType,
    NonNegativeTriplet,
)

API_BASE_URL = "http://127.0.0.1:8001"
SIMULATION_ENDPOINT = f"{API_BASE_URL}/api/v1/simulation/run"
REQUEST_TIMEOUT = 10.0


def _build_proposal_request(
    power_p10: float,
    power_p50: float,
    power_p90: float,
    gas_p10: float,
    gas_p50: float,
    gas_p90: float,
    base_benefit: float,
    daily_loss: float,
    years: int,
) -> ProposalRequest:
    """Construct a ProposalRequest from UI inputs for energy risk simulation."""
    return ProposalRequest(
        title="Energy Risk Simulation",
        years=years,
        p_success=0.85,
        cost=Triplet(low=0.0, likely=0.0, high=0.0),
        benefit=Triplet(low=base_benefit, likely=base_benefit, high=base_benefit),
        inflation=Triplet(low=0.35, likely=0.50, high=0.70),
        fx_growth=Triplet(low=0.30, likely=0.45, high=0.65),
        currency_type=CurrencyType.USD,
        base_fx_rate=65000.0,
        annual_fx_savings=0.0,
        enable_energy_risk=True,
        power_outage=Triplet(low=power_p10, likely=power_p50, high=power_p90),
        gas_outage=Triplet(low=gas_p10, likely=gas_p50, high=gas_p90),
        energy_daily_loss=daily_loss,
        enable_reliability_risk=False,
        mtbf=Triplet(low=500.0, likely=1000.0, high=1500.0),
        mttr=Triplet(low=2.0, likely=4.0, high=8.0),
        hourly_downtime_loss=0.0,
        annual_operating_hours=7200.0,
        enable_supply_chain_risk=False,
        planned_lead_time=90.0,
        actual_lead_time=Triplet(low=75.0, likely=110.0, high=180.0),
        daily_delay_cost=0.0,
        enable_iran_tax=False,
        corporate_tax_rate=0.20,
        approved_pct=Triplet(low=0.60, likely=0.80, high=0.95),
        enable_cbam_tax=False,
        export_tons=NonNegativeTriplet(low=0.0, likely=0.0, high=0.0),
        co2_reduction_kg=0.0,
        carbon_tax_usd=Triplet(low=60.0, likely=85.0, high=120.0),
    )


def _run_simulation_via_api(request: SimulationRequest) -> SimulationResponse:
    """Execute simulation via HTTP API with strict error handling."""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(
            SIMULATION_ENDPOINT,
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return SimulationResponse.model_validate(response.json())


def render_energy_risk_view():
    """
    نمایش صفحه تخصصی شبیه‌سازی ریسک ناترازی انرژی به همراه راهنماهای جامع.
    Thin Client: All computation delegated to FastAPI server at port 8001.
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
        proposal = _build_proposal_request(
            power_p10=power_p10,
            power_p50=power_p50,
            power_p90=power_p90,
            gas_p10=gas_p10,
            gas_p50=gas_p50,
            gas_p90=gas_p90,
            base_benefit=base_benefit,
            daily_loss=daily_loss,
            years=years,
        )
        sim_request = SimulationRequest(proposal=proposal, iterations=10000)

        try:
            with st.spinner("در حال شبیه‌سازی اثرات ناترازی انرژی از طریق API..."):
                results = _run_simulation_via_api(sim_request)

            st.success("✅ محاسبات شبیه‌سازی ریسک انرژی تکمیل شد.")

            # نمایش کارت‌های شاخص
            m1, m2, m3 = st.columns(3)
            m1.metric("میانگین روزهای قطعی برق سالانه", f"{results.power_outage_days_yearly:.1f} روز")
            m2.metric("میانگین روزهای قطعی گاز سالانه", f"{results.gas_outage_days_yearly:.1f} روز")
            m3.metric("منفعت خالص تعدیل‌شده سالانه", f"{results.adjusted_net_benefit:,.0f} م.ت")

            # Additional metrics row
            m4, m5, m6 = st.columns(3)
            m4.metric("میانگین ROI", f"{results.mean_roi:.2%}")
            m5.metric("VaR 95%", f"{results.var_95:,.0f} م.ت")
            m6.metric("احتمال زیان", f"{results.probability_of_loss:.1f}%")

        except httpx.TimeoutException:
            st.error("❌ خطای زمان‌بر (Timeout): سرور در ۱۰ ثانیه پاسخ نداد. لطفاً مطمئن شوید سرور FastAPI در پورت ۸۰۰۱ در حال اجراست.")
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error_resp = ErrorResponse.model_validate(error_data)
                st.error(f"❌ خطای سرور ({e.response.status_code}): {error_resp.message}")
                if error_resp.details:
                    st.json(error_resp.details)
            except Exception:
                st.error(f"❌ خطای HTTP {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            st.error(f"❌ خطای اتصال: نمی‌توان به سرور در {API_BASE_URL} متصل شد. {e}")
        except Exception as e:
            st.error(f"❌ خطای غیرمنتظره: {e}")