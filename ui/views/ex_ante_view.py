import streamlit as st
import httpx
from core.contracts import (
    ProposalRequest,
    EvaluationResponse,
    ErrorResponse,
    Triplet,
    NonNegativeTriplet,
    CurrencyType,
    RiskStatus,
)


API_BASE_URL = "http://127.0.0.1:8001/api/v1"
EVALUATE_ENDPOINT = f"{API_BASE_URL}/proposal/evaluate"
REQUEST_TIMEOUT = 10.0


def reset_proposal_form():
    """
    بازنشاری مقادیر ورودی فرم برای ثبت پروپوزال جدید
    """
    st.session_state["proposal_form_version"] = st.session_state.get("proposal_form_version", 0) + 1
    st.session_state["last_evaluated_result"] = None
    st.session_state["last_proposal_input"] = None
    st.session_state["last_api_error"] = None


def _build_proposal_request(form_v: int) -> ProposalRequest:
    """Construct and validate ProposalRequest from form inputs."""
    cost_p10 = st.session_state.get(f"c10_{form_v}", 0.0)
    cost_p50 = st.session_state.get(f"c50_{form_v}", 0.0)
    cost_p90 = st.session_state.get(f"c90_{form_v}", 0.0)
    benefit_p10 = st.session_state.get(f"b10_{form_v}", 0.0)
    benefit_p50 = st.session_state.get(f"b50_{form_v}", 0.0)
    benefit_p90 = st.session_state.get(f"b90_{form_v}", 0.0)

    costs_sorted = sorted([float(cost_p10), float(cost_p50), float(cost_p90)])
    benefits_sorted = sorted([float(benefit_p10), float(benefit_p50), float(benefit_p90)])

    inf_p10 = st.session_state.get(f"inf10_{form_v}", 35.0)
    inf_p50 = st.session_state.get(f"inf50_{form_v}", 50.0)
    inf_p90 = st.session_state.get(f"inf90_{form_v}", 70.0)
    inf_sorted = sorted([float(inf_p10) / 100.0, float(inf_p50) / 100.0, float(inf_p90) / 100.0])

    fxg_p10 = st.session_state.get(f"fxg10_{form_v}", 30.0)
    fxg_p50 = st.session_state.get(f"fxg50_{form_v}", 45.0)
    fxg_p90 = st.session_state.get(f"fxg90_{form_v}", 65.0)
    fxg_sorted = sorted([float(fxg_p10) / 100.0, float(fxg_p50) / 100.0, float(fxg_p90) / 100.0])

    pwr_p10 = st.session_state.get(f"pwr10_{form_v}", 10.0)
    pwr_p50 = st.session_state.get(f"pwr50_{form_v}", 15.0)
    pwr_p90 = st.session_state.get(f"pwr90_{form_v}", 25.0)
    pwr_sorted = sorted([float(pwr_p10), float(pwr_p50), float(pwr_p90)])

    gas_p10 = st.session_state.get(f"gas10_{form_v}", 15.0)
    gas_p50 = st.session_state.get(f"gas50_{form_v}", 20.0)
    gas_p90 = st.session_state.get(f"gas90_{form_v}", 30.0)
    gas_sorted = sorted([float(gas_p10), float(gas_p50), float(gas_p90)])

    mtbf_p10 = st.session_state.get(f"mtbf10_{form_v}", 1200.0)
    mtbf_p50 = st.session_state.get(f"mtbf50_{form_v}", 900.0)
    mtbf_p90 = st.session_state.get(f"mtbf90_{form_v}", 500.0)
    mtbf_sorted = sorted([float(mtbf_p10), float(mtbf_p50), float(mtbf_p90)])

    mttr_p10 = st.session_state.get(f"mttr10_{form_v}", 2.0)
    mttr_p50 = st.session_state.get(f"mttr50_{form_v}", 4.0)
    mttr_p90 = st.session_state.get(f"mttr90_{form_v}", 8.0)
    mttr_sorted = sorted([float(mttr_p10), float(mttr_p50), float(mttr_p90)])

    lt_p10 = st.session_state.get(f"lt10_{form_v}", 75.0)
    lt_p50 = st.session_state.get(f"lt50_{form_v}", 110.0)
    lt_p90 = st.session_state.get(f"lt90_{form_v}", 160.0)
    lt_sorted = sorted([float(lt_p10), float(lt_p50), float(lt_p90)])

    currency_choice = st.session_state.get(f"currency_choice_{form_v}", "دلار آمریکا (USD)")
    currency_code = "USD" if "USD" in currency_choice else ("EUR" if "EUR" in currency_choice else "CNY")

    approved_cost_pct = st.session_state.get(f"apppct_{form_v}", 0.80)
    approved_pct_triplet = (
        approved_cost_pct * 0.8,
        approved_cost_pct,
        min(approved_cost_pct * 1.2, 1.0),
    )

    enable_cbam = st.session_state.get(f"cbam_chk_{form_v}", False)
    export_tons = st.session_state.get(f"exptons_{form_v}", 0.0) if enable_cbam else 0.0
    co2_reduction_kg = st.session_state.get(f"co2red_{form_v}", 0.0) if enable_cbam else 0.0
    export_tons_triplet = (
        export_tons * 0.7,
        export_tons,
        export_tons * 1.3,
    ) if enable_cbam else (0.0, 0.0, 0.0)

    ct_p10 = st.session_state.get(f"ct10_{form_v}", 60.0) if enable_cbam else 60.0
    ct_p50 = st.session_state.get(f"ct50_{form_v}", 85.0) if enable_cbam else 85.0
    ct_p90 = st.session_state.get(f"ct90_{form_v}", 120.0) if enable_cbam else 120.0
    ct_sorted = sorted([float(ct_p10), float(ct_p50), float(ct_p90)])

    trl_level = int(st.session_state.get(f"trl_level_{form_v}", 6))
    team_capability = st.session_state.get(f"team_capability_{form_v}", "MEDIUM")
    technical_complexity = st.session_state.get(f"technical_complexity_{form_v}", "MEDIUM")
    supply_dependence = st.session_state.get(f"supply_dependence_{form_v}", "MODERATE_DELAY")

    return ProposalRequest(
        title=st.session_state.get(f"title_{form_v}", "Untitled Project"),
        years=int(st.session_state.get(f"years_{form_v}", 3)),
        p_success=float(st.session_state.get(f"psucc_{form_v}", 0.5)),
        cost=Triplet(low=costs_sorted[0], likely=costs_sorted[1], high=costs_sorted[2]),
        benefit=Triplet(low=benefits_sorted[0], likely=benefits_sorted[1], high=benefits_sorted[2]),
        inflation=Triplet(low=inf_sorted[0], likely=inf_sorted[1], high=inf_sorted[2]),
        fx_growth=Triplet(low=fxg_sorted[0], likely=fxg_sorted[1], high=fxg_sorted[2]),
        currency_type=CurrencyType(currency_code),
        base_fx_rate=float(st.session_state.get(f"base_fx_{form_v}", 65000.0)),
        annual_fx_savings=float(st.session_state.get(f"fx_sav_{form_v}", 40000.0)),
        enable_energy_risk=st.session_state.get(f"en_chk_{form_v}", True),
        power_outage=Triplet(low=pwr_sorted[0], likely=pwr_sorted[1], high=pwr_sorted[2]),
        gas_outage=Triplet(low=gas_sorted[0], likely=gas_sorted[1], high=gas_sorted[2]),
        energy_daily_loss=float(st.session_state.get(f"enloss_{form_v}", 40.0)),
        enable_reliability_risk=st.session_state.get(f"rel_chk_{form_v}", True),
        mtbf=Triplet(low=mtbf_sorted[0], likely=mtbf_sorted[1], high=mtbf_sorted[2]),
        mttr=Triplet(low=mttr_sorted[0], likely=mttr_sorted[1], high=mttr_sorted[2]),
        hourly_downtime_loss=float(st.session_state.get(f"hr_loss_{form_v}", 20.0)),
        annual_operating_hours=float(st.session_state.get(f"ann_hrs_{form_v}", 7200.0)),
        enable_supply_chain_risk=st.session_state.get(f"sc_chk_{form_v}", True),
        planned_lead_time=float(st.session_state.get(f"plt_{form_v}", 90.0)),
        actual_lead_time=Triplet(low=lt_sorted[0], likely=lt_sorted[1], high=lt_sorted[2]),
        daily_delay_cost=float(st.session_state.get(f"ddc_{form_v}", 10.0)),
        enable_iran_tax=st.session_state.get(f"irtax_chk_{form_v}", True),
        corporate_tax_rate=float(st.session_state.get(f"corptax_{form_v}", 20.0)) / 100.0,
        approved_pct=Triplet(low=approved_pct_triplet[0], likely=approved_pct_triplet[1], high=approved_pct_triplet[2]),
        enable_cbam_tax=enable_cbam,
        export_tons=NonNegativeTriplet(low=export_tons_triplet[0], likely=export_tons_triplet[1], high=export_tons_triplet[2]),
        co2_reduction_kg=float(co2_reduction_kg),
        carbon_tax_usd=Triplet(low=ct_sorted[0], likely=ct_sorted[1], high=ct_sorted[2]),
        trl_level=trl_level,
        team_capability=team_capability,
        technical_complexity=technical_complexity,
        supply_dependence=supply_dependence,
    )


def _call_evaluation_api(request: ProposalRequest) -> tuple[EvaluationResponse | None, ErrorResponse | None]:
    """Call the evaluation API endpoint with timeout and structured error handling."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                EVALUATE_ENDPOINT,
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            data = response.json()
            return EvaluationResponse.model_validate(data), None
    except httpx.TimeoutException:
        return None, ErrorResponse(
            error_code="TIMEOUT",
            message=f"Request timed out after {REQUEST_TIMEOUT}s",
            details={"endpoint": EVALUATE_ENDPOINT},
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
            message="Failed to connect to evaluation service",
            details={"error": str(e), "endpoint": EVALUATE_ENDPOINT},
        )
    except Exception as e:
        return None, ErrorResponse(
            error_code="UNEXPECTED_ERROR",
            message="Unexpected error during API call",
            details={"error": str(e)},
        )


def render_ex_ante_view():
    """
    نمایش صفحه ثبت و ارزیابی جامع پروپوزال با فرم چندتبی یکپارچه.
    تمام ارزیابی‌ها از طریق HTTP به Mock Server ارسال می‌شوند (Thin Client).
    """
    if "proposal_form_version" not in st.session_state:
        st.session_state["proposal_form_version"] = 0
    if "last_evaluated_result" not in st.session_state:
        st.session_state["last_evaluated_result"] = None
    if "last_proposal_input" not in st.session_state:
        st.session_state["last_proposal_input"] = None
    if "last_api_error" not in st.session_state:
        st.session_state["last_api_error"] = None

    form_v = st.session_state["proposal_form_version"]

    st.header("📋 ۱. ثبت و ارزیابی جامع پروژه (Project Intake)")

    with st.expander("📖 **راهنمای جامع ارزیابی یکپارچه (کلیک کنید)**", expanded=False):
        st.markdown("""
        این بخش نقطه شروع ارزیابی پروژه‌های تحقیق و توسعه است.
        * تمام ابعاد مالی، ریسک‌های صنعتی خط تولید و مشوق‌های قانونی را در تب‌های زیر وارد کنید.
        * سیستم تمام محاسبات را از طریق API Server انجام می‌دهد.
        * پس از ثبت، پروژه وارد **بانک اطلاعاتی و جدول چیدمان سبد پروژه‌ها** خواهد شد.
        """)

    st.subheader("📁 بارگذاری سند پروپوزال (اختیاری)")
    uploaded_file = st.file_uploader(
        "فایل PDF پروپوزال را انتخاب کنید",
        type=["pdf"],
        key=f"uploader_pdf_{form_v}"
    )

    parsed_defaults = {
        "cost_p10": 0.0, "cost_p50": 0.0, "cost_p90": 0.0,
        "benefit_p10": 0.0, "benefit_p50": 0.0, "benefit_p90": 0.0,
        "p_success": 0.5
    }

    if uploaded_file is not None:
        st.info("📄 فایل آپلود شد. استخراج متن در سمت سرور انجام می‌شود. لطفاً مقادیر مالی را در تب‌ها تأیید کنید.")

    st.markdown("---")
    title = st.text_input(
        "عنوان پروژه تحقیق و توسعه:",
        value="سامانه هوشمند عیب‌یابی و کنترل خط نورد",
        key=f"title_{form_v}"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 ۱. مالی و تورم",
        "⚡ ۲. ریسک انرژی",
        "⚙️ ۳. توقفات و MTBF",
        "🚚 ۴. لجستیک و ارز",
        "⚖️ ۵. مشوق‌های مالیاتی"
    ])

    with tab1:
        st.markdown("#### پارامترهای پایه مالی و نرخ تورم ریالی")
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("**تخمین هزینه اولیه پروژه (میلیون تومان):**")
            st.number_input("هزینه خوش‌بینانه (P10)", value=float(parsed_defaults["cost_p10"]), step=500.0, key=f"c10_{form_v}")
            st.number_input("هزینه محتمل (P50)", value=float(parsed_defaults["cost_p50"]), step=500.0, key=f"c50_{form_v}")
            st.number_input("هزینه بدبینانه (P90)", value=float(parsed_defaults["cost_p90"]), step=500.0, key=f"c90_{form_v}")

        with c_col2:
            st.markdown("**تخمین منافع ریالی سالانه پایه (میلیون تومان):**")
            st.number_input("منافع خوش‌بینانه (P10)", value=float(parsed_defaults["benefit_p10"]), step=500.0, key=f"b10_{form_v}")
            st.number_input("منافع محتمل (P50)", value=float(parsed_defaults["benefit_p50"]), step=500.0, key=f"b50_{form_v}")
            st.number_input("منافع بدبینانه (P90)", value=float(parsed_defaults["benefit_p90"]), step=500.0, key=f"b90_{form_v}")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.slider("احتمال موفقیت فنی پروژه:", 0.0, 1.0, float(parsed_defaults["p_success"]), step=0.05, key=f"psucc_{form_v}")
        with p_col2:
            st.slider("افق زمانی بهره‌برداری (سال):", 1, 10, 3, key=f"years_{form_v}")

        st.markdown("#### معیارهای کیفی احتمال موفقیت فنی")
        qual_col1, qual_col2 = st.columns(2)
        with qual_col1:
            st.selectbox(
                "سطح آمادگی فناوری (TRL)",
                options=list(range(1, 10)),
                index=5,
                key=f"trl_level_{form_v}"
            )
            team_capability_options = ["متوسط", "بالا", "پایین"]
            team_capability_map = {"متوسط": "MEDIUM", "بالا": "HIGH", "پایین": "LOW"}
            team_capability_display = st.selectbox(
                "توانمندی و سابقه تیم اجرایی",
                options=team_capability_options,
                index=0,
                key=f"team_capability_display_{form_v}"
            )
            st.session_state[f"team_capability_{form_v}"] = team_capability_map[team_capability_display]
        with qual_col2:
            technical_complexity_options = ["متوسط", "بالا", "پایین"]
            technical_complexity_map = {"متوسط": "MEDIUM", "بالا": "HIGH", "پایین": "LOW"}
            technical_complexity_display = st.selectbox(
                "پیچیدگی فنی پروژه",
                options=technical_complexity_options,
                index=0,
                key=f"technical_complexity_display_{form_v}"
            )
            st.session_state[f"technical_complexity_{form_v}"] = technical_complexity_map[technical_complexity_display]
            supply_dependence_options = [
                "تأخیر متعارف (واردات عادی)",
                "تأمین کاملاً داخلی",
                "واردات حیاتی و دارای ریسک تحریم"
            ]
            supply_dependence_map = {
                "تأمین کاملاً داخلی": "DOMESTIC",
                "تأخیر متعارف (واردات عادی)": "MODERATE_DELAY",
                "واردات حیاتی و دارای ریسک تحریم": "CRITICAL_IMPORT"
            }
            supply_dependence_display = st.selectbox(
                "وابستگی زنجیره تأمین و تجهیزات",
                options=supply_dependence_options,
                index=0,
                key=f"supply_dependence_display_{form_v}"
            )
            st.session_state[f"supply_dependence_{form_v}"] = supply_dependence_map[supply_dependence_display]

        st.markdown("**سه‌گانه درصد تورم سالانه ریالی:**")
        inf_col1, inf_col2, inf_col3 = st.columns(3)
        with inf_col1:
            st.number_input("حداقل تورم (P10 - درصد)", value=35.0, step=5.0, key=f"inf10_{form_v}")
        with inf_col2:
            st.number_input("محتمل‌ترین تورم (P50 - درصد)", value=50.0, step=5.0, key=f"inf50_{form_v}")
        with inf_col3:
            st.number_input("حداکثر تورم (P90 - درصد)", value=70.0, step=5.0, key=f"inf90_{form_v}")

    with tab2:
        st.markdown("#### شبیه‌سازی اثرات قطعی برق و گاز بر عملکرد پروژه")
        st.checkbox("فعال‌سازی ارزیابی ریسک انرژی برای این پروژه", value=True, key=f"en_chk_{form_v}")
        en_col1, en_col2 = st.columns(2)
        with en_col1:
            st.number_input("خوش‌بینانه برق (P10 - روز)", value=10.0, step=1.0, key=f"pwr10_{form_v}")
            st.number_input("محتمل برق (P50 - روز)", value=15.0, step=1.0, key=f"pwr50_{form_v}")
            st.number_input("بدبینانه برق (P90 - روز)", value=25.0, step=1.0, key=f"pwr90_{form_v}")
        with en_col2:
            st.number_input("خوش‌بینانه گاز (P10 - روز)", value=15.0, step=1.0, key=f"gas10_{form_v}")
            st.number_input("محتمل گاز (P50 - روز)", value=20.0, step=1.0, key=f"gas50_{form_v}")
            st.number_input("بدبینانه گاز (P90 - روز)", value=30.0, step=1.0, key=f"gas90_{form_v}")
        st.number_input("زیان روزانه توقف فرآیند ناشی از قطعی انرژی (میلیون تومان):", value=40.0, step=5.0, key=f"enloss_{form_v}")

    with tab3:
        st.markdown("#### ارزیابی نقص فنی و توقفات خط (MTBF / MTTR)")
        st.checkbox("فعال‌سازی ارزیابی MTBF و توقفات خط", value=True, key=f"rel_chk_{form_v}")
        rel_col1, rel_col2 = st.columns(2)
        with rel_col1:
            st.markdown("#### زمان رفع عیب و توقف (MTTR)")
            st.number_input("مدت توقف خوش‌بینانه (P10 - ساعت)", value=2.0, step=0.5, key=f"mttr10_{form_v}")
            st.number_input("محتمل (P50)", value=4.0, step=0.5, key=f"mttr50_{form_v}")
            st.number_input("بدبینانه (P90)", value=8.0, step=0.5, key=f"mttr90_{form_v}")
        with rel_col2:
            st.markdown("#### شاخص‌های فاصله بین خرابی‌ها (MTBF)")
            st.number_input("فاصله خرابی خوش‌بینانه (P10 - ساعت)", value=1200.0, step=100.0, key=f"mtbf10_{form_v}")
            st.number_input("محتمل (P50)", value=900.0, step=100.0, key=f"mtbf50_{form_v}")
            st.number_input("بدبینانه (P90)", value=500.0, step=100.0, key=f"mtbf90_{form_v}")
        op_col1, op_col2 = st.columns(2)
        with op_col1:
            st.number_input("کل ساعات کاری سالانه خط (ساعت):", value=7200.0, step=200.0, key=f"ann_hrs_{form_v}")
        with op_col2:
            st.number_input("خسارت هر ساعت توقف خط (میلیون تومان):", value=20.0, step=2.0, key=f"hr_loss_{form_v}")

    with tab4:
        currency_choice = st.session_state.get(f"currency_choice_{form_v}", "دلار آمریکا (USD)")
        currency_code = "USD" if "USD" in currency_choice else ("EUR" if "EUR" in currency_choice else "CNY")
        st.markdown("#### تأخیرات زنجیره تأمین و صرفه‌جویی ارزی بومی‌سازی")
        cur_col1, cur_col2 = st.columns(2)
        with cur_col1:
            st.selectbox("ارز مرجع صرفه‌جویی:", ["دلار آمریکا (USD)", "یورو (EUR)", "یوان چین (CNY)"], key=f"currency_choice_{form_v}")
            st.number_input(f"نرخ فعلی {currency_choice} به تومان:", value=65000.0, step=1000.0, key=f"base_fx_{form_v}")
        with cur_col2:
            st.number_input(f"صرفه‌جویی ارزی سالانه ({currency_code}):", value=40000.0, step=5000.0, key=f"fx_sav_{form_v}")

        st.markdown("**سه‌گانه نرخ رشد ارز سالانه (درصد):**")
        fxg_col1, fxg_col2, fxg_col3 = st.columns(3)
        with fxg_col1:
            st.number_input("حداقل رشد ارز (P10 - درصد)", value=30.0, step=5.0, key=f"fxg10_{form_v}")
        with fxg_col2:
            st.number_input("محتمل‌ترین رشد ارز (P50 - درصد)", value=45.0, step=5.0, key=f"fxg50_{form_v}")
        with fxg_col3:
            st.number_input("حداکثر رشد ارز (P90 - درصد)", value=65.0, step=5.0, key=f"fxg90_{form_v}")

        st.checkbox("فعال‌سازی ارزیابی تأخیر در ترخیص و تحویل تجهیزات", value=True, key=f"sc_chk_{form_v}")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            st.number_input("زمان مصوب تحویل در قرارداد (روز):", value=90.0, step=5.0, key=f"plt_{form_v}")
        with sc_col2:
            st.number_input("خسارت روزانه دیرکرد تحویل (میلیون تومان):", value=10.0, step=1.0, key=f"ddc_{form_v}")

        lt_col1, lt_col2, lt_col3 = st.columns(3)
        with lt_col1:
            st.number_input("سریع‌ترین تحویل (P10)", value=75.0, step=5.0, key=f"lt10_{form_v}")
        with lt_col2:
            st.number_input("تحویل عادی (P50)", value=110.0, step=5.0, key=f"lt50_{form_v}")
        with lt_col3:
            st.number_input("تحویل با تأخیر حاد (P90)", value=160.0, step=5.0, key=f"lt90_{form_v}")

    with tab5:
        st.markdown("#### اعتبار مالیاتی در ایران و معافیت کربن اروپا")
        st.checkbox("اعتبار مالیاتی تحقیق و توسعه طبق قانون جهش تولید در ایران", value=True, key=f"irtax_chk_{form_v}")
        tax_col1, tax_col2 = st.columns(2)
        with tax_col1:
            st.number_input("نرخ مالیات عملکرد شرکت (درصد):", value=20.0, step=1.0, key=f"corptax_{form_v}")
        with tax_col2:
            st.slider("درصد پیش‌بینی تایید هزینه‌های R&D توسط کارگروه:", 0.0, 1.0, 0.80, step=0.05, key=f"apppct_{form_v}")

        st.checkbox("معافیت از مالیات کربن صادراتی به اروپا (CBAM)", value=False, key=f"cbam_chk_{form_v}")
        if st.session_state.get(f"cbam_chk_{form_v}", False):
            cbam_col1, cbam_col2 = st.columns(2)
            with cbam_col1:
                st.number_input("تناژ صادرات سالانه مشمول عوارض (تن):", value=100000.0, step=10000.0, key=f"exptons_{form_v}")
            with cbam_col2:
                st.number_input("کاهش CO2 به‌ازای هر تن تولید (کیلوگرم):", value=120.0, step=10.0, key=f"co2red_{form_v}")

            st.markdown("**سه‌گانه مالیات کربن به ازای هر تن (دلار آمریکا):**")
            ct_col1, ct_col2, ct_col3 = st.columns(3)
            with ct_col1:
                st.number_input("حداقل مالیات کربن (P10 - دلار)", value=60.0, step=5.0, key=f"ct10_{form_v}")
            with ct_col2:
                st.number_input("محتمل‌ترین مالیات کربن (P50 - دلار)", value=85.0, step=5.0, key=f"ct50_{form_v}")
            with ct_col3:
                st.number_input("حداکثر مالیات کربن (P90 - دلار)", value=120.0, step=5.0, key=f"ct90_{form_v}")

    st.markdown("---")

    btn_col1, btn_col2 = st.columns([3, 2])
    with btn_col1:
        eval_btn = st.button("🚀 اجرای شبیه‌سازی جامع و ذخیره مستقیم در بانک پروژه‌ها", type="primary")
    with btn_col2:
        st.button("➕ ثبت اطلاعات از پروپوزال جدید", on_click=reset_proposal_form)

    if eval_btn:
        if min(
            float(st.session_state.get(f"c10_{form_v}", 0)),
            float(st.session_state.get(f"c50_{form_v}", 0)),
            float(st.session_state.get(f"c90_{form_v}", 0)),
            float(st.session_state.get(f"b10_{form_v}", 0)),
            float(st.session_state.get(f"b50_{form_v}", 0)),
            float(st.session_state.get(f"b90_{form_v}", 0)),
        ) <= 0:
            st.error(
                "⛔ مقادیر مالی باید مثبت و تأییدشده باشند. در صورت استخراج ناقص از سند، "
                "لطفاً سه‌گانه‌های هزینه و منافع را دستی وارد کنید."
            )
            st.stop()

        with st.spinner("در حال ارسال درخواست به سرور ارزیابی..."):
            request = _build_proposal_request(form_v)
            result, error = _call_evaluation_api(request)

        if error:
            st.session_state["last_api_error"] = error
            st.error(f"❌ خطای سرور: {error.message}")
            if error.details:
                st.json(error.details)
            st.stop()

        if result:
            st.session_state["last_evaluated_result"] = result
            st.session_state["last_proposal_input"] = request
            st.session_state["last_api_error"] = None
            st.success(f"✅ ارزیابی کامل شد: {result.title}")
            st.rerun()

    last_res = st.session_state.get("last_evaluated_result")
    last_inp = st.session_state.get("last_proposal_input")
    last_error = st.session_state.get("last_api_error")

    if last_error and not last_res:
        st.error(f"❌ آخرین تلاش ناموفق: {last_error.message}")
        if last_error.details:
            with st.expander("جزئیات خطا"):
                st.json(last_error.details)

    if last_res and last_inp:
        st.subheader("🎯 کارنامه نتایج تجمیعی و تعدیل‌شده پروژه ثبت‌شده")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("منفعت خالص تعدیل‌شده سالانه", f"{last_res.adjusted_net_benefit:,.0f} م.ت")
        m2.metric("مجموع ساعات توقف خط در سال", f"{last_res.total_downtime_hours} ساعت")
        m3.metric("میانگین بازگشت سرمایه (ROI)", f"{last_res.mean_roi:.1%}")
        m4.metric("شاخص ریسک VaR 95%", f"{last_res.var_95:,.0f}")

        llm_info = last_res.qualitative_assessment
        if llm_info:
            st.info(f"🤖 **توصیه استراتژیک سیستم:** {llm_info.risk_summary}")

        st.markdown("---")
        st.subheader("📊 معیارهای کمی تفصیلی")
        qm = last_res.quantitative_metrics
        q_col1, q_col2, q_col3 = st.columns(3)
        q_col1.metric("NPV (P10)", f"{qm.npv_p10:,.0f}")
        q_col1.metric("NPV (P50)", f"{qm.npv_p50:,.0f}")
        q_col1.metric("NPV (P90)", f"{qm.npv_p90:,.0f}")
        q_col2.metric("ROI (P10)", f"{qm.roi_p10:.1%}")
        q_col2.metric("ROI (P50)", f"{qm.roi_p50:.1%}")
        q_col2.metric("ROI (P90)", f"{qm.roi_p90:.1%}")
        q_col3.metric("دوره بازگشت سرمایه", f"{qm.payback_period_years:.1f} سال")
        q_col3.metric("نسبت منفعت به هزینه", f"{qm.benefit_cost_ratio:.2f}")

        st.markdown("---")
        st.subheader("📄 جزئیات محاسبات")
        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**هزینه کل (P10/P50/P90):** {last_res.cost.low:,.0f} / {last_res.cost.likely:,.0f} / {last_res.cost.high:,.0f} م.ت")
            st.write(f"**منافع کل (P10/P50/P90):** {last_res.benefit.low:,.0f} / {last_res.benefit.likely:,.0f} / {last_res.benefit.high:,.0f} م.ت")
            st.write(f"**احتمال موفقیت فنی:** {last_res.p_success:.0%}")
            st.write(f"**زیان انرژی:** {last_res.energy_loss_toman:,.0f} م.ت")
            st.write(f"**زیان توقفات:** {last_res.downtime_loss_toman:,.0f} م.ت")
        with detail_col2:
            st.write(f"**تأخیر تحویل:** {last_res.lead_time_delay_days:.1f} روز")
            st.write(f"**اعتبار مالیاتی ایران:** {last_res.iran_tax_credit_toman:,.0f} م.ت")
            st.write(f"**صرفه‌جویی کربن:** {last_res.carbon_savings_toman:,.0f} م.ت")
            st.write(f"**احتمال ضرر:** {last_res.probability_of_loss:.1f}%")
            st.write(f"**وضعیت ریسک:** {last_res.risk_status.value}")

        st.markdown("---")
        st.subheader("📥 دانلود گزارش Markdown")
        import json
        report_data = {
            "input": last_inp.model_dump(mode="json"),
            "output": last_res.model_dump(mode="json"),
        }
        st.download_button(
            label="📥 دانلود گزارش کامل (JSON)",
            data=json.dumps(report_data, ensure_ascii=False, indent=2),
            file_name=f"Report_{last_res.title}.json",
            mime="application/json",
        )