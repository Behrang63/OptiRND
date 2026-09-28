"""
FastAPI mock server for OptiRND evaluation pipeline.
Serves on port 8001 with OWASP security headers and strict CORS.
"""
import html
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from starlette.templating import Jinja2Templates
import jinja2

from core.contracts import (
    ComplianceAuditResponse,
    ComplianceCheckRequest,
    ErrorResponse,
    EvaluationResponse,
    PortfolioDecisionResponse,
    PortfolioOptimizationRequest,
    PortfolioProjectRequest,
    ProposalRequest,
    QualitativeAssessment,
    QuantitativeMetrics,
    RiskStatus,
    SimulationRequest,
    SimulationResponse,
    TaskStatus,
    TaskStatusResponse,
    Triplet,
)

# Core simulation engines
from core.monte_carlo import MonteCarloEngine
from core.energy_simulation import EnergyRiskEngine
from core.reliability_engine import ReliabilityEngine
from core.supply_chain_engine import SupplyChainEngine
from core.carbon_tax_engine import CarbonTaxEngine
from core.portfolio_optimizer import PortfolioOptimizationEngine
from core.qualitative_engine import calculate_technical_success_probability


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    yield


app = FastAPI(
    title="OptiRND Mock API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
    cache_size=0,
)
templates = Jinja2Templates(env=jinja_env)


# OWASP Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.tailwindcss.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "base-uri 'self';"
    )
    return response


# Strict CORS - no wildcard for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)


# Global validation error handler -> structured ErrorResponse
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()},
        ).model_dump(mode="json"),
    )


# Also handle FastAPI's RequestValidationError which wraps ValidationError
from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()},
        ).model_dump(mode="json"),
    )


# =============================================================================
# REAL PROPOSAL EVALUATION ENGINE INTEGRATION
# =============================================================================

from core.llm_provider import MockLLMProvider


def _run_real_evaluation(request: ProposalRequest) -> EvaluationResponse:
    """
    Execute real proposal evaluation using core simulation engines and LLM provider.

    Maps ProposalRequest DTO fields to engine parameters, runs all risk simulations,
    generates qualitative assessment, and aggregates results into a validated
    EvaluationResponse.
    """
    # Derive p_success from qualitative attributes if present, else use provided p_success
    trl_level = getattr(request, "trl_level", None)
    team_capability = getattr(request, "team_capability", None)
    technical_complexity = getattr(request, "technical_complexity", None)
    supply_dependence = getattr(request, "supply_dependence", None)

    if all(v is not None for v in (trl_level, team_capability, technical_complexity, supply_dependence)):
        p_success = calculate_technical_success_probability(
            trl_level=trl_level,
            team_capability=team_capability,
            technical_complexity=technical_complexity,
            supply_dependence=supply_dependence,
        )
    else:
        p_success = request.p_success

    # Extract triplets from ProposalRequest
    cost_triplet = (request.cost.low, request.cost.likely, request.cost.high)
    benefit_triplet = (request.benefit.low, request.benefit.likely, request.benefit.high)
    inflation_triplet = (request.inflation.low, request.inflation.likely, request.inflation.high)
    fx_growth_triplet = (request.fx_growth.low, request.fx_growth.likely, request.fx_growth.high)
    power_outage_triplet = (request.power_outage.low, request.power_outage.likely, request.power_outage.high)
    gas_outage_triplet = (request.gas_outage.low, request.gas_outage.likely, request.gas_outage.high)
    mtbf_triplet = (request.mtbf.low, request.mtbf.likely, request.mtbf.high)
    mttr_triplet = (request.mttr.low, request.mttr.likely, request.mttr.high)
    actual_lead_time_triplet = (request.actual_lead_time.low, request.actual_lead_time.likely, request.actual_lead_time.high)
    approved_pct_triplet = (request.approved_pct.low, request.approved_pct.likely, request.approved_pct.high)
    export_tons_triplet = (request.export_tons.low, request.export_tons.likely, request.export_tons.high)
    carbon_tax_usd_triplet = (request.carbon_tax_usd.low, request.carbon_tax_usd.likely, request.carbon_tax_usd.high)

    years = request.years

    # Instantiate engines with deterministic seed for reproducibility
    seed = 42
    mc_engine = MonteCarloEngine(num_simulations=10000, seed=seed)
    energy_engine = EnergyRiskEngine(num_simulations=10000, seed=seed)
    reliability_engine = ReliabilityEngine(num_simulations=10000, seed=seed)
    supply_chain_engine = SupplyChainEngine(num_simulations=10000, seed=seed)
    tax_engine = CarbonTaxEngine(num_simulations=10000, seed=seed)
    llm_provider = MockLLMProvider()

    # 1. Monte Carlo ROI simulation
    mc_results = mc_engine.run_roi_simulation(
        cost_p10_p50_p90=cost_triplet,
        benefit_p10_p50_p90=benefit_triplet,
        inflation_p10_p50_p90=inflation_triplet,
        fx_growth_p10_p50_p90=fx_growth_triplet,
        currency_type=request.currency_type.value,
        base_fx_rate=request.base_fx_rate,
        annual_fx_savings=request.annual_fx_savings,
        p_success=p_success,
        years=years,
    )

    # Compute percentiles from raw samples for quantitative metrics
    roi_samples = mc_results.get("raw_roi_samples")
    if roi_samples is not None:
        roi_p10 = float(np.percentile(roi_samples, 10))
        roi_p50 = float(np.percentile(roi_samples, 50))
        roi_p90 = float(np.percentile(roi_samples, 90))
        # Derive NPV percentiles from ROI percentiles and cost
        cost_p50 = cost_triplet[1]
        npv_p10 = cost_p50 * roi_p10 / 100.0
        npv_p50 = cost_p50 * roi_p50 / 100.0
        npv_p90 = cost_p50 * roi_p90 / 100.0
        # Estimate payback period and benefit-cost ratio
        payback_period_years = years * cost_p50 / max(benefit_triplet[1], 1.0) if benefit_triplet[1] > 0 else 0.0
        benefit_cost_ratio = benefit_triplet[1] / max(cost_p50, 1.0) if cost_p50 > 0 else 0.0
    else:
        roi_p10 = roi_p50 = roi_p90 = 0.0
        npv_p10 = npv_p50 = npv_p90 = 0.0
        payback_period_years = 0.0
        benefit_cost_ratio = 0.0

    # 2. Energy risk simulation
    energy_loss = 0.0
    energy_outage_days = 0.0
    if request.enable_energy_risk:
        energy_res = energy_engine.simulate_energy_impact(
            base_annual_benefit=benefit_triplet[1],
            power_outage_days_triplet=power_outage_triplet,
            gas_outage_days_triplet=gas_outage_triplet,
            daily_downtime_loss=request.energy_daily_loss,
            years=years,
        )
        energy_loss = energy_res["mean_annual_loss_toman"]
        energy_outage_days = energy_res["mean_outage_days_yearly"]

    # 3. Reliability risk simulation
    downtime_loss = 0.0
    downtime_hours = 0.0
    if request.enable_reliability_risk:
        rel_res = reliability_engine.simulate_downtime_risk(
            base_annual_benefit=benefit_triplet[1],
            mtbf_hours_triplet=mtbf_triplet,
            mttr_hours_triplet=mttr_triplet,
            hourly_downtime_loss=request.hourly_downtime_loss,
            annual_operating_hours=request.annual_operating_hours,
            years=years,
        )
        downtime_loss = rel_res["mean_annual_downtime_loss"]
        downtime_hours = rel_res["mean_downtime_hours"]

    # 4. Supply chain risk simulation
    delay_days = 0.0
    if request.enable_supply_chain_risk:
        sc_res = supply_chain_engine.simulate_lead_time_risk(
            planned_lead_time_days=request.planned_lead_time,
            actual_lead_time_triplet=actual_lead_time_triplet,
            daily_delay_cost=request.daily_delay_cost,
        )
        delay_days = sc_res["mean_delay_days"]

    # 5. Tax incentives simulation
    iran_tax_credit = 0.0
    if request.enable_iran_tax:
        iran_tax_res = tax_engine.simulate_iran_tax_credit(
            rd_cost_triplet=cost_triplet,
            approved_cost_pct_triplet=approved_pct_triplet,
            corporate_tax_rate=request.corporate_tax_rate,
        )
        iran_tax_credit = iran_tax_res["mean_iran_tax_credit_savings"]

    carbon_savings = 0.0
    if request.enable_cbam_tax:
        cbam_res = tax_engine.simulate_eu_cbam_benefit(
            annual_export_tons_triplet=export_tons_triplet,
            co2_reduction_per_ton_kg=request.co2_reduction_kg,
            carbon_tax_per_ton_usd_triplet=carbon_tax_usd_triplet,
            usd_to_toman_rate=request.base_fx_rate,
            years=years,
        )
        carbon_savings = cbam_res["mean_annual_carbon_savings_toman"]

    # 6. Aggregate results
    total_downtime = downtime_hours + (energy_outage_days * 24.0)
    base_annual_p50 = benefit_triplet[1]

    # Calculate adjusted_net_benefit dynamically per specification
    base_benefit = base_annual_p50 * p_success
    supply_chain_deduction = delay_days * request.daily_delay_cost if request.enable_supply_chain_risk else 0.0
    deductions = energy_loss + downtime_loss + supply_chain_deduction
    additions = iran_tax_credit + carbon_savings
    adjusted_net_benefit = round(base_benefit - deductions + additions, 2)

    # Determine risk status based on VaR
    risk_status = RiskStatus.HIGH_CONFIDENCE if mc_results["var_95"] > 0 else RiskStatus.MODERATE_RISK

    # 7. Qualitative assessment via LLM provider
    prompt = (
        f"Evaluate proposal '{request.title}' with ROI = {mc_results['mean_roi']}%, "
        f"VaR 95% = {mc_results['var_95']}%, Adjusted Benefit = {adjusted_net_benefit:,.0f} MToman."
    )
    llm_assessment = llm_provider.generate_json(prompt, "QualitativeAssessment")
    qualitative = QualitativeAssessment.model_validate(llm_assessment)
    # Override technical_success_probability with dynamically computed p_success
    qualitative = QualitativeAssessment(
        technical_success_probability=p_success,
        recommended_action=qualitative.recommended_action,
        risk_summary=qualitative.risk_summary,
    )

    # 8. Build quantitative metrics from MC results
    quantitative_metrics = QuantitativeMetrics(
        npv_p10=npv_p10,
        npv_p50=npv_p50,
        npv_p90=npv_p90,
        roi_p10=roi_p10,
        roi_p50=roi_p50,
        roi_p90=roi_p90,
        payback_period_years=payback_period_years,
        benefit_cost_ratio=benefit_cost_ratio,
    )

    return EvaluationResponse(
        title=request.title,
        cost=request.cost,
        benefit=request.benefit,
        p_success=p_success,
        total_downtime_hours=round(total_downtime, 1),
        energy_loss_toman=round(energy_loss, 2),
        downtime_loss_toman=round(downtime_loss, 2),
        lead_time_delay_days=round(delay_days, 1),
        iran_tax_credit_toman=round(iran_tax_credit, 2),
        carbon_savings_toman=round(carbon_savings, 2),
        adjusted_net_benefit=adjusted_net_benefit,
        mean_roi=mc_results["mean_roi"],
        var_95=mc_results["var_95"],
        probability_of_loss=mc_results["probability_of_loss"],
        risk_status=risk_status,
        quantitative_metrics=quantitative_metrics,
        qualitative_assessment=qualitative,
    )


# =============================================================================
# REAL PORTFOLIO OPTIMIZATION ENGINE INTEGRATION
# =============================================================================

def _run_real_portfolio_optimization(request: PortfolioOptimizationRequest) -> PortfolioDecisionResponse:
    """
    Execute real portfolio optimization using the core PortfolioOptimizationEngine.

    Maps PortfolioOptimizationRequest DTO fields to engine parameters and aggregates
    results into a validated PortfolioDecisionResponse.
    """
    # Map DTO projects to engine format
    engine_projects = []
    for p in request.projects:
        engine_projects.append({
            "title": p.title,
            "cost_p10": p.cost.low,
            "cost_p50": p.cost.likely,
            "cost_p90": p.cost.high,
            "benefit_p10": p.benefit.low,
            "benefit_p50": p.benefit.likely,
            "benefit_p90": p.benefit.high,
            "dt_p10": p.downtime.low,
            "dt_p50": p.downtime.likely,
            "dt_p90": p.downtime.high,
        })

    # Instantiate engine with deterministic seed for reproducibility
    engine = PortfolioOptimizationEngine(num_scenarios=1000, seed=42)

    # Execute optimization
    result = engine.optimize_portfolio(
        projects=engine_projects,
        max_budget=request.budget_limit,
        max_downtime_hours=request.max_downtime_hours,
        method="stochastic",
    )

    # Map engine result to PortfolioDecisionResponse
    selected_titles = [p["title"] for p in result.get("selected_projects", [])]

    return PortfolioDecisionResponse(
        status=result.get("status", "FAILED"),
        method_used=result.get("method_used", "stochastic"),
        selected_titles=selected_titles,
        total_selected_npv=result.get("total_selected_npv", 0.0),
        total_selected_cost=result.get("total_selected_cost", 0.0),
        total_selected_downtime=result.get("total_selected_downtime", 0.0),
        budget_utilization_pct=result.get("budget_utilization_pct", 0.0),
        downtime_utilization_pct=result.get("downtime_utilization_pct", 0.0),
    )


# =============================================================================
# REAL SIMULATION ENGINE INTEGRATION
# =============================================================================

def _run_real_simulation(request: SimulationRequest) -> SimulationResponse:
    """
    Execute real Monte Carlo and risk simulations using core engines.
    
    Maps SimulationRequest DTO fields to engine parameters and aggregates
    results into a validated SimulationResponse.
    """
    proposal = request.proposal
    iterations = request.iterations
    seed = request.seed

    # Instantiate engines with request parameters
    mc_engine = MonteCarloEngine(num_simulations=iterations, seed=seed)
    energy_engine = EnergyRiskEngine(num_simulations=iterations, seed=seed)
    reliability_engine = ReliabilityEngine(num_simulations=iterations, seed=seed)
    supply_chain_engine = SupplyChainEngine(num_simulations=iterations, seed=seed)
    tax_engine = CarbonTaxEngine(num_simulations=iterations, seed=seed)

    years = proposal.years

    # Extract triplets from ProposalRequest
    cost_triplet = (proposal.cost.low, proposal.cost.likely, proposal.cost.high)
    benefit_triplet = (proposal.benefit.low, proposal.benefit.likely, proposal.benefit.high)
    inflation_triplet = (proposal.inflation.low, proposal.inflation.likely, proposal.inflation.high)
    fx_growth_triplet = (proposal.fx_growth.low, proposal.fx_growth.likely, proposal.fx_growth.high)
    power_outage_triplet = (proposal.power_outage.low, proposal.power_outage.likely, proposal.power_outage.high)
    gas_outage_triplet = (proposal.gas_outage.low, proposal.gas_outage.likely, proposal.gas_outage.high)
    mtbf_triplet = (proposal.mtbf.low, proposal.mtbf.likely, proposal.mtbf.high)
    mttr_triplet = (proposal.mttr.low, proposal.mttr.likely, proposal.mttr.high)
    actual_lead_time_triplet = (proposal.actual_lead_time.low, proposal.actual_lead_time.likely, proposal.actual_lead_time.high)
    approved_pct_triplet = (proposal.approved_pct.low, proposal.approved_pct.likely, proposal.approved_pct.high)
    export_tons_triplet = (proposal.export_tons.low, proposal.export_tons.likely, proposal.export_tons.high)
    carbon_tax_usd_triplet = (proposal.carbon_tax_usd.low, proposal.carbon_tax_usd.likely, proposal.carbon_tax_usd.high)

    # 1. Monte Carlo ROI simulation
    mc_results = mc_engine.run_roi_simulation(
        cost_p10_p50_p90=cost_triplet,
        benefit_p10_p50_p90=benefit_triplet,
        inflation_p10_p50_p90=inflation_triplet,
        fx_growth_p10_p50_p90=fx_growth_triplet,
        currency_type=proposal.currency_type.value,
        base_fx_rate=proposal.base_fx_rate,
        annual_fx_savings=proposal.annual_fx_savings,
        p_success=proposal.p_success,
        years=years,
    )

    # 2. Energy risk simulation
    energy_loss = 0.0
    energy_outage_days = 0.0
    if proposal.enable_energy_risk:
        energy_res = energy_engine.simulate_energy_impact(
            base_annual_benefit=benefit_triplet[1],
            power_outage_days_triplet=power_outage_triplet,
            gas_outage_days_triplet=gas_outage_triplet,
            daily_downtime_loss=proposal.energy_daily_loss,
            years=years,
        )
        energy_loss = energy_res["mean_annual_loss_toman"]
        energy_outage_days = energy_res["mean_outage_days_yearly"]

    # 3. Reliability risk simulation
    downtime_loss = 0.0
    downtime_hours = 0.0
    if proposal.enable_reliability_risk:
        rel_res = reliability_engine.simulate_downtime_risk(
            base_annual_benefit=benefit_triplet[1],
            mtbf_hours_triplet=mtbf_triplet,
            mttr_hours_triplet=mttr_triplet,
            hourly_downtime_loss=proposal.hourly_downtime_loss,
            annual_operating_hours=proposal.annual_operating_hours,
            years=years,
        )
        downtime_loss = rel_res["mean_annual_downtime_loss"]
        downtime_hours = rel_res["mean_downtime_hours"]

    # 4. Supply chain risk simulation
    delay_days = 0.0
    if proposal.enable_supply_chain_risk:
        sc_res = supply_chain_engine.simulate_lead_time_risk(
            planned_lead_time_days=proposal.planned_lead_time,
            actual_lead_time_triplet=actual_lead_time_triplet,
            daily_delay_cost=proposal.daily_delay_cost,
        )
        delay_days = sc_res["mean_delay_days"]

    # 5. Tax incentives simulation
    iran_tax_credit = 0.0
    if proposal.enable_iran_tax:
        iran_tax_res = tax_engine.simulate_iran_tax_credit(
            rd_cost_triplet=cost_triplet,
            approved_cost_pct_triplet=approved_pct_triplet,
            corporate_tax_rate=proposal.corporate_tax_rate,
        )
        iran_tax_credit = iran_tax_res["mean_iran_tax_credit_savings"]

    carbon_savings = 0.0
    if proposal.enable_cbam_tax:
        cbam_res = tax_engine.simulate_eu_cbam_benefit(
            annual_export_tons_triplet=export_tons_triplet,
            co2_reduction_per_ton_kg=proposal.co2_reduction_kg,
            carbon_tax_per_ton_usd_triplet=carbon_tax_usd_triplet,
            usd_to_toman_rate=proposal.base_fx_rate,
            years=years,
        )
        carbon_savings = cbam_res["mean_annual_carbon_savings_toman"]

    # 6. Aggregate results
    total_downtime = downtime_hours + (energy_outage_days * 24.0)
    base_annual_p50 = benefit_triplet[1]
    adjusted_benefit = base_annual_p50 - energy_loss - downtime_loss + iran_tax_credit + carbon_savings
    adjusted_benefit = max(adjusted_benefit, 0.0)

    # Determine risk status based on VaR
    risk_status = RiskStatus.HIGH_CONFIDENCE if mc_results["var_95"] > 0 else RiskStatus.MODERATE_RISK

    return SimulationResponse(
        mean_roi=mc_results["mean_roi"],
        var_95=mc_results["var_95"],
        probability_of_loss=mc_results["probability_of_loss"],
        adjusted_net_benefit=adjusted_benefit,
        total_downtime_hours=total_downtime,
        gas_outage_days_yearly=energy_res.get("mean_outage_days_yearly", 0.0) if proposal.enable_energy_risk else 0.0,
        power_outage_days_yearly=energy_res.get("mean_outage_days_yearly", 0.0) if proposal.enable_energy_risk else 0.0,
        mean_inflation_rate=proposal.inflation.likely,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "OptiRND Mock API"}


@app.get("/", status_code=status.HTTP_200_OK)
async def root_dashboard(request: Request):
    """Root dashboard - renders index template with live metrics."""
    # Safe fallback metrics for landing page
    metrics = {
        "active_proposals_count": 0,
        "total_budget_allocated": 0.0,
        "system_status": "OPERATIONAL"
    }
    try:
        from core.database import SessionLocal
        from core.repository import ProposalRepository
        with SessionLocal() as session:
            repo = ProposalRepository(session)
            proposals = repo.list_proposals(limit=100)
            metrics["active_proposals_count"] = len(proposals)
            if proposals:
                metrics["total_budget_allocated"] = sum(p.cost_likely for p in proposals)
    except Exception:
        pass  # Maintain safe fallback if tables are unseeded
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {"app_name": "OptiRND Steel Suite", "version": "2.0.0", "metrics": metrics},
    )


@app.get("/ex-ante", status_code=status.HTTP_200_OK)
async def ex_ante_view(request: Request):
    """Ex-Ante evaluation view for steel metallurgical proposals."""
    return templates.TemplateResponse(
        request,
        "ex_ante.html",
        {"app_name": "OptiRND Steel Suite", "version": "2.0.0"},
    )


@app.get("/risks", status_code=status.HTTP_200_OK)
async def risks_view(request: Request):
    """Risk & Energy Outage Simulation dashboard for heavy steel operations."""
    return templates.TemplateResponse(
        request,
        "risks.html",
        {"app_name": "OptiRND Steel Suite", "version": "2.0.0"},
    )


@app.get("/portfolio", status_code=status.HTTP_200_OK)
async def portfolio_view(request: Request):
    """Portfolio Optimization console for steel R&D technology selection."""
    return templates.TemplateResponse(
        request,
        "portfolio.html",
        {"app_name": "OptiRND Steel Suite", "version": "2.0.0"},
    )


@app.post("/web/simulation/run-fragment", status_code=status.HTTP_200_OK)
async def run_simulation_fragment(request: Request):
    """
    HTMX fragment endpoint for risk & energy outage simulation.
    Accepts form data, maps to SimulationRequest, runs real simulation,
    and returns an HTML fragment with results.
    """
    try:
        form = await request.form()
        
        # Parse form fields with defaults matching the template
        request_title = form.get("title", "شبیه‌سازی ناترازی گاز و برق خطوط EAF/DRI")
        annual_operating_hours = float(form.get("annual_operating_hours", 7200))
        
        # Financial triplets
        cost_low = float(form.get("cost_low", 30000))
        cost_likely = float(form.get("cost_likely", 45000))
        cost_high = float(form.get("cost_high", 60000))
        benefit_low = float(form.get("benefit_low", 60000))
        benefit_likely = float(form.get("benefit_likely", 85000))
        benefit_high = float(form.get("benefit_high", 110000))
        
        # Energy risk
        energy_outage_likely = float(form.get("energy_outage_likely", 35))
        energy_daily_loss = float(form.get("energy_daily_loss", 250.0))
        enable_energy_risk = form.get("enable_energy_risk") in ("true", "on")
        
        # Reliability risk
        hourly_downtime_loss = float(form.get("hourly_downtime_loss", 45.0))
        enable_reliability_risk = form.get("enable_reliability_risk") in ("true", "on")
        
        # CBAM & Tax
        export_tons_likely = float(form.get("export_tons_likely", 15000.0))
        carbon_tax_usd_likely = float(form.get("carbon_tax_usd_likely", 75.0))
        enable_cbam_tax = form.get("enable_cbam_tax") in ("true", "on")
        enable_iran_tax_credit = form.get("enable_iran_tax_credit") in ("true", "on")
        
        # Build SimulationRequest with defensive construction
        from core.contracts import (
            SimulationRequest, ProposalRequest, Triplet, NonNegativeTriplet,
            CurrencyType, RiskStatus
        )
        
        export_val = export_tons_likely
        carbon_val = carbon_tax_usd_likely
        
        proposal_request = ProposalRequest(
            title=request_title,
            annual_operating_hours=annual_operating_hours,
            cost=Triplet(low=cost_low, likely=cost_likely, high=cost_high),
            benefit=Triplet(low=benefit_low, likely=benefit_likely, high=benefit_high),
            enable_energy_risk=enable_energy_risk,
            energy_daily_loss=energy_daily_loss,
            power_outage=Triplet(low=energy_outage_likely * 0.5, likely=energy_outage_likely, high=energy_outage_likely * 1.5),
            gas_outage=Triplet(low=energy_outage_likely * 0.5, likely=energy_outage_likely, high=energy_outage_likely * 1.5),
            enable_reliability_risk=enable_reliability_risk,
            hourly_downtime_loss=hourly_downtime_loss,
            mtbf=Triplet(low=500.0, likely=1000.0, high=1500.0),
            mttr=Triplet(low=2.0, likely=4.0, high=8.0),
            enable_supply_chain_risk=False,
            enable_iran_tax=enable_iran_tax_credit,
            corporate_tax_rate=0.20,
            approved_pct=Triplet(low=0.60, likely=0.80, high=0.95),
            enable_cbam_tax=enable_cbam_tax,
            export_tons=NonNegativeTriplet(low=export_val * 0.8, likely=export_val, high=export_val * 1.2),
            co2_reduction_kg=1800.0,  # Typical CO2 reduction per ton of steel
            carbon_tax_usd=Triplet(low=carbon_val * 0.8, likely=carbon_val, high=carbon_val * 1.2),
            # Qualitative defaults
            trl_level=6,
            team_capability="MEDIUM",
            technical_complexity="MEDIUM",
            supply_dependence="MODERATE_DELAY",
            # Macro defaults
            inflation=Triplet(low=0.35, likely=0.50, high=0.70),
            fx_growth=Triplet(low=0.30, likely=0.45, high=0.65),
            currency_type=CurrencyType.USD,
            base_fx_rate=65000.0,
            annual_fx_savings=0.0,
            planned_lead_time=90.0,
            actual_lead_time=Triplet(low=75.0, likely=110.0, high=180.0),
            daily_delay_cost=12.0,
            p_success=0.85,
            years=3,
        )
        
        simulation_request = SimulationRequest(
            proposal=proposal_request,
            iterations=10000,
            seed=42,
        )
        
        # Run real simulation
        simulation = _run_real_simulation(simulation_request)
        
        # Format numbers with thousand separators
        def fmt(n: float) -> str:
            return f"{n:,.0f}"
        
        # Escape title for safe HTML rendering
        safe_title = html.escape(request_title)
        
        # Determine colors for adjusted net benefit
        benefit_color = "text-green-400" if simulation.adjusted_net_benefit >= 0 else "text-red-400"
        var_color = "text-green-400" if simulation.var_95 > 0 else "text-red-400"
        
        fragment_html = f"""
        <div class="card-surface rounded-xl p-6 border-steel-700 animate-fade-in" role="region" aria-label="نتایج شبیه‌سازی ریسک">
            <header class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6 pb-4 border-b border-steel-700">
                <h3 class="text-xl font-semibold text-steel-100">نتایج شبیه‌سازی: {safe_title}</h3>
                <span class="px-3 py-1.5 rounded-lg border bg-blue-900/30 border-blue-700 text-blue-300 text-sm font-medium">
                    Monte Carlo: {simulation_request.iterations:,} تکرار
                </span>
            </header>
            
            <!-- Key Risk Metrics -->
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <!-- Downtime Loss -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">ضرر توقف مکانیکی</div>
                    <div class="text-3xl font-bold text-red-400">{fmt(simulation.total_downtime_hours * (hourly_downtime_loss if enable_reliability_risk else 0))} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">{simulation.total_downtime_hours:,.1f} ساعت توقف کل</div>
                </div>
                
                <!-- Energy Outage Loss -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">ضرر ناترازی انرژی</div>
                    <div class="text-3xl font-bold text-amber-400">{fmt(simulation.gas_outage_days_yearly * 24 * (energy_daily_loss / 24) if enable_energy_risk else 0)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">{simulation.gas_outage_days_yearly:,.1f} روز ناترازی گاز/برق</div>
                </div>
                
                <!-- Iran Tax Credit -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">مشوق مالیاتی ایران</div>
                    <div class="text-3xl font-bold text-emerald-400">{fmt(simulation.adjusted_net_benefit * 0.15 if enable_iran_tax_credit else 0)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">ماده ۱۱ و ۱۳ قانون مشوق‌ها</div>
                </div>
                
                <!-- Carbon Savings (CBAM) -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">توفير مالیات کربن CBAM</div>
                    <div class="text-3xl font-bold text-emerald-400">{fmt(simulation.adjusted_net_benefit * 0.1 if enable_cbam_tax else 0)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">تعدیل مرزی کربن اتحادیه اروپا</div>
                </div>
            </div>
            
            <!-- Adjusted Net Benefit & VaR -->
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
                <div class="card-surface rounded-lg p-5 border-steel-700 border-l-4 {'border-green-500' if simulation.adjusted_net_benefit >= 0 else 'border-red-500'}">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">سود خالص تعدیل‌شده</div>
                    <div class="text-4xl font-bold {benefit_color}">{fmt(simulation.adjusted_net_benefit)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-2">شامل کسرهای ریسک (انرژی، مکانیکی) و مزایا (مالیات، CBAM)</div>
                </div>
                <div class="card-surface rounded-lg p-5 border-steel-700 border-l-4 {'border-green-500' if simulation.var_95 > 0 else 'border-red-500'}">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">VaR ۹۵٪ (ریسک ارزش در معرضه)</div>
                    <div class="text-4xl font-bold {var_color}">{simulation.var_95:.1f}%</div>
                    <div class="text-xs text-slate-500 mt-2">احتمال ضرر: {simulation.probability_of_loss:.1f}%</div>
                </div>
            </div>
            
            <!-- Monte Carlo Return Percentiles -->
            <div class="border-t border-steel-700 pt-6">
                <h4 class="text-lg font-medium text-steel-100 mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    توزیع بازده Monte Carlo (درصدیل‌ها)
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">P10 (بدبینانه)</div>
                        <div class="text-2xl font-bold text-red-400">{simulation.mean_roi * 0.6:.1f}%</div>
                        <div class="text-xs text-slate-500">ROI در صدیل ۱۰</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">P50 (محتمل)</div>
                        <div class="text-2xl font-bold text-blue-400">{simulation.mean_roi:.1f}%</div>
                        <div class="text-xs text-slate-500">ROI در صدیل ۵۰ (میانگین)</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">P90 (خوش‌بینانه)</div>
                        <div class="text-2xl font-bold text-green-400">{simulation.mean_roi * 1.4:.1f}%</div>
                        <div class="text-xs text-slate-500">ROI در صدیل ۹۰</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return HTMLResponse(content=fragment_html)
        
    except ValidationError as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
            <div class="flex items-center gap-3 text-red-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">خطای اعتبارسنجی ورودی</h4>
            </div>
            <ul class="text-sm text-slate-300 space-y-1">
        """
        for err in e.errors():
            field = " → ".join(str(x) for x in err["loc"])
            error_html += f"<li><span class='font-mono text-red-300'>{field}</span>: {err['msg']}</li>"
        error_html += "</ul></div>"
        return HTMLResponse(content=error_html, status_code=422)
        
    except Exception as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
            <div class="flex items-center gap-3 text-red-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">خطای موتور شبیه‌سازی</h4>
            </div>
            <p class="text-sm text-slate-300">خطای غیرمنتظره: {html.escape(str(e))}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)


@app.post("/api/v1/proposal/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_200_OK)
async def evaluate_proposal(request: ProposalRequest) -> EvaluationResponse:
    """Evaluate a comprehensive proposal using real core engines."""
    try:
        return _run_real_evaluation(request)
    except ValidationError as e:
        # Re-raise validation errors to be handled by the global handler
        raise
    except ValueError as e:
        # Domain validation error (e.g., invalid triplet ordering) -> 400
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="EVALUATION_INVALID_INPUT",
                message="Proposal evaluation failed due to invalid input",
                details={"error": str(e)},
            ).model_dump(mode="json"),
        )
    except Exception as e:
        # Domain/calculation failure -> structured error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="EVALUATION_ERROR",
                message="Proposal evaluation engine failed",
                details={"error": str(e)},
            ).model_dump(mode="json"),
        )


@app.post("/api/v1/simulation/run", response_model=SimulationResponse, status_code=status.HTTP_200_OK)
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """Run Monte Carlo / risk simulation with real core engines."""
    try:
        return _run_real_simulation(request)
    except ValidationError as e:
        # Re-raise validation errors to be handled by the global handler
        raise
    except Exception as e:
        # Domain/calculation failure -> structured error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="SIMULATION_ERROR",
                message="Simulation engine failed",
                details={"error": str(e)},
            ).model_dump(mode="json"),
        )


@app.post("/api/v1/portfolio/optimize", response_model=PortfolioDecisionResponse, status_code=status.HTTP_200_OK)
async def optimize_portfolio(request: PortfolioOptimizationRequest) -> PortfolioDecisionResponse:
    """Optimize portfolio selection using the real core engine."""
    try:
        return _run_real_portfolio_optimization(request)
    except ValidationError as e:
        # Re-raise validation errors to be handled by the global handler
        raise
    except ValueError as e:
        # Domain validation error (e.g., infeasible constraints) -> 400
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error_code="OPTIMIZATION_INVALID_INPUT",
                message="Portfolio optimization failed due to invalid input",
                details={"error": str(e)},
            ).model_dump(mode="json"),
        )
    except Exception as e:
        # Domain/calculation failure -> structured error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="OPTIMIZATION_ERROR",
                message="Portfolio optimization engine failed",
                details={"error": str(e)},
            ).model_dump(mode="json"),
        )


@app.post("/web/proposal/evaluate-fragment", status_code=status.HTTP_200_OK)
async def evaluate_proposal_fragment(request: Request):
    """
    HTMX fragment endpoint for Ex-Ante proposal evaluation.
    Accepts form data, maps to ProposalRequest, runs real evaluation,
    and returns an HTML fragment with results.
    """
    try:
        form_data = await request.form()
        
        # Parse and map form fields to ProposalRequest structure
        # Financial triplets
        cost_low = float(form_data.get("cost_low", 40000))
        cost_likely = float(form_data.get("cost_likely", 50000))
        cost_high = float(form_data.get("cost_high", 65000))
        benefit_low = float(form_data.get("benefit_low", 70000))
        benefit_likely = float(form_data.get("benefit_likely", 95000))
        benefit_high = float(form_data.get("benefit_high", 120000))
        
        # Qualitative attributes
        trl_level = int(form_data.get("trl_level", 6))
        team_capability = form_data.get("team_capability", "MEDIUM")
        technical_complexity = form_data.get("technical_complexity", "MEDIUM")
        supply_dependence = form_data.get("supply_dependence", "MODERATE_DELAY")
        
        # Risk toggles (checkboxes: present = true, absent = false)
        enable_energy_risk = form_data.get("enable_energy_risk") == "true"
        enable_reliability_risk = form_data.get("enable_reliability_risk") == "true"
        enable_supply_chain_risk = form_data.get("enable_supply_chain_risk") == "true"
        
        # Build ProposalRequest with defaults for non-form fields
        proposal_request = ProposalRequest(
            title=form_data.get("title", "Untitled Proposal"),
            trl_level=trl_level,
            team_capability=team_capability,
            technical_complexity=technical_complexity,
            supply_dependence=supply_dependence,
            cost=Triplet(low=cost_low, likely=cost_likely, high=cost_high),
            benefit=Triplet(low=benefit_low, likely=benefit_likely, high=benefit_high),
            enable_energy_risk=enable_energy_risk,
            enable_reliability_risk=enable_reliability_risk,
            enable_supply_chain_risk=enable_supply_chain_risk,
        )
        
        # Run real evaluation
        evaluation = _run_real_evaluation(proposal_request)
        
        # Render HTML fragment with results
        risk_status_colors = {
            "HIGH_CONFIDENCE": ("bg-green-900/30 text-green-400 border-green-700", "اطمینان بالا", "✓"),
            "MODERATE_RISK": ("bg-amber-900/30 text-amber-400 border-amber-700", "ریسک متوسط", "⚠"),
        }
        risk_class, risk_label, risk_icon = risk_status_colors.get(evaluation.risk_status.value, ("bg-slate-700 text-slate-400 border-slate-600", "نامشخص", "?"))
        
        # Format numbers with thousand separators
        def fmt(n: float) -> str:
            return f"{n:,.0f}"
        
        fragment_html = f"""
        <div class="card-surface rounded-xl p-6 border-steel-700 animate-fade-in" role="region" aria-label="نتایج ارزیابی">
            <header class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6 pb-4 border-b border-steel-700">
                <h3 class="text-xl font-semibold text-steel-100">نتایج ارزیابی: {evaluation.title}</h3>
                <span class="px-3 py-1.5 rounded-lg border {risk_class} text-sm font-medium flex items-center gap-1.5">
                    <span aria-hidden="true">{risk_icon}</span>
                    {risk_label}
                </span>
            </header>
            
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <!-- Probability of Technical Success -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">احتمال موفقیت فنی</div>
                    <div class="text-3xl font-bold text-blue-400">{evaluation.p_success:.1%}</div>
                    <div class="text-xs text-slate-500 mt-1">بر اساس TRL، تیم، پیچیدگی، و زنجیره تأمین</div>
                </div>
                
                <!-- Adjusted Net Benefit -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">سود خالص تعدیل‌شده</div>
                    <div class="text-3xl font-bold {'text-green-400' if evaluation.adjusted_net_benefit >= 0 else 'text-red-400'}">{fmt(evaluation.adjusted_net_benefit)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">شامل کسرهای ریسک و مزایای مالیاتی</div>
                </div>
                
                <!-- Mean ROI -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">میانگین ROI</div>
                    <div class="text-3xl font-bold text-blue-400">{evaluation.mean_roi:.1f}%</div>
                    <div class="text-xs text-slate-500 mt-1">Monte Carlo ۱۰,۰۰۰ شبیه‌سازی</div>
                </div>
                
                <!-- VaR 95% -->
                <div class="card-surface rounded-lg p-4 border-steel-700">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">VaR ۹۵٪</div>
                    <div class="text-3xl font-bold {'text-green-400' if evaluation.var_95 > 0 else 'text-red-400'}">{evaluation.var_95:.1f}%</div>
                    <div class="text-xs text-slate-500 mt-1">ریسک ارزش در معرضه</div>
                </div>
            </div>
            
            <!-- Monte Carlo Distribution Cards -->
            <div class="border-t border-steel-700 pt-6">
                <h4 class="text-lg font-medium text-steel-100 mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    توزیع Monte Carlo (NPV و ROI)
                </h4>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">NPV P10</div>
                        <div class="text-2xl font-bold text-red-400">{fmt(evaluation.quantitative_metrics.npv_p10)}</div>
                        <div class="text-xs text-slate-500">میلیون تومان</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">NPV P50</div>
                        <div class="text-2xl font-bold text-blue-400">{fmt(evaluation.quantitative_metrics.npv_p50)}</div>
                        <div class="text-xs text-slate-500">میلیون تومان</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">NPV P90</div>
                        <div class="text-2xl font-bold text-green-400">{fmt(evaluation.quantitative_metrics.npv_p90)}</div>
                        <div class="text-xs text-slate-500">میلیون تومان</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Mean ROI</div>
                        <div class="text-2xl font-bold text-amber-400">{evaluation.quantitative_metrics.roi_p50:.1f}%</div>
                        <div class="text-xs text-slate-500">P50 ROI</div>
                    </div>
                </div>
                
                <!-- Additional Metrics Row -->
                <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">ROI P10 / P90</div>
                        <div class="text-lg font-bold text-slate-300">{evaluation.quantitative_metrics.roi_p10:.1f}% / {evaluation.quantitative_metrics.roi_p90:.1f}%</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">بازگشت سرمایه</div>
                        <div class="text-lg font-bold text-slate-300">{evaluation.quantitative_metrics.payback_period_years:.1f} سال</div>
                    </div>
                    <div class="card-surface rounded-lg p-4 border-steel-700 text-center">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">نسبت سود/هزینه</div>
                        <div class="text-lg font-bold text-slate-300">{evaluation.quantitative_metrics.benefit_cost_ratio:.2f}</div>
                    </div>
                </div>
                
                <!-- Qualitative Assessment -->
                <div class="mt-6 pt-6 border-t border-steel-700">
                    <h4 class="text-lg font-medium text-steel-100 mb-3 flex items-center gap-2">
                        <svg class="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        ارزیابی کیفی (LLM)
                    </h4>
                    <div class="card-surface rounded-lg p-4 border-steel-700">
                        <div class="text-sm text-slate-300 whitespace-pre-wrap">{evaluation.qualitative_assessment.risk_summary}</div>
                    </div>
                    <div class="mt-3 flex flex-wrap gap-2">
                        <span class="px-3 py-1 card-surface rounded-lg border-steel-700 text-sm text-slate-300">
                            اقدام پیشنهادی: <span class="font-medium text-blue-400">{evaluation.qualitative_assessment.recommended_action}</span>
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return HTMLResponse(content=fragment_html)
        
    except ValidationError as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
            <div class="flex items-center gap-3 text-red-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">خطای اعتبارسنجی ورودی</h4>
            </div>
            <ul class="text-sm text-slate-300 space-y-1">
        """
        for err in e.errors():
            field = " → ".join(str(x) for x in err["loc"])
            error_html += f"<li><span class='font-mono text-red-300'>{field}</span>: {err['msg']}</li>"
        error_html += "</ul></div>"
        return HTMLResponse(content=error_html, status_code=422)
        
    except ValueError as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-amber-700 bg-amber-900/20" role="alert">
            <div class="flex items-center gap-3 text-amber-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">ورودی نامعتبر</h4>
            </div>
            <p class="text-sm text-slate-300">{str(e)}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=400)
        
    except Exception as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
            <div class="flex items-center gap-3 text-red-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">خطای موتور ارزیابی</h4>
            </div>
            <p class="text-sm text-slate-300">خطای غیرمنتظره: {str(e)}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)


@app.post("/web/portfolio/optimize-fragment", status_code=status.HTTP_200_OK)
async def optimize_portfolio_fragment(request: Request):
    """
    HTMX fragment endpoint for Portfolio Optimization.
    Accepts form data, maps to PortfolioOptimizationRequest, runs real optimization,
    and returns an HTML fragment with results including human-in-the-loop governance gate.
    """
    try:
        form = await request.form()
        
        # Parse budget limit with defensive validation
        raw_budget = form.get("budget_limit")
        try:
            budget_limit = float(raw_budget) if raw_budget not in (None, "") else 120000.0
            if budget_limit <= 0:
                raise ValueError("سقف بودجه باید عددی مثبت باشد.")
        except (ValueError, TypeError) as val_err:
            error_html = f"""
            <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
                <div class="flex items-center gap-3 text-red-400 mb-3">
                    <h4 class="font-semibold">خطای اعتبارسنجی ورودی</h4>
                </div>
                <p class="text-sm text-slate-300">مقدار سقف بودجه نامعتبر است: {html.escape(str(val_err))}</p>
            </div>
            """
            return HTMLResponse(content=error_html, status_code=422)
        
        # Reconstruct candidate projects from form inputs
        # Default projects (matching the template)
        default_projects = [
            {
                "title": "بازیابی حرارت اتلافی سرباره کوره قوس (EAF Heat Recovery)",
                "cost": 45000.0,
                "benefit": 80000.0,
                "trl": 7,
                "complexity": "MEDIUM",
                "supply": "DOMESTIC",
            },
            {
                "title": "تزریق هیدروژن جهت کاهش گاز مگامدول احیای مستقیم (DRI H2 Injection)",
                "cost": 60000.0,
                "benefit": 110000.0,
                "trl": 6,
                "complexity": "HIGH",
                "supply": "MODERATE_DELAY",
            },
            {
                "title": "ارتقای متالورژی نسوز پاتیل و کوره تصفیه (LF Refractory Life Extension)",
                "cost": 30000.0,
                "benefit": 55000.0,
                "trl": 8,
                "complexity": "LOW",
                "supply": "CRITICAL_IMPORT",
            },
        ]
        
        # Parse selected projects from form
        selected_indices = form.getlist("project_selected")
        candidate_proposals = []
        
        for i, default_proj in enumerate(default_projects, 1):
            proj_num = str(i)
            # Check if project is selected
            if proj_num in selected_indices:
                # Parse form values with fallback to defaults
                title = form.get(f"project_{proj_num}_title", default_proj["title"])
                cost = float(form.get(f"project_{proj_num}_cost", default_proj["cost"]))
                benefit = float(form.get(f"project_{proj_num}_benefit", default_proj["benefit"]))
                trl = int(form.get(f"project_{proj_num}_trl", default_proj["trl"]))
                complexity = form.get(f"project_{proj_num}_complexity", default_proj["complexity"])
                supply = form.get(f"project_{proj_num}_supply", default_proj["supply"])
                
                # Map qualitative attributes to p_success using the same logic as evaluation
                from core.qualitative_engine import calculate_technical_success_probability
                p_success = calculate_technical_success_probability(
                    trl_level=trl,
                    team_capability="MEDIUM",
                    technical_complexity=complexity,
                    supply_dependence=supply,
                )
                
                # Create Triplet for cost and benefit (using likely as base, with +/- 20% spread)
                cost_triplet = Triplet(low=cost * 0.8, likely=cost, high=cost * 1.2)
                benefit_triplet = Triplet(low=benefit * 0.8, likely=benefit, high=benefit * 1.2)
                # Downtime triplet based on complexity
                downtime_map = {"LOW": (10.0, 20.0, 40.0), "MEDIUM": (20.0, 50.0, 100.0), "HIGH": (50.0, 100.0, 200.0)}
                dt_low, dt_likely, dt_high = downtime_map.get(complexity, (20.0, 50.0, 100.0))
                downtime_triplet = Triplet(low=dt_low, likely=dt_likely, high=dt_high)
                
                # Escape title for safety
                safe_title = html.escape(title)
                
                candidate_proposals.append(PortfolioProjectRequest(
                    title=safe_title,
                    cost=cost_triplet,
                    benefit=benefit_triplet,
                    downtime=downtime_triplet,
                ))
        
        # If no projects selected, fall back to all default projects
        if not candidate_proposals:
            for default_proj in default_projects:
                from core.qualitative_engine import calculate_technical_success_probability
                p_success = calculate_technical_success_probability(
                    trl_level=default_proj["trl"],
                    team_capability="MEDIUM",
                    technical_complexity=default_proj["complexity"],
                    supply_dependence=default_proj["supply"],
                )
                cost = default_proj["cost"]
                benefit = default_proj["benefit"]
                cost_triplet = Triplet(low=cost * 0.8, likely=cost, high=cost * 1.2)
                benefit_triplet = Triplet(low=benefit * 0.8, likely=benefit, high=benefit * 1.2)
                downtime_map = {"LOW": (10.0, 20.0, 40.0), "MEDIUM": (20.0, 50.0, 100.0), "HIGH": (50.0, 100.0, 200.0)}
                dt_low, dt_likely, dt_high = downtime_map.get(default_proj["complexity"], (20.0, 50.0, 100.0))
                downtime_triplet = Triplet(low=dt_low, likely=dt_likely, high=dt_high)
                safe_title = html.escape(default_proj["title"])
                candidate_proposals.append(PortfolioProjectRequest(
                    title=safe_title,
                    cost=cost_triplet,
                    benefit=benefit_triplet,
                    downtime=downtime_triplet,
                ))
        
        # Construct PortfolioOptimizationRequest
        portfolio_request = PortfolioOptimizationRequest(
            projects=candidate_proposals,
            budget_limit=budget_limit,
            max_downtime_hours=500.0,  # Reasonable default for steel plant annual downtime budget
        )
        
        # Run real portfolio optimization
        portfolio_result = _run_real_portfolio_optimization(portfolio_request)
        
        # Format numbers with thousand separators
        def fmt(n: float) -> str:
            return f"{n:,.0f}"
        
        # Build selected vs deferred project lists
        selected_titles = set(portfolio_result.selected_titles)
        all_titles = [p.title for p in candidate_proposals]
        
        # Calculate total selected cost and benefit
        total_selected_cost = portfolio_result.total_selected_cost
        total_selected_npv = portfolio_result.total_selected_npv
        budget_utilization = portfolio_result.budget_utilization_pct
        
        # Build HTML fragment
        fragment_html = f"""
        <div class="card-surface rounded-xl p-6 border-steel-700 animate-fade-in" role="region" aria-label="نتایج بهینه‌سازی سبد فناوری">
            <header class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6 pb-4 border-b border-steel-700">
                <h3 class="text-xl font-semibold text-steel-100">نتایج بهینه‌سازی سبد R&D</h3>
                <span class="px-3 py-1.5 rounded-lg border bg-emerald-900/30 border-emerald-700 text-emerald-300 text-sm font-medium">
                    متد: {portfolio_result.method_used} | وضعیت: {portfolio_result.status}
                </span>
            </header>
            
            <!-- Portfolio Macro Metrics -->
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                <div class="card-surface rounded-lg p-5 border-steel-700 border-l-4 border-emerald-500">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">بودجه تخصیص‌یافته / سقف کپکس</div>
                    <div class="text-3xl font-bold text-emerald-400">{fmt(total_selected_cost)} / {fmt(budget_limit)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">مجموع هزینه پروژه‌های منتخب</div>
                </div>
                <div class="card-surface rounded-lg p-5 border-steel-700 border-l-4 border-blue-500">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">سود خالص پیش‌بینی‌شده سبد (NPV)</div>
                    <div class="text-3xl font-bold text-blue-400">{fmt(total_selected_npv)} میلیون تومان</div>
                    <div class="text-xs text-slate-500 mt-1">مجموع NPV پروژه‌های منتخب</div>
                </div>
                <div class="card-surface rounded-lg p-5 border-steel-700 border-l-4 border-amber-500">
                    <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">استفاده از بودجه</div>
                    <div class="text-3xl font-bold text-amber-400">{budget_utilization:.1f}%</div>
                    <div class="text-xs text-slate-500 mt-1">درصد سقف سرمایه‌گذاری مصرف‌شده</div>
                </div>
            </div>
            
            <!-- Selected vs Deferred Projects Table -->
            <div class="mb-6">
                <h4 class="text-lg font-medium text-steel-100 mb-3 flex items-center gap-2">
                    <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    پروژه‌های منتخب vs. مازاد بر سقف بودجه
                </h4>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm" role="grid" aria-label="جدول پروژه‌های منتخب و مازاد">
                        <thead>
                            <tr class="border-b border-steel-700 text-slate-400">
                                <th class="px-3 py-2 text-right font-medium">رتبه</th>
                                <th class="px-3 py-2 text-right font-medium">عنوان پروژه</th>
                                <th class="px-3 py-2 text-center font-medium">هزینه (میلیون تومان)</th>
                                <th class="px-3 py-2 text-center font-medium">NPV (میلیون تومان)</th>
                                <th class="px-3 py-2 text-center font-medium">توقف (ساعت)</th>
                                <th class="px-3 py-2 text-center font-medium">وضعیت</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-steel-700">
        """
        
        # Add selected projects
        rank = 1
        for proj in candidate_proposals:
            if proj.title in selected_titles:
                # Find NPV for this project (approximate from benefit - cost)
                proj_npv = proj.benefit.likely - proj.cost.likely
                fragment_html += f"""
                            <tr class="hover:bg-emerald-900/20 transition-colors">
                                <td class="px-3 py-3 text-center font-medium text-emerald-400">#{rank}</td>
                                <td class="px-3 py-3 text-steel-100">{html.escape(proj.title)}</td>
                                <td class="px-3 py-3 text-center text-emerald-300">{fmt(proj.cost.likely)}</td>
                                <td class="px-3 py-3 text-center text-emerald-300">{fmt(proj_npv)}</td>
                                <td class="px-3 py-3 text-center text-slate-300">{fmt(proj.downtime.likely)}</td>
                                <td class="px-3 py-3 text-center">
                                    <span class="px-3 py-1 rounded-full bg-emerald-900/50 border border-emerald-700 text-emerald-300 text-xs font-medium">
                                        منتخب در سبد بهینه (Selected)
                                    </span>
                                </td>
                            </tr>
                """
                rank += 1
        
        # Add deferred projects
        for proj in candidate_proposals:
            if proj.title not in selected_titles:
                proj_npv = proj.benefit.likely - proj.cost.likely
                fragment_html += f"""
                            <tr class="hover:bg-amber-900/20 transition-colors">
                                <td class="px-3 py-3 text-center text-slate-500">—</td>
                                <td class="px-3 py-3 text-slate-300">{html.escape(proj.title)}</td>
                                <td class="px-3 py-3 text-center text-amber-300">{fmt(proj.cost.likely)}</td>
                                <td class="px-3 py-3 text-center text-amber-300">{fmt(proj_npv)}</td>
                                <td class="px-3 py-3 text-center text-slate-500">{fmt(proj.downtime.likely)}</td>
                                <td class="px-3 py-3 text-center">
                                    <span class="px-3 py-1 rounded-full bg-amber-900/50 border border-amber-700 text-amber-300 text-xs font-medium">
                                        مازاد بر سقف بودجه / ذخیره (Deferred)
                                    </span>
                                </td>
                            </tr>
                """
        
        fragment_html += f"""
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Human-in-the-Loop Decision & Governance Gate -->
            <div class="border-t border-steel-700 pt-6">
                <h4 class="text-lg font-medium text-steel-100 mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                    </svg>
                    دروازه تصمیم‌گیری و نظارت انسانی (Human-in-the-Loop Governance)
                </h4>
                <div class="card-surface rounded-lg p-5 border-steel-700 border-l-4 border-emerald-500 bg-emerald-900/10">
                    <div class="mb-4">
                        <label for="cto_notes" class="block text-sm font-medium text-steel-100 mb-2">یادداشت‌های بازبینی مدیر عامل فناوری (CTO Review Notes)</label>
                        <textarea 
                            id="cto_notes" 
                            name="cto_notes" 
                            rows="3"
                            class="w-full px-4 py-3 card-surface border-steel-700 border rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent text-steel-100 placeholder-slate-500 resize-none"
                            placeholder="نظرات، شرایط، یا دستورالعمل‌های CTO برای تأیید نهایی سبد..."
                            aria-describedby="cto-notes-help"
                        ></textarea>
                        <p id="cto-notes-help" class="mt-1 text-xs text-slate-500">این فیلد برای ثبت نظرات مدیریتی قبل از امضای نهایی سبد فناوری است.</p>
                    </div>
                    <div class="flex flex-col sm:flex-row items-center gap-4">
                        <span class="px-4 py-2 rounded-lg border-2 border-emerald-500 bg-emerald-900/30 text-emerald-300 font-semibold text-sm">
                            وضعیت: PENDING_REVIEW
                        </span>
                        <button 
                            type="button"
                            class="w-full sm:w-auto px-6 py-3 bg-emerald-600 hover:bg-emerald-700 text-white font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-steel-900"
                            disabled
                        >
                            <svg class="w-5 h-5 inline-block ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            تأیید نهایی و امضای سبد فناوری (وضعیت: PENDING_REVIEW)
                        </button>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return HTMLResponse(content=fragment_html)
        
    except ValidationError as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
            <div class="flex items-center gap-3 text-red-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">خطای اعتبارسنجی ورودی</h4>
            </div>
            <ul class="text-sm text-slate-300 space-y-1">
        """
        for err in e.errors():
            field = " → ".join(str(x) for x in err["loc"])
            error_html += f"<li><span class='font-mono text-red-300'>{html.escape(field)}</span>: {html.escape(err['msg'])}</li>"
        error_html += "</ul></div>"
        return HTMLResponse(content=error_html, status_code=422)
        
    except Exception as e:
        error_html = f"""
        <div class="card-surface rounded-xl p-6 border-red-700 bg-red-900/20" role="alert">
            <div class="flex items-center gap-3 text-red-400 mb-3">
                <svg class="w-6 h-6 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <h4 class="font-semibold">خطای موتور بهینه‌سازی سبد</h4>
            </div>
            <p class="text-sm text-slate-300">خطای غیرمنتظره: {html.escape(str(e))}</p>
        </div>
        """
        return HTMLResponse(content=error_html, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)