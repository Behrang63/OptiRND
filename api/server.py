"""
FastAPI mock server for OptiRND evaluation pipeline.
Serves on port 8001 with OWASP security headers and strict CORS.
"""
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

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
)

# Core simulation engines
from core.monte_carlo import MonteCarloEngine
from core.energy_simulation import EnergyRiskEngine
from core.reliability_engine import ReliabilityEngine
from core.supply_chain_engine import SupplyChainEngine
from core.carbon_tax_engine import CarbonTaxEngine
from core.portfolio_optimizer import PortfolioOptimizationEngine


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


# OWASP Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
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
# MOCK RESPONSE GENERATORS (kept for /evaluate endpoint only)
# =============================================================================

def _mock_evaluation_response(request: ProposalRequest) -> EvaluationResponse:
    """Generate a mock EvaluationResponse from a ProposalRequest."""
    return EvaluationResponse(
        title=request.title,
        cost=request.cost,
        benefit=request.benefit,
        p_success=request.p_success,
        total_downtime_hours=120.5,
        energy_loss_toman=5000000.0,
        downtime_loss_toman=2400000.0,
        lead_time_delay_days=15.0,
        iran_tax_credit_toman=15000000.0,
        carbon_savings_toman=0.0,
        adjusted_net_benefit=25000000.0,
        mean_roi=0.35,
        var_95=-5000000.0,
        probability_of_loss=12.5,
        risk_status=RiskStatus.MODERATE_RISK,
        quantitative_metrics=QuantitativeMetrics(
            npv_p10=10000000.0,
            npv_p50=25000000.0,
            npv_p90=45000000.0,
            roi_p10=0.15,
            roi_p50=0.35,
            roi_p90=0.55,
            payback_period_years=2.8,
            benefit_cost_ratio=1.4,
        ),
        qualitative_assessment=QualitativeAssessment(
            technical_success_probability=0.85,
            recommended_action="APPROVE_WITH_CONDITIONS",
            risk_summary="Moderate risk profile with acceptable ROI. Energy and supply chain risks identified.",
        ),
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


@app.post("/api/v1/proposal/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_200_OK)
async def evaluate_proposal(request: ProposalRequest) -> EvaluationResponse:
    """Evaluate a comprehensive proposal."""
    return _mock_evaluation_response(request)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)