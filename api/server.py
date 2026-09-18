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
# MOCK RESPONSE GENERATORS
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


def _mock_simulation_response(request: SimulationRequest) -> SimulationResponse:
    """Generate a mock SimulationResponse from a SimulationRequest."""
    return SimulationResponse(
        mean_roi=0.34,
        var_95=-4800000.0,
        probability_of_loss=13.2,
        adjusted_net_benefit=24500000.0,
        total_downtime_hours=118.7,
        gas_outage_days_yearly=12.3,
        power_outage_days_yearly=8.7,
        mean_inflation_rate=0.48,
    )


def _mock_portfolio_response(request: PortfolioOptimizationRequest) -> PortfolioDecisionResponse:
    """Generate a mock PortfolioDecisionResponse from a PortfolioOptimizationRequest."""
    selected = [p.title for p in request.projects[:3]]
    return PortfolioDecisionResponse(
        status="OPTIMAL",
        method_used="stochastic",
        selected_titles=selected,
        total_selected_npv=75000000.0,
        total_selected_cost=45000000.0,
        total_selected_downtime=300.0,
        budget_utilization_pct=75.0,
        downtime_utilization_pct=60.0,
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
    """Run Monte Carlo / risk simulation."""
    return _mock_simulation_response(request)


@app.post("/api/v1/portfolio/optimize", response_model=PortfolioDecisionResponse, status_code=status.HTTP_200_OK)
async def optimize_portfolio(request: PortfolioOptimizationRequest) -> PortfolioDecisionResponse:
    """Optimize portfolio selection."""
    return _mock_portfolio_response(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)