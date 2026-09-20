"""
Live black-box tests for the OptiRND mock API server.
Tests against the FastAPI ASGI app using httpx test client.
Validates all endpoints return HTTP 200 and schema-valid responses.
"""
import pytest
import httpx
from api.server import app
from core.contracts import (
    EvaluationResponse,
    PortfolioDecisionResponse,
    PortfolioOptimizationRequest,
    PortfolioProjectRequest,
    ProposalRequest,
    SimulationRequest,
    SimulationResponse,
    Triplet,
    NonNegativeTriplet,
    CurrencyType,
    ErrorResponse,
)


@pytest.fixture
def client():
    """Create an httpx AsyncClient for testing the FastAPI app."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def valid_proposal_request() -> ProposalRequest:
    """Create a valid ProposalRequest for testing."""
    return ProposalRequest(
        title="Test Solar Project",
        years=5,
        p_success=0.9,
        cost=Triplet(low=100.0, likely=150.0, high=200.0),
        benefit=Triplet(low=200.0, likely=300.0, high=400.0),
        inflation=Triplet(low=0.3, likely=0.4, high=0.5),
        fx_growth=Triplet(low=0.2, likely=0.35, high=0.5),
        currency_type=CurrencyType.USD,
        base_fx_rate=65000.0,
        annual_fx_savings=10000.0,
        enable_energy_risk=True,
        power_outage=Triplet(low=5.0, likely=10.0, high=20.0),
        gas_outage=Triplet(low=10.0, likely=20.0, high=30.0),
        energy_daily_loss=30.0,
        enable_reliability_risk=True,
        mtbf=Triplet(low=800.0, likely=1200.0, high=1600.0),
        mttr=Triplet(low=1.0, likely=3.0, high=6.0),
        hourly_downtime_loss=15.0,
        annual_operating_hours=7000.0,
        enable_supply_chain_risk=True,
        planned_lead_time=60.0,
        actual_lead_time=Triplet(low=50.0, likely=80.0, high=120.0),
        daily_delay_cost=10.0,
        enable_iran_tax=True,
        corporate_tax_rate=0.2,
        approved_pct=Triplet(low=0.7, likely=0.85, high=0.95),
        enable_cbam_tax=False,
        export_tons=NonNegativeTriplet(low=0.0, likely=0.0, high=0.0),
        co2_reduction_kg=0.0,
        carbon_tax_usd=Triplet(low=50.0, likely=75.0, high=100.0),
    )


@pytest.fixture
def valid_portfolio_request() -> PortfolioOptimizationRequest:
    """Create a valid PortfolioOptimizationRequest for testing."""
    return PortfolioOptimizationRequest(
        projects=[
            PortfolioProjectRequest(
                title="Project A",
                cost=Triplet(low=50.0, likely=75.0, high=100.0),
                benefit=Triplet(low=100.0, likely=150.0, high=200.0),
                downtime=Triplet(low=10.0, likely=20.0, high=30.0),
            ),
            PortfolioProjectRequest(
                title="Project B",
                cost=Triplet(low=30.0, likely=50.0, high=70.0),
                benefit=Triplet(low=80.0, likely=120.0, high=160.0),
                downtime=Triplet(low=5.0, likely=15.0, high=25.0),
            ),
            PortfolioProjectRequest(
                title="Project C",
                cost=Triplet(low=20.0, likely=40.0, high=60.0),
                benefit=Triplet(low=60.0, likely=90.0, high=120.0),
                downtime=Triplet(low=8.0, likely=18.0, high=28.0),
            ),
        ],
        budget_limit=200.0,
        max_downtime_hours=100.0,
    )


@pytest.fixture
def valid_simulation_request(valid_proposal_request: ProposalRequest) -> SimulationRequest:
    """Create a valid SimulationRequest for testing."""
    return SimulationRequest(
        proposal=valid_proposal_request,
        iterations=5000,
        seed=42,
    )


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "OptiRND Mock API"

    @pytest.mark.asyncio
    async def test_health_has_security_headers(self, client):
        response = await client.get("/api/v1/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Content-Security-Policy" in response.headers


class TestProposalEvaluateEndpoint:
    """Tests for POST /api/v1/proposal/evaluate."""

    @pytest.mark.asyncio
    async def test_evaluate_valid_request_returns_200(self, client, valid_proposal_request):
        response = await client.post(
            "/api/v1/proposal/evaluate",
            json=valid_proposal_request.model_dump(mode="json"),
        )
        assert response.status_code == 200

        # Validate response against EvaluationResponse schema
        eval_response = EvaluationResponse.model_validate(response.json())
        assert eval_response.title == valid_proposal_request.title
        assert 0.1 <= eval_response.p_success <= 0.95
        assert eval_response.p_success == 0.68
        assert eval_response.risk_status in ["HIGH_CONFIDENCE", "MODERATE_RISK"]
        assert 0 <= eval_response.probability_of_loss <= 100
        assert isinstance(eval_response.adjusted_net_benefit, (int, float))

    @pytest.mark.asyncio
    async def test_evaluate_invalid_payload_returns_422(self, client):
        response = await client.post("/api/v1/proposal/evaluate", json={})
        assert response.status_code == 422

        # Validate error response structure
        error_response = ErrorResponse.model_validate(response.json())
        assert error_response.error_code == "VALIDATION_ERROR"
        assert error_response.message == "Request validation failed"
        assert error_response.details is not None
        assert "errors" in error_response.details

    @pytest.mark.asyncio
    async def test_evaluate_empty_payload_returns_422(self, client):
        response = await client.post("/api/v1/proposal/evaluate", json=None)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_evaluate_missing_required_fields_returns_422(self, client):
        # Missing required 'title', 'cost', 'benefit'
        response = await client.post(
            "/api/v1/proposal/evaluate",
            json={"title": "Test", "years": 3},
        )
        assert response.status_code == 422
        error_response = ErrorResponse.model_validate(response.json())
        assert error_response.error_code == "VALIDATION_ERROR"


class TestSimulationRunEndpoint:
    """Tests for POST /api/v1/simulation/run."""

    @pytest.mark.asyncio
    async def test_simulation_valid_request_returns_200(self, client, valid_simulation_request):
        response = await client.post(
            "/api/v1/simulation/run",
            json=valid_simulation_request.model_dump(mode="json"),
        )
        assert response.status_code == 200

        # Validate response against SimulationResponse schema
        sim_response = SimulationResponse.model_validate(response.json())
        assert sim_response.mean_roi >= 0
        assert sim_response.probability_of_loss >= 0
        assert sim_response.adjusted_net_benefit >= 0
        assert sim_response.total_downtime_hours >= 0

    @pytest.mark.asyncio
    async def test_simulation_invalid_payload_returns_422(self, client):
        response = await client.post("/api/v1/simulation/run", json={})
        assert response.status_code == 422
        error_response = ErrorResponse.model_validate(response.json())
        assert error_response.error_code == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_simulation_missing_proposal_returns_422(self, client):
        response = await client.post(
            "/api/v1/simulation/run",
            json={"iterations": 1000},
        )
        assert response.status_code == 422


class TestPortfolioOptimizeEndpoint:
    """Tests for POST /api/v1/portfolio/optimize."""

    @pytest.mark.asyncio
    async def test_optimize_valid_request_returns_200(self, client, valid_portfolio_request):
        response = await client.post(
            "/api/v1/portfolio/optimize",
            json=valid_portfolio_request.model_dump(mode="json"),
        )
        assert response.status_code == 200

        # Validate response against PortfolioDecisionResponse schema
        port_response = PortfolioDecisionResponse.model_validate(response.json())
        assert port_response.status in ["OPTIMAL", "FAILED", "EMPTY"]
        assert isinstance(port_response.selected_titles, list)
        assert port_response.total_selected_npv >= 0
        assert port_response.total_selected_cost >= 0
        assert 0 <= port_response.budget_utilization_pct <= 100
        assert 0 <= port_response.downtime_utilization_pct <= 100

    @pytest.mark.asyncio
    async def test_optimize_invalid_payload_returns_422(self, client):
        response = await client.post("/api/v1/portfolio/optimize", json={})
        assert response.status_code == 422
        error_response = ErrorResponse.model_validate(response.json())
        assert error_response.error_code == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_optimize_empty_projects_returns_422(self, client):
        response = await client.post(
            "/api/v1/portfolio/optimize",
            json={"projects": [], "budget_limit": 100.0, "max_downtime_hours": 50.0},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_optimize_negative_budget_returns_422(self, client, valid_portfolio_request):
        invalid_request = valid_portfolio_request.model_copy(update={"budget_limit": -10.0})
        response = await client.post(
            "/api/v1/portfolio/optimize",
            json=invalid_request.model_dump(mode="json"),
        )
        assert response.status_code == 422


class TestSecurityHeaders:
    """Tests for OWASP security headers on all endpoints."""

    @pytest.mark.asyncio
    async def test_all_endpoints_have_security_headers(self, client, valid_proposal_request):
        endpoints = [
            ("GET", "/api/v1/health", None),
            ("POST", "/api/v1/proposal/evaluate", valid_proposal_request.model_dump(mode="json")),
            ("POST", "/api/v1/simulation/run", {"proposal": valid_proposal_request.model_dump(mode="json"), "iterations": 1000}),
            ("POST", "/api/v1/portfolio/optimize", {"projects": [{"title": "P1", "cost": {"low": 10, "likely": 20, "high": 30}, "benefit": {"low": 20, "likely": 30, "high": 40}, "downtime": {"low": 5, "likely": 10, "high": 15}}], "budget_limit": 100, "max_downtime_hours": 50}),
        ]

        for method, path, payload in endpoints:
            if method == "GET":
                response = await client.get(path)
            else:
                response = await client.post(path, json=payload)

            # All responses should have security headers regardless of status code
            assert response.headers.get("X-Content-Type-Options") == "nosniff", f"Missing on {path}"
            assert response.headers.get("X-Frame-Options") == "DENY", f"Missing on {path}"
            assert response.headers.get("X-XSS-Protection") == "1; mode=block", f"Missing on {path}"


class TestCORSConfiguration:
    """Tests for strict CORS configuration."""

    @pytest.mark.asyncio
    async def test_cors_preflight_allowed_origin(self, client):
        response = await client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert response.status_code == 200
        assert response.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    @pytest.mark.asyncio
    async def test_cors_preflight_disallowed_origin(self, client):
        response = await client.options(
            "/api/v1/health",
            headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "GET"},
        )
        # Should not allow the evil origin
        assert response.headers.get("Access-Control-Allow-Origin") != "http://evil.com"
        # Wildcard should not be present
        assert response.headers.get("Access-Control-Allow-Origin") != "*"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])