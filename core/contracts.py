"""
Pydantic v2 data contracts for the OptiRND evaluation pipeline.

Frozen Data Contracts (DTOs) - Phase 1: Contract Freezing
- All inputs/outputs cross module boundaries as validated models
- Strict sanitization: explicit string constraints, numerical boundaries, no Any/dict
- Pure Pydantic models and Enums only - no domain logic, no deep imports
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskStatus(str, Enum):
    """Enumeration of task execution states for the async task engine."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CheckerVerdict(str, Enum):
    """Final compliance verdict from the Checker Sub-Agent."""
    APPROVED = "APPROVED"
    APPROVED_WITH_CONDITIONS = "APPROVED_WITH_CONDITIONS"
    REJECTED = "REJECTED"


class RiskStatus(str, Enum):
    """Risk classification from evaluation."""
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MODERATE_RISK = "MODERATE_RISK"


class CurrencyType(str, Enum):
    """Supported currency types."""
    USD = "USD"
    EUR = "EUR"
    CNY = "CNY"


class Triplet(BaseModel):
    """Ordered triplet (low, likely, high) with strict ordering constraint."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float = Field(gt=0)
    likely: float = Field(gt=0)
    high: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "Triplet":
        if not (self.low <= self.likely <= self.high):
            raise ValueError(f"Triplet must satisfy low <= likely <= high, got ({self.low}, {self.likely}, {self.high})")
        return self


class NonNegativeTriplet(BaseModel):
    """Ordered triplet allowing zero values (for export_tons, etc.)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float = Field(ge=0)
    likely: float = Field(ge=0)
    high: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "NonNegativeTriplet":
        if not (self.low <= self.likely <= self.high):
            raise ValueError(f"Triplet must satisfy low <= likely <= high, got ({self.low}, {self.likely}, {self.high})")
        return self


# =============================================================================
# REQUEST DTOs (Incoming Analysis, Scenario Configs, Simulation Inputs)
# =============================================================================

class ProposalRequest(BaseModel):
    """Validated input for comprehensive proposal evaluation."""
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255)
    years: int = Field(default=3, ge=1, le=10)
    p_success: float = Field(default=0.85, ge=0.0, le=1.0)

    # Financial triplets (million Toman) - strictly positive
    cost: Triplet
    benefit: Triplet

    # Macro assumptions
    inflation: Triplet = Field(default_factory=lambda: Triplet(low=0.35, likely=0.50, high=0.70))
    fx_growth: Triplet = Field(default_factory=lambda: Triplet(low=0.30, likely=0.45, high=0.65))
    currency_type: CurrencyType = CurrencyType.USD
    base_fx_rate: float = Field(default=65000.0, gt=0)
    annual_fx_savings: float = Field(default=0.0, ge=0)

    # Operational risk toggles & parameters
    enable_energy_risk: bool = True
    power_outage: Triplet = Field(default_factory=lambda: Triplet(low=10.0, likely=20.0, high=35.0))
    gas_outage: Triplet = Field(default_factory=lambda: Triplet(low=15.0, likely=30.0, high=45.0))
    energy_daily_loss: float = Field(default=50.0, ge=0)

    enable_reliability_risk: bool = True
    mtbf: Triplet = Field(default_factory=lambda: Triplet(low=500.0, likely=1000.0, high=1500.0))
    mttr: Triplet = Field(default_factory=lambda: Triplet(low=2.0, likely=4.0, high=8.0))
    hourly_downtime_loss: float = Field(default=20.0, ge=0)
    annual_operating_hours: float = Field(default=7200.0, gt=0)

    enable_supply_chain_risk: bool = True
    planned_lead_time: float = Field(default=90.0, ge=0)
    actual_lead_time: Triplet = Field(default_factory=lambda: Triplet(low=75.0, likely=110.0, high=180.0))
    daily_delay_cost: float = Field(default=12.0, ge=0)

    # Tax incentives
    enable_iran_tax: bool = True
    corporate_tax_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    approved_pct: Triplet = Field(default_factory=lambda: Triplet(low=0.60, likely=0.80, high=0.95))

    enable_cbam_tax: bool = False
    export_tons: NonNegativeTriplet = Field(default_factory=lambda: NonNegativeTriplet(low=0.0, likely=0.0, high=0.0))
    co2_reduction_kg: float = Field(default=0.0, ge=0)
    carbon_tax_usd: Triplet = Field(default_factory=lambda: Triplet(low=60.0, likely=85.0, high=120.0))

    @model_validator(mode="after")
    def _validate_cbam(self) -> "ProposalRequest":
        if self.enable_cbam_tax and self.export_tons.likely <= 0:
            raise ValueError("export_tons.likely must be positive when CBAM is enabled")
        return self


class PortfolioProjectRequest(BaseModel):
    """Single project row for portfolio optimization."""
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255)
    cost: Triplet
    benefit: Triplet
    downtime: Triplet


class PortfolioOptimizationRequest(BaseModel):
    """Request for portfolio optimization."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    projects: List[PortfolioProjectRequest] = Field(min_length=1, max_length=100)
    budget_limit: float = Field(gt=0)
    max_downtime_hours: float = Field(ge=0)


class SimulationRequest(BaseModel):
    """Request for Monte Carlo / risk simulation."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal: ProposalRequest
    iterations: int = Field(default=10000, ge=1000, le=100000)
    seed: Optional[int] = Field(default=None, ge=0)


class ComplianceCheckRequest(BaseModel):
    """Request for compliance audit."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal: ProposalRequest
    simulation_result: "SimulationResponse"


# =============================================================================
# RESPONSE DTOs (System Outputs, Evaluation Scores, Execution Status, Errors)
# =============================================================================

class QualitativeAssessment(BaseModel):
    """Structured LLM (or mock) assessment block."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    technical_success_probability: float = Field(default=0.85, ge=0.0, le=1.0)
    recommended_action: str = Field(default="APPROVE_WITH_CONDITIONS", max_length=64)
    risk_summary: str = Field(default="", max_length=2000)


class QuantitativeMetrics(BaseModel):
    """Structured quantitative metrics - replaces loose dict."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    npv_p10: float
    npv_p50: float
    npv_p90: float
    roi_p10: float
    roi_p50: float
    roi_p90: float
    payback_period_years: float = Field(ge=0)
    benefit_cost_ratio: float = Field(ge=0)


class EvaluationResponse(BaseModel):
    """Typed output of comprehensive proposal evaluation."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, max_length=255)
    cost: Triplet
    benefit: Triplet
    p_success: float = Field(ge=0.0, le=1.0)

    total_downtime_hours: float = Field(ge=0)
    energy_loss_toman: float = Field(ge=0)
    downtime_loss_toman: float = Field(ge=0)
    lead_time_delay_days: float = Field(ge=0)
    iran_tax_credit_toman: float = Field(ge=0)
    carbon_savings_toman: float = Field(ge=0)
    adjusted_net_benefit: float = Field(ge=0)

    mean_roi: float
    var_95: float
    probability_of_loss: float = Field(ge=0, le=100)
    risk_status: RiskStatus

    quantitative_metrics: QuantitativeMetrics
    qualitative_assessment: QualitativeAssessment


class SimulationResponse(BaseModel):
    """Quantitative outputs from Monte Carlo / risk engines."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    mean_roi: float = 0.0
    var_95: float = 0.0
    probability_of_loss: float = 0.0
    adjusted_net_benefit: float = 0.0
    total_downtime_hours: float = 0.0
    gas_outage_days_yearly: float = 0.0
    power_outage_days_yearly: float = 0.0
    mean_inflation_rate: float = 0.0


class PortfolioDecisionResponse(BaseModel):
    """Typed output of the portfolio optimization step."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["OPTIMAL", "FAILED", "EMPTY"]
    method_used: str = Field(default="stochastic", max_length=32)
    selected_titles: List[str] = Field(default_factory=list)
    total_selected_npv: float = 0.0
    total_selected_cost: float = 0.0
    total_selected_downtime: float = 0.0
    budget_utilization_pct: float = Field(default=0.0, ge=0, le=100)
    downtime_utilization_pct: float = Field(default=0.0, ge=0, le=100)


class ComplianceAuditResponse(BaseModel):
    """Structured compliance audit output from the Checker Sub-Agent."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_approved: bool
    compliance_score: float = Field(ge=0.0, le=1.0)
    flagged_risks: List[str] = Field(default_factory=list)
    recommended_mitigations: List[str] = Field(default_factory=list)
    checker_verdict: CheckerVerdict


class TaskStatusResponse(BaseModel):
    """Result payload for an asynchronous evaluation task."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    status: TaskStatus
    progress: float = Field(ge=0.0, le=1.0)
    stage: str = Field(default="", max_length=128)
    result: Optional[EvaluationResponse] = None
    error_message: Optional[str] = Field(default=None, max_length=2000)
    created_at: str
    completed_at: Optional[str] = None


class ErrorResponse(BaseModel):
    """Structured error payload for API boundaries."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=512)
    details: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


# Forward reference resolution
ComplianceCheckRequest.model_rebuild()