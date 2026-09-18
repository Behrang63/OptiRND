"""
Pydantic v2 data contracts for the OptiRND evaluation pipeline.

Purpose (Phase 2 - Contract Enforcement):
    * All agent inputs/outputs cross module boundaries as validated models,
      not loose dicts - eliminating the ~30-key stringly-typed contract.
    * Invalid data (inverted triplets, non-positive financials, out-of-range
      probabilities) is rejected at the boundary with precise field errors
      instead of silently flowing into the risk engines.
"""
from __future__ import annotations

import datetime
from enum import Enum
from typing import List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Triplet = Tuple[float, float, float]


class TaskStatus(str, Enum):
    """Enumeration of task execution states for the async task engine."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskResult(BaseModel):
    """
    Result payload for an asynchronous evaluation task.
    
    Thread-safe contract for the AsyncTaskManager in-process worker pool.
    Progress and stage fields enable real-time UI polling without blocking.
    """
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: str = ""
    result: Optional["EvaluationResult"] = None
    error_message: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    completed_at: Optional[str] = None


def _check_triplet_ordering(name: str, value: Triplet) -> Triplet:
    low, likely, high = value
    if not (low <= likely <= high):
        raise ValueError(
            f"{name} must satisfy low <= likely <= high, got {value!r}"
        )
    return value


class ProposalInput(BaseModel):
    """Validated input for ExAnteAgent.evaluate_comprehensive_proposal()."""

    model_config = ConfigDict(strict=False, str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255)
    years: int = Field(default=3, ge=1, le=10)
    p_success: float = Field(default=0.85, ge=0.0, le=1.0)

    # --- Financial triplets (million Toman). Positive by data-integrity rule. ---
    cost_p10: float = Field(gt=0)
    cost_p50: float = Field(gt=0)
    cost_p90: float = Field(gt=0)
    benefit_p10: float = Field(gt=0)
    benefit_p50: float = Field(gt=0)
    benefit_p90: float = Field(gt=0)

    # --- Macro assumptions ---
    inflation_triplet: Triplet = (0.35, 0.50, 0.70)
    fx_growth_triplet: Triplet = (0.30, 0.45, 0.65)
    currency_type: Literal["USD", "EUR", "CNY"] = "USD"
    base_fx_rate: float = Field(default=65000.0, gt=0)
    annual_fx_savings: float = Field(default=0.0, ge=0)

    # --- Operational risk toggles & parameters ---
    enable_energy_risk: bool = True
    power_outage_triplet: Triplet = (10.0, 20.0, 35.0)
    gas_outage_triplet: Triplet = (15.0, 30.0, 45.0)
    energy_daily_loss: float = Field(default=50.0, ge=0)

    enable_reliability_risk: bool = True
    mtbf_triplet: Triplet = (500.0, 1000.0, 1500.0)
    mttr_triplet: Triplet = (2.0, 4.0, 8.0)
    hourly_downtime_loss: float = Field(default=20.0, ge=0)
    annual_operating_hours: float = Field(default=7200.0, gt=0)

    enable_supply_chain_risk: bool = True
    planned_lead_time: float = Field(default=90.0, ge=0)
    actual_lead_time_triplet: Triplet = (75.0, 110.0, 180.0)
    daily_delay_cost: float = Field(default=12.0, ge=0)

    # --- Tax incentives ---
    enable_iran_tax: bool = True
    corporate_tax_rate: float = Field(default=0.20, ge=0.0, le=1.0)
    approved_pct_triplet: Triplet = (0.60, 0.80, 0.95)

    enable_cbam_tax: bool = False
    export_tons_triplet: Triplet = (0.0, 0.0, 0.0)
    co2_reduction_kg: float = Field(default=0.0, ge=0)
    carbon_tax_usd_triplet: Triplet = (60.0, 85.0, 120.0)

    # --- Cross-field validation: triplet ordering (independent validators,
    # so one inverted triplet reports on its own field only) ---
    @field_validator("inflation_triplet", "fx_growth_triplet",
                     "power_outage_triplet", "gas_outage_triplet",
                     "mtbf_triplet", "mttr_triplet",
                     "actual_lead_time_triplet", "approved_pct_triplet",
                     "export_tons_triplet", "carbon_tax_usd_triplet")
    @classmethod
    def _triplets_ordered(cls, v: Triplet, info) -> Triplet:
        return _check_triplet_ordering(info.field_name, v)

    @model_validator(mode="after")
    def _cost_benefit_ordering(self) -> "ProposalInput":
        _check_triplet_ordering("cost", (self.cost_p10, self.cost_p50, self.cost_p90))
        _check_triplet_ordering("benefit", (self.benefit_p10, self.benefit_p50, self.benefit_p90))
        if self.enable_cbam_tax and self.export_tons_triplet[1] <= 0:
            raise ValueError("export_tons_triplet P50 must be positive when CBAM is enabled")
        return self


class QualitativeAssessment(BaseModel):
    """Structured LLM (or mock) assessment block."""

    model_config = ConfigDict(extra="allow")

    technical_success_probability: float = Field(default=0.85, ge=0.0, le=1.0)
    recommended_action: str = "APPROVE_WITH_CONDITIONS"
    risk_summary: str = ""


class EvaluationResult(BaseModel):
    """Typed output of ExAnteAgent.evaluate_comprehensive_proposal()."""

    model_config = ConfigDict(extra="allow")  # rag_evidence & raw arrays pass through

    title: str = Field(min_length=1)
    cost_p10: float
    cost_p50: float
    cost_p90: float
    benefit_p10: float
    benefit_p50: float
    benefit_p90: float
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
    risk_status: Literal["HIGH_CONFIDENCE", "MODERATE_RISK"]

    quantitative_metrics: dict
    qualitative_assessment: QualitativeAssessment


class PortfolioProject(BaseModel):
    """Row consumed by PortfolioOptimizationEngine.optimize_portfolio()."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(min_length=1)
    cost_p10: float = Field(ge=0)
    cost_p50: float = Field(ge=0)
    cost_p90: float = Field(ge=0)
    benefit_p10: float = Field(ge=0)
    benefit_p50: float = Field(ge=0)
    benefit_p90: float = Field(ge=0)
    dt_p10: float = Field(ge=0)
    dt_p50: float = Field(ge=0)
    dt_p90: float = Field(ge=0)


class PortfolioDecision(BaseModel):
    """Typed output of the portfolio optimization step."""

    status: Literal["OPTIMAL", "FAILED", "EMPTY"]
    method_used: str = "stochastic"
    selected_titles: List[str] = Field(default_factory=list)
    total_selected_npv: float = 0.0
    total_selected_cost: float = 0.0
    total_selected_downtime: float = 0.0
    budget_utilization_pct: float = Field(default=0.0, ge=0)
    downtime_utilization_pct: float = Field(default=0.0, ge=0)


class SimulationResult(BaseModel):
    """Quantitative outputs from Monte Carlo / risk engines.

    Used by the Compliance Checker Sub-Agent for threshold verification.
    Fields are optional (total=False) because not all engines expose every metric.
    """
    model_config = ConfigDict(extra="allow")

    mean_roi: float = 0.0
    var_95: float = 0.0
    probability_of_loss: float = 0.0
    adjusted_net_benefit: float = 0.0
    total_downtime_hours: float = 0.0
    gas_outage_days_yearly: float = 0.0
    power_outage_days_yearly: float = 0.0
    mean_inflation_rate: float = 0.0


class PortfolioDecision(BaseModel):
    """Typed output of the portfolio optimization step."""

    status: Literal["OPTIMAL", "FAILED", "EMPTY"]
    method_used: str = "stochastic"
    selected_titles: List[str] = Field(default_factory=list)
    total_selected_npv: float = 0.0
    total_selected_cost: float = 0.0
    total_selected_downtime: float = 0.0
    budget_utilization_pct: float = Field(default=0.0, ge=0)
    downtime_utilization_pct: float = Field(default=0.0, ge=0)


class ComplianceAuditReport(BaseModel):
    """Structured compliance audit output from the Checker Sub-Agent.

    Guarantees (OWASP LLM06 / Excessive Agency Mitigation):
        * Deterministic verdict based on hard industrial thresholds, NOT LLM discretion.
        * No database write capability (pure domain verifier).
        * Bounded compliance_score in [0, 1] with diagnostic flag list.
        * Always returns a verdict — never raises on unexpected state.
    """

    model_config = ConfigDict(extra="allow")

    is_approved: bool = Field(
        description="Whether the proposal passes all compliance gates."
    )
    compliance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite score (0 = non-compliant, 1 = fully compliant).",
    )
    flagged_risks: List[str] = Field(
        default_factory=list,
        description="Human-readable list of detected risk violations.",
    )
    recommended_mitigations: List[str] = Field(
        default_factory=list,
        description="Suggested corrective actions for each flagged risk.",
    )
    checker_verdict: Literal["APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED"] = Field(
        description="Final compliance verdict."
    )
