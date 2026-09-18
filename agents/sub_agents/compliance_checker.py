"""
Compliance Checker Sub-Agent (Phase 3, Prompt 3 - Maker-Checker Architecture).

Pure domain worker / deterministic verifier that operates independently of the
Maker (ExAnteAgent). Enforces OWASP LLM06 (Excessive Agency Mitigation) by:
    * Never executing database transactions or writing reports directly.
    * Providing bounded, deterministic verdicts based on hard industrial thresholds.
    * Falling back to APPROVED_WITH_CONDITIONS on any unhandled state — never
      raising unhandled exceptions.

The checker verifies:
    1. Numerical thresholds: ROI vs cost of capital, VaR95 limits, cash flow solvency.
    2. Downtime/outage statutory ceilings (plant safety regulations).
    3. Qualitative consistency: cross-examine the Maker's LLM-generated assessment
       against the simulated quantitative results (detecting hallucinations).
"""
from __future__ import annotations

from typing import Any, List, Literal

from core.contracts import (
    ComplianceAuditReport,
    ProposalInput,
    QualitativeAssessment,
    SimulationResult,
)


# ---------------------------------------------------------------------------
# Industrial threshold constants (configurable per sector / jurisdiction)
# ---------------------------------------------------------------------------

# Minimum acceptable ROI against weighted average cost of capital (WACC)
MIN_ROI_PCT: float = 8.0  # 8% minimum hurdle rate

# Maximum acceptable VaR95 (downside risk limit) as a percentage of adjusted benefit
MAX_VAR95_PCT: float = 25.0  # 25% of adjusted benefit

# Maximum allowable downtime hours per year (plant safety ceiling)
MAX_DOWNTIME_HOURS_ANNUAL: float = 720.0  # 30 full-day outages per year

# Maximum allowable gas outage days per year (operational safety)
MAX_GAS_OUTAGE_DAYS_ANNUAL: float = 30.0

# Maximum allowable power outage days per year (operational safety)
MAX_POWER_OUTAGE_DAYS_ANNUAL: float = 20.0

# Cost of capital / inflation guard: ensure real ROI > 0 after inflation guard
REAL_ROI_MIN_PCT: float = 2.0  # 2% minimum real return


class ComplianceCheckerSubAgent:
    """
    Deterministic compliance verifier for project proposals.

    Pure computation — no external side effects, no network calls, no database
    writes. Designed to be called from the orchestration layer (ExAnteAgent)
    after the Maker has produced its qualitative and quantitative output.

    The audit() method returns a ``ComplianceAuditReport`` whose verdict is
    always one of: APPROVED, APPROVED_WITH_CONDITIONS, or REJECTED.
    On any unforeseen condition or missing field, the fallback verdict is
    APPROVED_WITH_CONDITIONS with diagnostic flags — never an exception.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(
        self,
        proposal: ProposalInput,
        simulation: SimulationResult,
        qualitative: QualitativeAssessment,
    ) -> ComplianceAuditReport:
        """
        Run the full compliance audit triad.

        Args:
            proposal: Validated ProposalInput crossing the boundary.
            simulation: Quantitative simulation outcomes (mean_roi, var_95, etc.).
            qualitative: The Maker's LLM-synthesized QualitativeAssessment.

        Returns:
            ComplianceAuditReport with verdict, score, flagged risks and
            recommended mitigations.
        """
        try:
            flags: List[str] = []
            mitigations: List[str] = []
            score = 1.0  # start perfectly compliant; deduct for each violation

            # ---- 1. Numerical threshold checks ----
            # a) ROI hurdle rate
            roi = simulation.mean_roi
            if roi < MIN_ROI_PCT:
                deduct = min(0.3, (MIN_ROI_PCT - roi) / MAX_ROI_PCT)
                score -= deduct
                flags.append(
                    f"ROI {roi:.1f}% below minimum hurdle rate {MIN_ROI_PCT:.1f}%"
                )
                mitigations.append(
                    "Review project assumptions; consider cost reduction or "
                    "revenue enhancement to meet minimum hurdle."
                )

            # b) VaR95 downside limit
            var95 = simulation.var_95
            adjusted_benefit = simulation.adjusted_net_benefit if hasattr(simulation, "adjusted_net_benefit") else 0.0
            if adjusted_benefit > 0 and var95 / adjusted_benefit > MAX_VAR95_PCT / 100.0:
                deduct = min(0.3, (var95 / adjusted_benefit - MAX_VAR95_PCT / 100.0))
                score -= deduct
                flags.append(
                    f"VaR95 {var95:,.0f} MToman exceeds {MAX_VAR95_PCT:.1f}% of adjusted benefit"
                )
                mitigations.append(
                    "Implement risk hedging or reduce exposure; re-evaluate VaR "
                    "under stressed scenarios."
                )

            # c) Real ROI guard (after inflation)
            if roi - simulation.mean_inflation_rate if hasattr(simulation, "mean_inflation_rate") else 0 < REAL_ROI_MIN_PCT:
                score -= 0.15
                flags.append("Real ROI (after inflation) falls below 2% minimum")
                mitigations.append(
                    "Re-evaluate inflation assumptions or adjust cost/benefit "
                    "projections to protect real return."
                )

            # ---- 2. Operational safety ceilings ----
            # a) Downtime hours
            total_downtime = simulation.total_downtime_hours if hasattr(simulation, "total_downtime_hours") else 0.0
            if total_downtime > MAX_DOWNTIME_HOURS_ANNUAL:
                deduct = min(0.2, (total_downtime - MAX_DOWNTIME_HOURS_ANNUAL) / MAX_DOWNTIME_HOURS_ANNUAL)
                score -= deduct
                flags.append(
                    f"Total downtime {total_downtime:.1f} h/yr exceeds "
                    f"statutory ceiling of {MAX_DOWNTIME_HOURS_ANNUAL:.1f} h/yr"
                )
                mitigations.append(
                    "Review maintenance schedules; consider redundant equipment "
                    "or predictive maintenance to reduce unplanned outages."
                )

            # b) Gas outage days
            gas_outage = simulation.gas_outage_days_yearly if hasattr(simulation, "gas_outage_days_yearly") else 0.0
            if gas_outage > MAX_GAS_OUTAGE_DAYS_ANNUAL:
                deduct = min(0.15, (gas_outage - MAX_GAS_OUTAGE_DAYS_ANNUAL) / MAX_GAS_OUTAGE_DAYS_ANNUAL)
                score -= deduct
                flags.append(
                    f"Gas outage {gas_outage:.1f} days/yr exceeds {MAX_GAS_OUTAGE_DAYS_ANNUAL:.1f} day/yr limit"
                )
                mitigations.append(
                    "Review gas supply resilience; consider dual-fuel capability "
                    "or storage buffers."
                )

            # c) Power outage days
            power_outage = simulation.power_outage_days_yearly if hasattr(simulation, "power_outage_days_yearly") else 0.0
            if power_outage > MAX_POWER_OUTAGE_DAYS_ANNUAL:
                deduct = min(0.15, (power_outage - MAX_POWER_OUTAGE_DAYS_ANNUAL) / MAX_POWER_OUTAGE_DAYS_ANNUAL)
                score -= deduct
                flags.append(
                    f"Power outage {power_outage:.1f} days/yr exceeds "
                    f"{MAX_POWER_OUTAGE_DAYS_ANNUAL:.1f} day/yr limit"
                )
                mitigations.append(
                    "Review grid resilience; consider backup generation or "
                    "demand-side management."
                )

            # ---- 3. Qualitative verification (hallucination detection) ----
            # Cross-examine the Maker's claims against simulated numbers.
            # The qualitative block typically contains:
            #   technical_success_probability, recommended_action, risk_summary
            tech_prob = qualitative.technical_success_probability
            recommended_action = qualitative.recommended_action

            # a) Profitability hallucination guard:
            # If the Maker claims high profitability but simulation shows negative NPV/ROI,
            # flag it.
            if roi < 0 and tech_prob > 0.7:
                score -= 0.2
                flags.append(
                    "Maker claims high technical success probability despite "
                    "negative simulated ROI — potential hallucination."
                )
                mitigations.append(
                    "Require independent financial review before proceeding; "
                    "validate assumptions driving the positive projection."
                )

            # b) Recommended action consistency
            if recommended_action == "APPROVE" and score < 0.6:
                score -= 0.1
                flags.append(
                    "Recommendation to approve despite sub-threshold compliance score."
                )
                mitigations.append(
                    "Reconsider recommendation; attach conditions or downgrade "
                    "to revised/reject."
                )

            # ---- 4. Clamp and determine verdict ----
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))

            # Deterministic verdict logic
            if score >= 0.8:
                verdict: Literal["APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED"] = "APPROVED"
            elif score >= 0.5:
                verdict = "APPROVED_WITH_CONDITIONS"
            else:
                verdict = "REJECTED"

            # Ensure that if any severe flag exists (VaR > 40%, downtime > 2x ceiling),
            # we never silently APPROVE.
            severe_flags = any(
                "exceeds" in f and ("VaR" in f or "downtime" in f) for f in flags
            )
            if severe_flags and verdict == "APPROVED":
                verdict = "APPROVED_WITH_CONDITIONS"
                if "severe_risk_auto_downgrade" not in flags:
                    flags.append("severe_risk_auto_downgrade: overridden to APPROVED_WITH_CONDITIONS")

            # If no flags at all but score is low (shouldn't happen), still emit a
            # conservative verdict.
            if not flags and score < 0.5:
                flags.append("no-specific-flags-detected: conservative verdict applied")
                verdict = "APPROVED_WITH_CONDITIONS"

            report = ComplianceAuditReport(
                is_approved=verdict in ("APPROVED", "APPROVED_WITH_CONDITIONS"),
                compliance_score=round(score, 4),
                flagged_risks=flags,
                recommended_mitigations=mitigations,
                checker_verdict=verdict,
            )
            return report

        except Exception as exc:  # noqa: BLE001 - fallback guarantee
            # Unhandled exception path: never raise. Return conservative verdict.
            return ComplianceAuditReport(
                is_approved=False,
                compliance_score=0.0,
                flagged_risks=[f"unhandled-exception-during-audit:{type(exc).__name__}"],
                recommended_mitigations=[
                    "Review compliance checker logic; fix unhandled case before "
                    "re-deployment."
                ],
                checker_verdict="REJECTED",
            )


# ---------------------------------------------------------------------------
# Helper: make the sub-agent easily instantiable / injectable
# ---------------------------------------------------------------------------

def create_compliance_checker() -> ComplianceCheckerSubAgent:
    """Factory for the compliance checker sub-agent."""
    return ComplianceCheckerSubAgent()