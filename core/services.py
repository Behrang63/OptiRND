"""
Service layer (Phase 2 - Service-Repository pattern).

EvaluationService owns the complete ex-ante business flow:
validate -> evaluate (agent) -> persist (repository) -> report (markdown).

Atomicity rule: the Markdown report is generated ONLY after the database
transaction commits - eliminating the orphan-report-file failure mode that
the Phase 1 audit identified. Views orchestrate nothing; they render results.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from agents.main_agents.ex_ante_agent import ExAnteAgent
from core.contracts import EvaluationResult, ProposalInput, TaskResult, TaskStatus
from core.database import DatabaseTransactionError
from core.md_reporter import MarkdownReportGenerator
from core.repository import ProposalRepository


class EvaluationService:
    """Application service coordinating the full proposal evaluation flow."""

    def __init__(self, repository: ProposalRepository, agent: ExAnteAgent):
        self.repository = repository
        self.agent = agent

    def evaluate_and_register(
        self,
        proposal_input: Dict[str, Any],
        raw_text: Optional[str] = None,
        write_report: bool = True,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline and return a result envelope.

        Progress callback interface (opt-in, None-safe):
            callback(progress: float, stage: str)
        
        Stage progression:
            - "SIMULATION"   : progress 0.0 -> 0.3  (Monte Carlo + risk engines)
            - "LLM_SYNTHESIS": progress 0.3 -> 0.6  (LLM qualitative assessment)
            - "COMPLIANCE_AUDIT": progress 0.6 -> 0.8  (Compliance checks, tax calc)
            - "PERSISTENCE"  : progress 0.8 -> 1.0  (DB save + report generation)

        Raises:
            pydantic.ValidationError: on invalid input contract.
            DatabaseTransactionError: when persistence fails (no report is
                written in that case - the flow fails atomically).
        """
        # 1. Contract enforcement at the service boundary.
        validated = ProposalInput.model_validate(proposal_input)
        payload = validated.model_dump()

        # 2. Domain evaluation (agent fans out to the five risk engines),
        #    with progress updates at defined stages.
        if progress_callback is not None:
            progress_callback(0.0, "SIMULATION")

        # --- Simulation stage (0.0 -> 0.3) ---
        # We run the heavy Monte Carlo + risk simulations first.
        result = self.agent.evaluate_comprehensive_proposal(payload, raw_text=raw_text)
        EvaluationResult.model_validate(result)  # output contract check

        if progress_callback is not None:
            progress_callback(0.3, "LLM_SYNTHESIS")

        # --- LLM Synthesis stage (0.3 -> 0.6) ---
        # The qualitative assessment is already embedded in evaluate_comprehensive_proposal
        # via the _qualitative_assessment call. We just ensure the progress moves forward.

        if progress_callback is not None:
            progress_callback(0.6, "COMPLIANCE_AUDIT")

        # --- Compliance Audit stage (0.6 -> 0.8) ---
        # At this point the evaluation result contains all computed metrics.
        # We could add explicit compliance gate checks here if desired,
        # but the current flow integrates compliance logic within the agent.
        # Progress advances to reflect that Monte Carlo + LLM are done.

        if progress_callback is not None:
            progress_callback(0.8, "PERSISTENCE")

        # 3. Persistence - transactional; failure aborts everything downstream.
        saved = self.repository.save_evaluation(result)

        # 4. Reporting - only after a successful commit.
        report_path: Optional[str] = None
        if write_report:
            safe_title = "".join(
                c for c in validated.title if c.isalnum() or c in (" ", "_", "-")
            ).strip().replace(" ", "_")
            md_text = MarkdownReportGenerator.generate_project_md(payload, result)
            report_path = MarkdownReportGenerator.save_md_file(
                md_text, f"Report_{safe_title}.md", directory="data/reports"
            )

        # Ensure progress reaches 1.0 on completion.
        if progress_callback is not None:
            progress_callback(1.0, "PERSISTENCE & REPORTING")

        return {
            "proposal_id": saved.id,
            "result": result,
            "report_path": report_path,
        }

    # ---- Read-through convenience API ----------------------------------- #
    def list_proposals(self):
        return self.repository.list_all()

    def portfolio_rows(self):
        return self.repository.list_for_portfolio()

    def delete_proposal(self, proposal_id: int) -> None:
        self.repository.delete(proposal_id)
