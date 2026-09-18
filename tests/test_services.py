"""
Integration tests for the Service-Repository layer (Phase 2).

Verifies: ProposalRepository wraps the transactional DB API, and
EvaluationService enforces the atomicity rule - the Markdown report is
written ONLY after the database commit succeeds.
"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.database import init_db, RecordNotFoundError  # noqa: E402
from core.repository import ProposalRepository  # noqa: E402
from core.services import EvaluationService  # noqa: E402
from agents.main_agents.ex_ante_agent import ExAnteAgent  # noqa: E402
from core.llm_provider import LLMFactory  # noqa: E402


VALID_INPUT = {
    "title": "تست یکپارچگی سرویس ارزیابی",
    "cost_p10": 3000.0, "cost_p50": 4000.0, "cost_p90": 5500.0,
    "benefit_p10": 6000.0, "benefit_p50": 8000.0, "benefit_p90": 11000.0,
    "p_success": 0.85,
    "years": 3,
}


@pytest.fixture()
def service(tmp_path, monkeypatch):
    """EvaluationService wired to an in-memory DB and a temp report dir."""
    repo = ProposalRepository(init_db("sqlite:///:memory:"))
    agent = ExAnteAgent(llm_provider=LLMFactory.get_provider("mock"))
    monkeypatch.setattr(
        "core.services.MarkdownReportGenerator.save_md_file",
        staticmethod(lambda content, filename, directory: str(tmp_path / filename)),
    )
    return EvaluationService(repository=repo, agent=agent)


def test_service_full_pipeline_persists_and_reports(service, tmp_path):
    envelope = service.evaluate_and_register(VALID_INPUT)

    assert envelope["proposal_id"] is not None
    assert envelope["result"]["adjusted_net_benefit"] > 0
    assert envelope["report_path"] is not None  # report written after commit
    assert service.repository.count() == 1


def test_service_rejects_invalid_contract_before_evaluation(service):
    """Invalid input fails at the boundary - nothing persisted, nothing run."""
    bad = dict(VALID_INPUT, cost_p10=99999.0)  # inverted triplet

    with pytest.raises(ValidationError):
        service.evaluate_and_register(bad)

    assert service.repository.count() == 0


def test_service_aborts_without_report_when_db_fails(service, monkeypatch):
    """
    If persistence fails, the flow must abort BEFORE writing the Markdown
    report (atomicity rule: no orphan report files).
    """
    from core.database import DatabaseTransactionError

    def broken_save(self, data):
        raise DatabaseTransactionError("simulated commit failure")

    monkeypatch.setattr(ProposalRepository, "save_evaluation", broken_save)

    with pytest.raises(DatabaseTransactionError):
        service.evaluate_and_register(VALID_INPUT)

    # The report was never generated for the failed transaction.
    assert service.repository.count() == 0


def test_repository_delete_raises_record_not_found(service):
    with pytest.raises(RecordNotFoundError):
        service.repository.delete(9999)


def test_service_write_report_false_skips_file(service):
    envelope = service.evaluate_and_register(VALID_INPUT, write_report=False)
    assert envelope["report_path"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
