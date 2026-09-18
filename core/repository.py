"""
Repository layer (Phase 2 - Service-Repository pattern).

ProposalRepository is the ONLY module that talks to the persistence layer.
UI views and services depend on this interface, not on SQLAlchemy sessions -
decoupling presentation from storage and enabling future swaps (PostgreSQL,
API-backed repository) without touching callers.
"""
from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import sessionmaker

from core.database import (
    DatabaseTransactionError,
    Proposal,
    RecordNotFoundError,
    delete_proposal_by_id,
    get_all_proposals,
    get_proposals_for_portfolio,
    save_comprehensive_proposal,
    transaction_scope,
)


class ProposalRepository:
    """Transactional data-access facade over the Proposal aggregate."""

    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    # ---- Commands (write path) ------------------------------------------ #
    def save_evaluation(self, evaluation_result: Dict[str, Any]) -> Proposal:
        """Persist a comprehensive evaluation result atomically."""
        return save_comprehensive_proposal(self.session_factory, evaluation_result)

    def delete(self, proposal_id: int) -> None:
        """Delete a proposal; raises RecordNotFoundError when absent."""
        delete_proposal_by_id(self.session_factory, int(proposal_id))

    # ---- Queries (read path) -------------------------------------------- #
    def list_all(self) -> List[Proposal]:
        return get_all_proposals(self.session_factory)

    def list_for_portfolio(self) -> List[Dict[str, Any]]:
        return get_proposals_for_portfolio(self.session_factory)

    def count(self) -> int:
        with transaction_scope(self.session_factory) as session:
            return session.query(Proposal).count()

    # Re-exported for callers that need the exception vocabulary without
    # importing the storage module directly.
    DatabaseTransactionError = DatabaseTransactionError
    RecordNotFoundError = RecordNotFoundError
