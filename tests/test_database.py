"""
Transactional persistence tests for core/database.py.

Covers: CRUD through transaction_scope(), parameterized ORM queries,
rollback-on-failure (no partial state), custom exceptions, WAL pragma
activation, and lock release after failures.
"""
import sys
import sqlite3
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.database import (  # noqa: E402
    Proposal,
    DatabaseTransactionError,
    RecordNotFoundError,
    get_all_proposals,
    get_proposals_for_portfolio,
    delete_proposal_by_id,
    init_db,
    save_comprehensive_proposal,
    transaction_scope,
)


SAMPLE = {
    "title": "سامانه بهینه‌سازی مصرف نسوز",
    "cost_p10": 2000.0, "cost_p50": 2500.0, "cost_p90": 3500.0,
    "benefit_p10": 4000.0, "benefit_p50": 5500.0, "benefit_p90": 7000.0,
    "p_success": 0.90,
    "total_downtime_hours": 18.5,
    "energy_loss_toman": 250.0,
    "downtime_loss_toman": 180.0,
    "iran_tax_credit_toman": 400.0,
    "carbon_savings_toman": 300.0,
    "adjusted_net_benefit": 5770.0,  # 5500 - 250 - 180 + 400 + 300
    "mean_roi": 130.8,
    "var_95": 25.4,
    "risk_status": "LOW_RISK",
}


@pytest.fixture()
def db():
    """Fresh in-memory SQLAlchemy session factory per test."""
    return init_db("sqlite:///:memory:")


# --------------------------------------------------------------------------- #
# CRUD via transactional API
# --------------------------------------------------------------------------- #
def test_save_comprehensive_proposal_commits(db):
    saved = save_comprehensive_proposal(db, SAMPLE)
    assert saved.id is not None
    assert saved.total_downtime_hours == 18.5
    assert saved.adjusted_net_benefit == 5770.0

    rows = get_all_proposals(db)
    assert len(rows) == 1  # commit really persisted


def test_portfolio_projection(db):
    save_comprehensive_proposal(db, SAMPLE)
    portfolio_rows = get_proposals_for_portfolio(db)
    assert len(portfolio_rows) == 1
    assert portfolio_rows[0]["عنوان پروژه"] == "سامانه بهینه‌سازی مصرف نسوز"
    assert portfolio_rows[0]["منافع P50"] == 5770.0


def test_delete_by_id(db):
    saved = save_comprehensive_proposal(db, SAMPLE)
    assert delete_proposal_by_id(db, saved.id) is True
    assert get_all_proposals(db) == []


def test_delete_missing_record_raises_record_not_found(db):
    with pytest.raises(RecordNotFoundError):
        delete_proposal_by_id(db, 9999)


# --------------------------------------------------------------------------- #
# Rollback / atomicity guarantees
# --------------------------------------------------------------------------- #
def test_rollback_on_exception_leaves_no_partial_state(db):
    """
    A domain exception raised inside the scope must roll back the pending
    insert completely - the table stays empty.
    """
    with pytest.raises(ValueError):
        with transaction_scope(db) as session:
            session.add(Proposal(title="will-rollback", cost_p10=1.0, cost_p50=1.0,
                                 cost_p90=1.0, benefit_p10=1.0, benefit_p50=1.0,
                                 benefit_p90=1.0))
            raise ValueError("simulated domain failure mid-transaction")

    assert get_all_proposals(db) == []


def test_sqlalchemy_error_translated_to_domain_exception(db, monkeypatch):
    """SQLAlchemy failures inside the scope become DatabaseTransactionError."""
    from sqlalchemy.exc import OperationalError

    def broken_commit(self):
        raise OperationalError("COMMIT", None, RuntimeError("simulated db crash"))

    with pytest.raises(DatabaseTransactionError) as excinfo:
        with transaction_scope(db) as session:
            session.add(Proposal(title="x", cost_p10=1.0, cost_p50=1.0, cost_p90=1.0,
                                 benefit_p10=1.0, benefit_p50=1.0, benefit_p90=1.0))
            monkeypatch.setattr(type(session), "commit", broken_commit, raising=True)
    assert "rolled back" in str(excinfo.value)


def test_session_closed_after_exception_releases_locks(tmp_path):
    """
    After a failed transaction the session must be closed: a fresh connection
    can immediately acquire the write lock (no lingering SQLite lock).
    """
    db = init_db(f"sqlite:///{tmp_path / 'lock_test.db'}")

    with pytest.raises(ValueError):
        with transaction_scope(db) as session:
            session.add(Proposal(title="locked", cost_p10=1.0, cost_p50=1.0, cost_p90=1.0,
                                 benefit_p10=1.0, benefit_p50=1.0, benefit_p90=1.0))
            raise ValueError("boom")

    # The pool was disposed by nothing, but sessions are closed; a new write
    # must succeed without 'database is locked'.
    saved = save_comprehensive_proposal(db, SAMPLE)
    assert saved.id is not None
    assert len(get_all_proposals(db)) == 1


def test_save_via_helper_rolls_back_on_bad_payload(tmp_path, monkeypatch):
    """
    If float() conversion fails inside save_comprehensive_proposal, the
    transaction is rolled back and the database stays clean.
    """
    db = init_db(f"sqlite:///{tmp_path / 'rollback_helper.db'}")
    bad = dict(SAMPLE, mean_roi="not-a-number")

    with pytest.raises(Exception):  # noqa: B017 - ValueError/TypeError both acceptable
        save_comprehensive_proposal(db, bad)

    assert get_all_proposals(db) == []


# --------------------------------------------------------------------------- #
# SQLite pragmas (WAL + timeout) - file-backed DB required
# --------------------------------------------------------------------------- #
def test_wal_mode_and_busy_timeout_applied(tmp_path):
    """
    WAL is persistent in the DB file (visible to ANY connection);
    busy_timeout / foreign_keys are per-connection, so they must be
    asserted on a connection created by the engine's pool.
    """
    db_path = tmp_path / "wal_test.db"
    SessionLocal = init_db(f"sqlite:///{db_path}")

    # 1. WAL persists in the file - check via an independent raw connection.
    raw = sqlite3.connect(str(db_path))
    try:
        journal_mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        raw.close()
    assert journal_mode.lower() == "wal"

    # 2. Per-connection pragmas on an engine-managed connection.
    engine_conn = SessionLocal.kw["bind"].raw_connection()
    try:
        busy_timeout = engine_conn.execute("PRAGMA busy_timeout").fetchone()[0]
        foreign_keys = engine_conn.execute("PRAGMA foreign_keys").fetchone()[0]
    finally:
        engine_conn.close()
    assert busy_timeout == 5000
    assert foreign_keys == 1


def test_concurrent_write_and_read_no_lock_error(tmp_path):
    """A held read cursor must not block a writer (WAL) within the busy timeout."""
    db = init_db(f"sqlite:///{tmp_path / 'concurrency.db'}")
    save_comprehensive_proposal(db, SAMPLE)

    reader = sqlite3.connect(str(tmp_path / "concurrency.db"), timeout=30)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM proposals").fetchone()

        # Second writer through SQLAlchemy while the raw read txn is open:
        save_comprehensive_proposal(db, dict(SAMPLE, title="پروژه دوم"))

        reader.execute("SELECT count(*) FROM proposals").fetchone()
        reader.rollback()
    finally:
        reader.close()

    assert len(get_all_proposals(db)) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
