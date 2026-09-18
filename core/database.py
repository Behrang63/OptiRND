import datetime
import logging
from contextlib import contextmanager
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, Index, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

logger = logging.getLogger("optirnd.database")


# --------------------------------------------------------------------------- #
# Custom persistence exceptions
# --------------------------------------------------------------------------- #
class DatabaseError(Exception):
    """Base class for all database-layer domain errors."""


class DatabaseTransactionError(DatabaseError):
    """Raised when a transaction fails to commit or rolls back unexpectedly."""


class RecordNotFoundError(DatabaseError):
    """Raised when an expected record does not exist in the database."""


# --------------------------------------------------------------------------- #
# ORM base
# --------------------------------------------------------------------------- #
Base = declarative_base()


def get_utc_now():
    """
    تولید زمان جاری استاندارد بر اساس UTC جهت جلوگیری از خطاهای DeprecationWarning.
    """
    return datetime.datetime.now(datetime.timezone.utc)


class Proposal(Base):
    """
    جدول جامع پروپوزال‌ها:
    ذخیره مشخصات پایه مالی، سه‌گانه‌های PERT، ریسک‌های عملیاتی (انرژی، MTBF، لجستیک)،
    معافیت‌های مالیاتی و شاخص‌های تجمیعی نهایی جهت استفاده در بهینه‌سازی سبد پروژه‌ها.
    """
    __tablename__ = 'proposals'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    organization = Column(String(255), default="فولاد مبارکه")
    created_at = Column(DateTime, default=get_utc_now)

    # ۱. سه‌گانه‌های تخمین هزینه اولیه (میلیون تومان)
    cost_p10 = Column(Float, nullable=False, default=8000.0)
    cost_p50 = Column(Float, nullable=False, default=10000.0)
    cost_p90 = Column(Float, nullable=False, default=14000.0)

    # ۲. سه‌گانه‌های تخمین منافع سالانه پایه (میلیون تومان)
    benefit_p10 = Column(Float, nullable=False, default=3000.0)
    benefit_p50 = Column(Float, nullable=False, default=5000.0)
    benefit_p90 = Column(Float, nullable=False, default=8000.0)

    # ۳. احتمال موفقیت فنی پروژه
    p_success = Column(Float, default=0.85)

    # ۴. شاخص‌های ریسک عملیاتی و توقفات خط
    total_downtime_hours = Column(Float, default=0.0)      # مجموع ساعات توقف خط در سال
    energy_loss_toman = Column(Float, default=0.0)         # خسارت ریالی سالانه ناشی از ناترازی گاز و برق
    downtime_loss_toman = Column(Float, default=0.0)       # خسارت ریالی سالانه توقفات فنی تجهیز (MTBF)
    lead_time_delay_days = Column(Float, default=0.0)      # روزهای تأخیر پیش‌بینی‌شده لجستیک

    # ۵. مشوق‌ها و معافیت‌های مالیاتی
    iran_tax_credit_toman = Column(Float, default=0.0)     # اعتبار مالیاتی حاصل از قانون جهش تولید در ایران
    carbon_savings_toman = Column(Float, default=0.0)      # ارزش معافیت مالیات کربن صادراتی اروپا (CBAM)

    # ۶. شاخص‌های خروجی تجمیعی و تعدیل‌شده
    adjusted_net_benefit = Column(Float, default=0.0)      # منفعت خالص تعدیل‌شده سالانه (با احتساب ریسک‌ها)
    mean_roi = Column(Float, nullable=True)                # میانگین بازگشت سرمایه
    var_95 = Column(Float, nullable=True)                  # شاخص ریسک VaR 95%
    probability_of_loss = Column(Float, nullable=True)     # احتمال زیان کل
    risk_status = Column(String(50), nullable=True)        # وضعیت سطح ریسک

    # رابطه یک‌به‌چند با مایلستون‌ها
    milestones = relationship("Milestone", back_populates="proposal", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_proposals_created_at", "created_at"),
    )


class Milestone(Base):
    """
    جدول مایلستون‌ها: پایش فرآیندی زمان و بودجه در طول فازهای اجرایی پروژه.
    """
    __tablename__ = 'milestones'

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(Integer, ForeignKey('proposals.id'))
    title = Column(String(255), nullable=False)
    
    # بودجه برنامه‌ریزی‌شده و واقعی (میلیون تومان)
    planned_budget = Column(Float, nullable=False)
    actual_budget = Column(Float, default=0.0)
    
    # زمان‌بندی برنامه‌ریزی‌شده و واقعی (بر حسب روز)
    planned_duration_days = Column(Integer, nullable=False)
    actual_duration_days = Column(Integer, default=0)
    
    # درصد پیشرفت فاز
    completion_percentage = Column(Float, default=0.0)

    # رابطه با پروپوزال مربوطه
    proposal = relationship("Proposal", back_populates="milestones")

    __table_args__ = (
        Index("ix_milestones_proposal_id", "proposal_id"),
    )


# --------------------------------------------------------------------------- #
# Engine factory & SQLite pragmas (WAL + busy timeout)
# --------------------------------------------------------------------------- #
def init_db(db_path: str = "sqlite:///pajoheshyar.db") -> sessionmaker:
    """
    راه‌اندازی اتصال دیتابیس SQLite و ساخت خودکار جدول‌های به‌روزشده.

    Concurrency hardening:
      * WAL (Write-Ahead Logging) journal mode - readers do not block the writer
        and vice versa, eliminating most "database is locked" errors.
      * busy_timeout of 5000 ms on every connection plus a 30 s pool-level
        connect timeout.
      * foreign_keys pragma enforced per connection.
    """
    engine = create_engine(
        db_path,
        echo=False,
        connect_args={"timeout": 30},          # pool-level lock timeout (30 s)
    )

    if db_path.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


# --------------------------------------------------------------------------- #
# Transaction lifecycle context manager
# --------------------------------------------------------------------------- #
@contextmanager
def transaction_scope(session_factory: sessionmaker):
    """
    Safe database session/transaction scope.

    Guarantees:
      * session.commit() on clean exit,
      * session.rollback() on ANY exception inside the block,
      * session.close() in finally -> SQLite file locks are always released,
      * SQLAlchemyError is translated into DatabaseTransactionError with the
        original exception chained (full traceback preserved).

    Usage:
        with transaction_scope(SessionLocal) as session:
            session.add(obj)
        # committed here; on exception: rolled back + closed + raised
    """
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Transaction rolled back due to SQLAlchemy error")
        raise DatabaseTransactionError(
            f"Database transaction failed and was rolled back: {exc}"
        ) from exc
    except Exception:
        # Domain/validation errors: still roll back so no partial state lingers.
        session.rollback()
        logger.exception("Transaction rolled back due to unexpected error")
        raise
    finally:
        session.close()


def save_comprehensive_proposal(session_factory: sessionmaker, data: Dict[str, Any]) -> Proposal:
    """
    ذخیره‌سازی یکپارچه پروپوزال به همراه تمامی شاخص‌های تعدیل‌شده ریسک و مالیاتی.

    Transactional API: accepts a session_factory, wraps the insert in
    transaction_scope() (commit / rollback / close guaranteed).
    All values are bound through ORM model parameters - zero raw SQL.
    """
    with transaction_scope(session_factory) as session:
        new_proposal = Proposal(
            title=data.get("title", "پروژه بدون عنوان"),
            cost_p10=float(data.get("cost_p10", 8000.0)),
            cost_p50=float(data.get("cost_p50", 10000.0)),
            cost_p90=float(data.get("cost_p90", 14000.0)),
            benefit_p10=float(data.get("benefit_p10", 3000.0)),
            benefit_p50=float(data.get("benefit_p50", 5000.0)),
            benefit_p90=float(data.get("benefit_p90", 8000.0)),
            p_success=float(data.get("p_success", 0.85)),
            total_downtime_hours=float(data.get("total_downtime_hours", 0.0)),
            energy_loss_toman=float(data.get("energy_loss_toman", 0.0)),
            downtime_loss_toman=float(data.get("downtime_loss_toman", 0.0)),
            lead_time_delay_days=float(data.get("lead_time_delay_days", 0.0)),
            iran_tax_credit_toman=float(data.get("iran_tax_credit_toman", 0.0)),
            carbon_savings_toman=float(data.get("carbon_savings_toman", 0.0)),
            adjusted_net_benefit=float(data.get("adjusted_net_benefit", data.get("benefit_p50", 5000.0))),
            mean_roi=float(data.get("mean_roi", 0.0)),
            var_95=float(data.get("var_95", 0.0)),
            probability_of_loss=float(data.get("probability_of_loss", 0.0)),
            risk_status=data.get("risk_status", "MODERATE_RISK")
        )
        session.add(new_proposal)
    # session.flush() inside the scope assigned the PK; expire_on_commit=False
    # keeps attributes readable after commit.
    return new_proposal


def get_all_proposals(session_factory: sessionmaker) -> List[Proposal]:
    """
    استخراج تمامی پروپوزال‌های ثبت‌شده به ترتیب جدیدترین تاریخ.
    Read-only query executed inside a guaranteed-close scope.
    """
    with transaction_scope(session_factory) as session:
        return session.query(Proposal).order_by(Proposal.created_at.desc()).all()


def get_proposals_for_portfolio(session_factory: sessionmaker) -> List[Dict[str, Any]]:
    """
    استخراج لیست پروژه‌ها با فیلدهای استاندارد جهت بارگذاری مستقیم در جدول چیدمان سبد.
    """
    with transaction_scope(session_factory) as session:
        proposals = session.query(Proposal).order_by(Proposal.created_at.desc()).all()
        formatted_list = []

        for p in proposals:
            formatted_list.append({
                "شناسه": p.id,
                "عنوان پروژه": p.title,
                "هزینه P10": p.cost_p10,
                "هزینه P50": p.cost_p50,
                "هزینه P90": p.cost_p90,
                "منافع P10": round(p.benefit_p10 * (p.adjusted_net_benefit / max(p.benefit_p50, 1.0)), 2),
                "منافع P50": round(p.adjusted_net_benefit, 2),
                "منافع P90": round(p.benefit_p90 * (p.adjusted_net_benefit / max(p.benefit_p50, 1.0)), 2),
                "توقف P50": round(p.total_downtime_hours, 1)
            })
        return formatted_list

def delete_proposal_by_id(session_factory: sessionmaker, proposal_id: int) -> bool:
    """
    حذف یک پروپوزال مشخص از پایگاه داده بر اساس شناسه (parameterized ORM filter).
    Raises RecordNotFoundError when the identifier does not exist.
    """
    with transaction_scope(session_factory) as session:
        proposal = session.query(Proposal).filter(Proposal.id == proposal_id).first()
        if proposal is None:
            raise RecordNotFoundError(f"Proposal with id={proposal_id} not found.")
        session.delete(proposal)
    return True