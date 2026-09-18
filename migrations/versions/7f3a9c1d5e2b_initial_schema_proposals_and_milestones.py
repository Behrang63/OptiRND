"""Initial schema: proposals and milestones (Phase 2 baseline)

Revision ID: 7f3a9c1d5e2b
Revises:
Create Date: 2026-09-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "7f3a9c1d5e2b"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "proposals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("organization", sa.String(length=255),
                  server_default="فولاد مبارکه"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        # Financial triplets (million Toman)
        sa.Column("cost_p10", sa.Float(), nullable=False, server_default="8000.0"),
        sa.Column("cost_p50", sa.Float(), nullable=False, server_default="10000.0"),
        sa.Column("cost_p90", sa.Float(), nullable=False, server_default="14000.0"),
        sa.Column("benefit_p10", sa.Float(), nullable=False, server_default="3000.0"),
        sa.Column("benefit_p50", sa.Float(), nullable=False, server_default="5000.0"),
        sa.Column("benefit_p90", sa.Float(), nullable=False, server_default="8000.0"),
        sa.Column("p_success", sa.Float(), server_default="0.85"),
        # Operational risk indices
        sa.Column("total_downtime_hours", sa.Float(), server_default="0.0"),
        sa.Column("energy_loss_toman", sa.Float(), server_default="0.0"),
        sa.Column("downtime_loss_toman", sa.Float(), server_default="0.0"),
        sa.Column("lead_time_delay_days", sa.Float(), server_default="0.0"),
        # Tax incentives
        sa.Column("iran_tax_credit_toman", sa.Float(), server_default="0.0"),
        sa.Column("carbon_savings_toman", sa.Float(), server_default="0.0"),
        # Aggregated outputs
        sa.Column("adjusted_net_benefit", sa.Float(), server_default="0.0"),
        sa.Column("mean_roi", sa.Float(), nullable=True),
        sa.Column("var_95", sa.Float(), nullable=True),
        sa.Column("probability_of_loss", sa.Float(), nullable=True),
        sa.Column("risk_status", sa.String(length=50), nullable=True),
    )
    op.create_index("ix_proposals_created_at", "proposals", ["created_at"])

    op.create_table(
        "milestones",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("proposal_id", sa.Integer(),
                  sa.ForeignKey("proposals.id"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("planned_budget", sa.Float(), nullable=False),
        sa.Column("actual_budget", sa.Float(), server_default="0.0"),
        sa.Column("planned_duration_days", sa.Integer(), nullable=False),
        sa.Column("actual_duration_days", sa.Integer(), server_default="0"),
        sa.Column("completion_percentage", sa.Float(), server_default="0.0"),
    )
    op.create_index("ix_milestones_proposal_id", "milestones", ["proposal_id"])


def downgrade() -> None:
    op.drop_index("ix_milestones_proposal_id", table_name="milestones")
    op.drop_table("milestones")
    op.drop_index("ix_proposals_created_at", table_name="proposals")
    op.drop_table("proposals")
