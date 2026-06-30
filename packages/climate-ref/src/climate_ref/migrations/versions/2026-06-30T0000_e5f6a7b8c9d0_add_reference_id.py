"""add reference_id to series metric values

Adds a nullable content-hash column to ``metric_value`` used to deduplicate
reference (observation) series across executions. ``kind`` is added separately as a
controlled-vocabulary dimension column (handled by the env.py dimension auto-add),
so only ``reference_id`` needs an explicit revision here.

Plain DDL only, so it applies identically on SQLite and PostgreSQL.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-06-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.add_column(sa.Column("reference_id", sa.Text(), nullable=True))
        batch_op.create_index(batch_op.f("ix_metric_value_reference_id"), ["reference_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_metric_value_reference_id"))
        batch_op.drop_column("reference_id")
