"""add execution failure reason

Adds a nullable ``failure_reason`` column to ``execution``
so a failed execution records why it failed.
Failures recorded before this column existed are set to ``unknown``.

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-10-16 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7f8a9b0c1d2"
down_revision: str | None = "d6e7f8a9b0c1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add ``failure_reason`` to ``execution``."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.add_column(sa.Column("failure_reason", sa.String(length=32), nullable=True))

    execution = sa.table("execution", sa.column("successful", sa.Boolean), sa.column("failure_reason"))
    op.execute(
        execution.update().where(execution.c.successful == sa.false()).values(failure_reason="unknown")
    )


def downgrade() -> None:
    """Drop ``failure_reason`` from ``execution``."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.drop_column("failure_reason")
