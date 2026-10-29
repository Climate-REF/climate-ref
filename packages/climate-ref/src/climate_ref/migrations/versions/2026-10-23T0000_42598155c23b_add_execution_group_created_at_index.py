"""add execution group created at index

Adds a composite index on ``execution (execution_group_id, created_at, id)``
so finding each group's newest execution reads one index entry rather than sorting the group's executions.

Revision ID: 42598155c23b
Revises: b0c1d2e3f4a5
Create Date: 2026-10-23 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "42598155c23b"
down_revision: str | None = "b0c1d2e3f4a5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add an index on ``execution (execution_group_id, created_at, id)``."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.create_index(
            "ix_execution_group_created_at", ["execution_group_id", "created_at", "id"], unique=False
        )


def downgrade() -> None:
    """Drop the index on ``execution (execution_group_id, created_at, id)``."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.drop_index("ix_execution_group_created_at")
