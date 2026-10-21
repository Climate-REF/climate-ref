"""add metric value execution id and type index

Adds a composite index on ``metric_value (execution_id, type)``
so checking whether an execution has scalar or series values reads only the index,
not the series rows and their large JSON payloads.

Revision ID: a9b0c1d2e3f4
Revises: f8a9b0c1d2e3
Create Date: 2026-10-21 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a9b0c1d2e3f4"
down_revision: str | None = "f8a9b0c1d2e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add an index on ``metric_value (execution_id, type)``."""
    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.create_index("ix_metric_value_execution_id_type", ["execution_id", "type"], unique=False)


def downgrade() -> None:
    """Drop the index on ``metric_value (execution_id, type)``."""
    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.drop_index("ix_metric_value_execution_id_type")
