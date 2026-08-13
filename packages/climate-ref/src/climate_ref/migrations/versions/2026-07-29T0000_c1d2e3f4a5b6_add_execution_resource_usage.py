"""add execution resource usage columns

Adds nullable columns to ``execution`` recording what a run cost:
wall time, CPU time, peak memory and the limits in force at the time.

Revision ID: c1d2e3f4a5b6
Revises: a4c5d6e7f8b9
Create Date: 2026-07-29 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: str | None = "a4c5d6e7f8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = (
    ("wall_seconds", sa.Float()),
    ("cpu_seconds", sa.Float()),
    ("peak_memory_bytes", sa.BigInteger()),
    ("memory_source", sa.String()),
    ("memory_limit_bytes", sa.BigInteger()),
    ("cpu_limit", sa.Float()),
    ("resources_exclusive", sa.Boolean()),
    ("queue_seconds", sa.Float()),
    ("resource_context", sa.JSON()),
)


def upgrade() -> None:
    """Add the resource usage columns to ``execution``."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        for name, column_type in _COLUMNS:
            batch_op.add_column(sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    """Drop the resource usage columns from ``execution``."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        for name, _ in reversed(_COLUMNS):
            batch_op.drop_column(name)
