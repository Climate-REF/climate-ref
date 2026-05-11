"""add diagnostic versioning columns

Adds the read-path schema for diagnostic versioning:

- ``execution_group.diagnostic_version`` -- the diagnostic version that produced this group.
- ``diagnostic.promoted_version`` -- the currently-promoted version for default queries.
- ``execution.provider_version`` -- snapshot of the provider version at run time.

Also flips ``execution_group``'s unique constraint from ``(diagnostic_id, key)`` to
``(diagnostic_id, key, diagnostic_version)`` so that v1 and v2 groups for the same
key can coexist.

Existing rows backfill to ``diagnostic_version = 1`` and ``promoted_version = 1``;
``provider_version`` stays NULL for executions that predate the column.

Revision ID: b1c2d3e4f5a6
Revises: a3b4c5d6e7f8
Create Date: 2026-05-11 00:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, None] = "a3b4c5d6e7f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply diagnostic-versioning schema changes."""
    # execution_group.diagnostic_version + swap unique constraint
    # The column defaults to 1 so existing rows backfill cleanly,
    # and the new unique constraint allows v1/v2 coexistence for the same key.
    with op.batch_alter_table("execution_group", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "diagnostic_version",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("1"),
            )
        )
        batch_op.drop_constraint("execution_ident", type_="unique")
        batch_op.create_unique_constraint(
            "execution_ident",
            ["diagnostic_id", "key", "diagnostic_version"],
        )

    # diagnostic.promoted_version -- naive max(diagnostic_version) cache.
    with op.batch_alter_table("diagnostic", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "promoted_version",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("1"),
            )
        )

    # execution.provider_version -- nullable snapshot recorded by the worker at run time.
    # Rows that predate this column stay NULL; there is no backfill source.
    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.add_column(sa.Column("provider_version", sa.String(), nullable=True))


def downgrade() -> None:
    """Revert diagnostic-versioning schema changes."""
    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.drop_column("provider_version")

    with op.batch_alter_table("diagnostic", schema=None) as batch_op:
        batch_op.drop_column("promoted_version")

    with op.batch_alter_table("execution_group", schema=None) as batch_op:
        batch_op.drop_constraint("execution_ident", type_="unique")
        batch_op.create_unique_constraint(
            "execution_ident",
            ["diagnostic_id", "key"],
        )
        batch_op.drop_column("diagnostic_version")
