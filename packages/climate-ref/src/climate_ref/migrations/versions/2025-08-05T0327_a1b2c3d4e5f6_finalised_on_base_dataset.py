"""finalised-on-base-dataset

Move finalised from cmip6_dataset to base dataset table and default all existing rows to True.

Revision ID: a1b2c3d4e5f6
Revises: 94beace57a9c
Create Date: 2025-08-05 03:27:00

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ba5e"
down_revision: Union[str, None] = "94beace57a9c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add finalised to base dataset with default True, non-null
    with op.batch_alter_table("dataset", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("finalised", sa.Boolean(), nullable=True, server_default=sa.text("true"))
        )

    # Backfill: ensure all existing rows are True
    op.execute("UPDATE dataset SET finalised = TRUE WHERE finalised IS NULL")

    # Enforce NOT NULL after backfill
    with op.batch_alter_table("dataset", schema=None) as batch_op:
        batch_op.alter_column("finalised", nullable=False)

    # Drop column from cmip6_dataset if it exists
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cmip6_cols = {col["name"] for col in inspector.get_columns("cmip6_dataset")}
    if "finalised" in cmip6_cols:
        with op.batch_alter_table("cmip6_dataset", schema=None) as batch_op:
            batch_op.drop_column("finalised")


def downgrade() -> None:
    # Re-create cmip6_dataset.finalised as non-nullable boolean default False
    # Note: Original migration 94beace57a9c added cmip6_dataset.finalised NOT NULL, with no default.
    with op.batch_alter_table("cmip6_dataset", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("finalised", sa.Boolean(), nullable=False, server_default=sa.text("false"))
        )

    # Drop base dataset finalised
    with op.batch_alter_table("dataset", schema=None) as batch_op:
        batch_op.drop_column("finalised")
