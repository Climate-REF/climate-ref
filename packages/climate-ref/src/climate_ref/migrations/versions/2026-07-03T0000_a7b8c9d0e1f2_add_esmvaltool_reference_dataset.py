"""add esmvaltool reference dataset

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-03 00:00:00.000000

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # On PostgreSQL ``dataset_type`` is backed by a native ENUM, so the new value must be
    # added to the type explicitly. SQLite stores the column as a plain VARCHAR (no CHECK
    # constraint is emitted), so no enum change is needed there.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE sourcedatasettype ADD VALUE IF NOT EXISTS 'ESMValToolReference'")

    op.create_table(
        "esmvaltool_reference_dataset",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("variable_id", sa.String(), nullable=False),
        sa.Column("table_id", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("data_type", sa.String(), nullable=True),
        sa.Column("tier", sa.Integer(), nullable=True),
        sa.Column("long_name", sa.String(), nullable=True),
        sa.Column("units", sa.String(), nullable=True),
        sa.Column("instance_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["id"], ["dataset.id"], name=op.f("fk_esmvaltool_reference_dataset_id_dataset")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_esmvaltool_reference_dataset")),
    )
    with op.batch_alter_table("esmvaltool_reference_dataset", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_instance_id"), ["instance_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_project"), ["project"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_source_id"), ["source_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_variable_id"), ["variable_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("esmvaltool_reference_dataset", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_variable_id"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_source_id"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_project"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_instance_id"))

    op.drop_table("esmvaltool_reference_dataset")
    # Note: the ``ESMValToolReference`` value is intentionally left in the PostgreSQL enum type.
    # PostgreSQL does not support removing a value from an enum without recreating the type,
    # and leaving it is harmless.
