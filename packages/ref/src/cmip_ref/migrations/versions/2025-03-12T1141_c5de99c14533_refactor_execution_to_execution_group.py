"""refactor_execution_to_execution_group

Revision ID: c5de99c14533
Revises: 1f5969a92b85
Create Date: 2025-03-12 11:41:44.543184

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c5de99c14533"
down_revision: Union[str, None] = "1f5969a92b85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # we changed so much in the metric_execution_group table (and renamed it) that it
    # makes more sense to re-create and drop the old. Maybe alembic can handle this
    # completely also using alter_table, but it autogenerates code for dropping and
    # creating, so it is easier to use that.
    op.create_table(
        "metric_execution_group",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("metric_id", sa.Integer(), nullable=False),
        sa.Column("dataset_key", sa.String(), nullable=False),
        sa.Column("dirty", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(
            ["metric_id"],
            ["metric.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("metric_id", "dataset_key", name="metric_execution_group_ident"),
    )
    with op.batch_alter_table("metric_execution_group", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_metric_execution_group_dataset_key"), ["dataset_key"], unique=False
        )

    op.execute(
        "INSERT INTO metric_execution_group (id, metric_id, dataset_key, dirty, created_at, updated_at) "
        'SELECT id, metric_id, "key", dirty, created_at, updated_at FROM metric_execution'
    )

    # now update the metric_execution_result table for the new foreign key constraint and other updates
    with op.batch_alter_table(
        "metric_execution_result",
        schema=None,
        naming_convention={"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"},
    ) as batch_op:
        batch_op.add_column(sa.Column("retracted", sa.Boolean(), nullable=False, default=False))
        batch_op.drop_constraint(
            "fk_metric_execution_result_metric_execution_id_metric_execution", type_="foreignkey"
        )
        batch_op.alter_column("metric_execution_id", new_column_name="metric_execution_group_id")
        batch_op.create_foreign_key(
            "fk_metric_execution_result_metric_execution_group_id_metric_execution_group",
            "metric_execution_group",
            ["metric_execution_group_id"],
            ["id"],
        )

    with op.batch_alter_table("metric_execution", schema=None) as batch_op:
        batch_op.drop_index("ix_metric_execution_key")
    op.drop_table("metric_execution")


def downgrade() -> None:
    op.create_table(
        "metric_execution",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("retracted", sa.BOOLEAN(), nullable=False),
        sa.Column("metric_id", sa.INTEGER(), nullable=False),
        sa.Column("key", sa.VARCHAR(), nullable=False),
        sa.Column("dirty", sa.BOOLEAN(), nullable=False),
        sa.Column("created_at", sa.DATETIME(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DATETIME(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(
            ["metric_id"],
            ["metric.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("metric_id", "key", name="metric_execution_ident"),
    )
    with op.batch_alter_table("metric_execution", schema=None) as batch_op:
        batch_op.create_index("ix_metric_execution_key", ["key"], unique=False)

    op.execute(
        'INSERT INTO metric_execution (id, retracted, metric_id, "key", dirty, created_at, updated_at) '
        "SELECT id, false, metric_id, dataset_key, dirty, created_at, updated_at FROM metric_execution_group"
    )

    with op.batch_alter_table(
        "metric_execution_result",
        schema=None,
        naming_convention={"fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"},
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_metric_execution_result_metric_execution_group_id_metric_execution_group", type_="foreignkey"
        )
        batch_op.alter_column("metric_execution_group_id", new_column_name="metric_execution_id")
        batch_op.create_foreign_key(
            "fk_metric_execution_result_metric_execution_id_metric_execution",
            "metric_execution",
            ["metric_execution_id"],
            ["id"],
        )
        batch_op.drop_column("retracted")

    with op.batch_alter_table("metric_execution_group", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_metric_execution_group_dataset_key"))
    op.drop_table("metric_execution_group")
