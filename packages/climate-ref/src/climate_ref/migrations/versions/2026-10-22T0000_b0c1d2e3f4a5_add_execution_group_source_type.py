"""add execution group source type

Adds ``execution_group_source_type``, one row per source dataset type an execution group draws on.
Existing groups are backfilled from the keys of ``execution_group.selectors``.

Revision ID: b0c1d2e3f4a5
Revises: a9b0c1d2e3f4
Create Date: 2026-10-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from climate_ref_core.source_types import SourceDatasetType

# revision identifiers, used by Alembic.
revision: str = "b0c1d2e3f4a5"
down_revision: str | None = "a9b0c1d2e3f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``execution_group_source_type`` and backfill it from ``selectors``."""
    link = op.create_table(
        "execution_group_source_type",
        sa.Column("execution_group_id", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_group_id"],
            ["execution_group.id"],
            name=op.f("fk_execution_group_source_type_execution_group_id_execution_group"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "execution_group_id", "source_type", name=op.f("pk_execution_group_source_type")
        ),
    )

    execution_group = sa.table(
        "execution_group", sa.column("id", sa.Integer), sa.column("selectors", sa.JSON)
    )
    rows = op.get_bind().execute(sa.select(execution_group.c.id, execution_group.c.selectors)).all()
    # Stored by enum name, matching ``dataset.dataset_type``.
    links = [
        {"execution_group_id": group_id, "source_type": SourceDatasetType(key).name}
        for group_id, selectors in rows
        for key in selectors or {}
    ]
    if links:
        op.bulk_insert(link, links)


def downgrade() -> None:
    """Drop ``execution_group_source_type``."""
    op.drop_table("execution_group_source_type")
