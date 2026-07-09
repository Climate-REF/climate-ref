"""add dataset.retracted_at

Adds a nullable ``retracted_at`` timestamp to the base ``dataset`` table: retraction (the term
ESGF uses for pulling a published dataset from recommended use) for datasets that can never be
hard-deleted once they have run (``execution_datasets.dataset_id`` has no ``ondelete="CASCADE"``,
so removing a row that has an execution link either violates the foreign key or destroys execution
provenance). Non-null means retracted.

This is plain DDL -- one nullable column, no default beyond ``NULL`` -- and needs no dialect
branching: SQLite and PostgreSQL both handle ``ADD COLUMN ... NULL`` identically here.

Revision ID: b7c8d9e0a1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c8d9e0a1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("dataset", sa.Column("retracted_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("dataset", "retracted_at")
