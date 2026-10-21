"""add dataset file path index

Adds an index on ``dataset_file.path``
so ingest can look up already-ingested files without scanning the table.

Revision ID: f8a9b0c1d2e3
Revises: e7f8a9b0c1d2
Create Date: 2026-10-19 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f8a9b0c1d2e3"
down_revision: str | None = "e7f8a9b0c1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add an index on ``dataset_file.path``."""
    with op.batch_alter_table("dataset_file", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_dataset_file_path"), ["path"], unique=False)


def downgrade() -> None:
    """Drop the index on ``dataset_file.path``."""
    with op.batch_alter_table("dataset_file", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_dataset_file_path"))
