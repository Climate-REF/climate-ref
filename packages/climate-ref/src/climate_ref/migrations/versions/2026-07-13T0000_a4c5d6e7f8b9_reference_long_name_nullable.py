"""make reference dataset long_name nullable

``long_name`` is read from the *variable's* attributes, not the global ones, and obs4MIPs does not
require it. The canonical obs4REF FLUXNET2015 ``gpp`` file omits it, which made ingesting that
dataset fail with a ``NOT NULL`` ``IntegrityError`` and abort the whole obs4MIPs ingest.

Every other dataset table already treats it as optional: ``cmip6_dataset``, ``cmip7_dataset`` and
``esmvaltool_reference_dataset`` all declare it nullable, and the ESMValTool adapter sets it to
``None`` unconditionally. This aligns the three ``ReferenceDatasetMixin`` tables with them.

``long_name`` is descriptive metadata only -- it is not part of the dataset identity and no
``instance_id`` or slug derives from it -- so relaxing it cannot collapse two datasets into one.

SQLite cannot ``ALTER COLUMN``, so the change goes through ``batch_alter_table``, which rebuilds
each table. The downgrade is intentionally lossy-safe: it cannot restore a ``NOT NULL`` constraint
while rows hold ``NULL``, so it backfills those rows with the empty string first.

Revision ID: a4c5d6e7f8b9
Revises: e1f2a3b4c5d6
Create Date: 2026-07-13
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a4c5d6e7f8b9"
down_revision: str | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The three tables built from ReferenceDatasetMixin
_TABLES = ("obs4mips_dataset", "pmp_climatology_dataset", "obs4ref_dataset")


def upgrade() -> None:
    """Relax long_name to nullable on the reference dataset tables."""
    for table in _TABLES:
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column("long_name", existing_type=sa.String(), nullable=True)


def downgrade() -> None:
    """Restore the NOT NULL constraint, backfilling NULLs so the constraint can be applied."""
    for table in _TABLES:
        op.execute(sa.text(f"UPDATE {table} SET long_name = '' WHERE long_name IS NULL"))  # noqa: S608
        with op.batch_alter_table(table) as batch_op:
            batch_op.alter_column("long_name", existing_type=sa.String(), nullable=False)
