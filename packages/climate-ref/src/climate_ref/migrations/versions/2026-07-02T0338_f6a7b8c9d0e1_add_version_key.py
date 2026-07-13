"""add version_key to base dataset table

Adds ``dataset.version_key``, a numeric ordering key for each dataset's ``version`` string,
computed by ``climate_ref_core.datasets.version_sort_key``. It powers the SQL-side
latest-version window function (``RANK() OVER (PARTITION BY <dataset_id_metadata> ORDER BY
version_key DESC)``) used by ``select_datasets(..., latest_group_by=...)``.

``version`` itself lives on the four subclass tables (``cmip6_dataset``, ``cmip7_dataset``,
``obs4mips_dataset``, ``pmp_climatology_dataset``), not the base ``dataset`` table, so the
backfill below reads each subclass table's ``version`` column and writes the computed key onto
the corresponding base-table row. Parsing versions in SQL was rejected -- ``version_sort_key``'s
regex/int-cast logic has no portable SQL expression across SQLite and PostgreSQL -- so the
backfill runs in Python.

After this migration, a SQLAlchemy ``before_insert``/``before_update`` mapper event on the
``Dataset`` model keeps ``version_key`` in sync for every future write.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-07-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from climate_ref_core.datasets import version_sort_key

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The four Dataset subclass tables that own a ``version`` column.
_SUBCLASS_TABLES = ("cmip6_dataset", "cmip7_dataset", "obs4mips_dataset", "pmp_climatology_dataset")


def _backfill() -> None:
    """Compute ``version_key`` in Python from each subclass table's ``version`` column.

    ``version_sort_key`` matches a leading ``v`` followed by digits and casts them to an int,
    falling back to ``-1``.
    That regex + int-cast has no expression that compiles portably across SQLite and PostgreSQL,
    so the key is derived in Python and written with one ``UPDATE`` per distinct version value.

    Worst case this is one ``UPDATE`` per row: date-versioned datasets are effectively all-distinct,
    so the distinct-version count approaches the row count,
    but only runs once.
    """
    bind = op.get_bind()
    metadata = sa.MetaData()
    dataset = sa.Table("dataset", metadata, autoload_with=bind)

    for table_name in _SUBCLASS_TABLES:
        sub = sa.Table(table_name, metadata, autoload_with=bind)
        distinct_versions = bind.execute(sa.select(sub.c.version).distinct()).scalars().all()

        for version in distinct_versions:
            key = version_sort_key(version)
            # ``version`` is NOT NULL on every subclass table, so ``==`` is correct and portable.
            # ``col.is_(value)`` would compile to ``version IS 'v10'``,
            # which SQLite tolerates but PostgreSQL rejects (``IS`` only accepts NULL/boolean).
            id_subquery = sa.select(sub.c.id).where(sub.c.version == version)
            bind.execute(dataset.update().where(dataset.c.id.in_(id_subquery)).values(version_key=key))


def upgrade() -> None:
    with op.batch_alter_table("dataset", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("version_key", sa.BigInteger(), nullable=False, server_default=sa.text("-1"))
        )

    _backfill()


def downgrade() -> None:
    with op.batch_alter_table("dataset", schema=None) as batch_op:
        batch_op.drop_column("version_key")
