"""normalise series index into a shared index_axis table

Series metric values previously stored their full index array (and index name)
inline on every row.

In practice a small number of axes are shared by tens of thousands of series (e.g. one monthly time axis),
so the index dominated the database (~40% of a 922MB production database, ~100% redundant).

This migration moves the index into a deduplicated ``index_axis`` table keyed by
a content hash, and replaces ``metric_value.index`` / ``index_name`` with an
``index_id`` foreign key. It is lossless.

Revision ID: d4e5f6a7b8c9
Revises: b1c2d3e4f5a6
Create Date: 2026-06-19
"""

import hashlib
import json
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import quoted_name

# ``index`` is not in SQLAlchemy's dialect reserved-word tables, so identifiers are
# emitted unquoted by default. PostgreSQL then rejects ``... COLUMN index``; forcing the
# quote keeps the DDL valid on both PostgreSQL and SQLite.
_INDEX_COL = quoted_name("index", quote=True)

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "b1c2d3e4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Number of rows updated per executemany batch during the backfill.
_BATCH_SIZE = 5000


def _hash(name, values) -> str:
    # Must match SeriesIndex.compute_hash in the model layer.
    payload = json.dumps([name, list(values)], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _as_list(idx):
    # Raw SELECT of a JSON column returns text on SQLite and a parsed object on
    # PostgreSQL; normalise to a Python list either way.
    if idx is None:
        return None
    return idx if isinstance(idx, (list, dict)) else json.loads(idx)


def _backfill() -> None:
    bind = op.get_bind()
    index_axis = sa.Table("index_axis", sa.MetaData(), autoload_with=bind)

    # Single streaming pass over the series rows: dedupe axes and record each
    # row's hash. Only the (few) distinct axes are held in memory, not every row.
    result = bind.execution_options(stream_results=True).execute(
        sa.text('SELECT id, index_name, "index" FROM metric_value WHERE type = :t AND "index" IS NOT NULL'),
        {"t": "SERIES"},
    )
    axis_payload: dict[str, dict] = {}  # hash -> row to insert
    row_hashes: list[tuple[int, str]] = []  # (metric_value.id, hash)
    for mv_id, name, idx in result:
        values = _as_list(idx)
        digest = _hash(name, values)
        row_hashes.append((mv_id, digest))
        if digest not in axis_payload:
            axis_payload[digest] = {"hash": digest, "name": name, "values": values, "length": len(values)}

    if axis_payload:
        bind.execute(index_axis.insert(), list(axis_payload.values()))

    hash_to_id = {h: i for i, h in bind.execute(sa.text("SELECT id, hash FROM index_axis"))}

    upd = sa.text("UPDATE metric_value SET index_id = :aid WHERE id = :mid")
    batch: list[dict] = []
    for mv_id, digest in row_hashes:
        batch.append({"aid": hash_to_id[digest], "mid": mv_id})
        if len(batch) >= _BATCH_SIZE:
            bind.execute(upd, batch)
            batch = []
    if batch:
        bind.execute(upd, batch)


def upgrade() -> None:
    op.create_table(
        "index_axis",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("hash", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("values", sa.JSON(), nullable=False),
        sa.Column("length", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_index_axis")),
    )
    op.create_index(op.f("ix_index_axis_hash"), "index_axis", ["hash"], unique=True)

    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.add_column(sa.Column("index_id", sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f("ix_metric_value_index_id"), ["index_id"], unique=False)
        batch_op.create_foreign_key(
            batch_op.f("fk_metric_value_index_id_index_axis"), "index_axis", ["index_id"], ["id"]
        )

    _backfill()

    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.drop_column(_INDEX_COL)
        batch_op.drop_column("index_name")


def downgrade() -> None:
    # Best-effort: recreate the inline columns and copy the axis content back.
    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.add_column(sa.Column(_INDEX_COL, sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("index_name", sa.String(), nullable=True))

    op.get_bind().execute(
        sa.text(
            "UPDATE metric_value SET "
            '"index" = (SELECT a."values" FROM index_axis a WHERE a.id = metric_value.index_id), '
            "index_name = (SELECT a.name FROM index_axis a WHERE a.id = metric_value.index_id) "
            "WHERE index_id IS NOT NULL"
        )
    )

    with op.batch_alter_table("metric_value", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_metric_value_index_id_index_axis"), type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_metric_value_index_id"))
        batch_op.drop_column("index_id")

    op.drop_index(op.f("ix_index_axis_hash"), table_name="index_axis")
    op.drop_table("index_axis")
