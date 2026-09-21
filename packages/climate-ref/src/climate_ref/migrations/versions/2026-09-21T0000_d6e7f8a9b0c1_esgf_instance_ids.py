"""rewrite stored instance_ids to the identifiers ESGF publishes

The adapters now build ``instance_id``s that match the identifiers ESGF publishes:

- CMIP6 ids name only the primary activity of a dataset that belongs to several,
  so ``CMIP6.C4MIP CDRMIP.MIROC.[...]`` becomes ``CMIP6.C4MIP.MIROC.[...]``.
- obs4MIPs-style ids (obs4MIPs, obs4REF, PMP climatologies) drop the duplicated
  collection prefix and the ``nominal_resolution`` segment,
  so ``obs4MIPs.obs4MIPs.NASA-LaRC.CERES-EBAF-4-2-1.mon.rsutcs.100km.gn.v20260220``
  becomes ``obs4MIPs.NASA-LaRC.CERES-EBAF-4-2-1.mon.rsutcs.gn.v20260220``.

Without this rewrite an existing database keeps its old identifiers,
and the next ingestion re-registers every dataset under a new id alongside the old row.
The ids are rebuilt from the metadata columns rather than by string surgery,
so the rewrite is idempotent and independent of the old format.

Executions store a hash over the ids of the datasets they consumed.
Those hashes are not rewritten, so affected execution groups re-run on the next solve.

Revision ID: d6e7f8a9b0c1
Revises: c1d2e3f4a5b6
Create Date: 2026-09-21
"""

from collections.abc import Callable, Sequence

import sqlalchemy as sa
from alembic import op
from loguru import logger

revision: str = "d6e7f8a9b0c1"
down_revision: str | None = "c1d2e3f4a5b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables whose ids are built by the obs4MIPs adapter family.
_REFERENCE_TABLES = ("obs4mips_dataset", "obs4ref_dataset", "pmp_climatology_dataset")

_CMIP6_COLUMNS = (
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "member_id",
    "table_id",
    "variable_id",
    "grid_label",
    "version",
)

_REFERENCE_COLUMNS = (
    "activity_id",
    "institution_id",
    "source_id",
    "frequency",
    "variable_id",
    "nominal_resolution",
    "grid_label",
    "version",
)


def _join(parts: list[str | None]) -> str | None:
    if any(part is None for part in parts):
        return None
    return ".".join(parts)  # type: ignore[arg-type]


def _cmip6_new(row: sa.Row) -> str | None:
    activity = row.activity_id.split(" ", 1)[0] if row.activity_id else None
    return _join(
        [
            "CMIP6",
            activity,
            row.institution_id,
            row.source_id,
            row.experiment_id,
            row.member_id,
            row.table_id,
            row.variable_id,
            row.grid_label,
            row.version,
        ]
    )


def _cmip6_old(row: sa.Row) -> str | None:
    return _join(
        [
            "CMIP6",
            row.activity_id,
            row.institution_id,
            row.source_id,
            row.experiment_id,
            row.member_id,
            row.table_id,
            row.variable_id,
            row.grid_label,
            row.version,
        ]
    )


def _reference_new(row: sa.Row) -> str | None:
    return _join(
        [
            row.activity_id,
            row.institution_id,
            row.source_id,
            row.frequency,
            row.variable_id,
            row.grid_label,
            row.version,
        ]
    )


def _reference_old(row: sa.Row) -> str | None:
    resolution = row.nominal_resolution.replace(" ", "") if row.nominal_resolution else None
    return _join(
        [
            row.activity_id,
            row.activity_id,
            row.institution_id,
            row.source_id,
            row.frequency,
            row.variable_id,
            resolution,
            row.grid_label,
            row.version,
        ]
    )


def _rewrite(table_name: str, columns: Sequence[str], build: Callable[[sa.Row], str | None]) -> None:
    """Rebuild each row's ``instance_id`` and keep ``dataset.slug`` in step with it.

    ``dataset.slug`` is globally unique,
    so a rebuilt id that is already taken (or produced twice) leaves those rows untouched,
    because silently collapsing two datasets into one would destroy provenance.
    """
    bind = op.get_bind()
    metadata = sa.MetaData()
    dataset = sa.Table("dataset", metadata, autoload_with=bind)
    sub = sa.Table(table_name, metadata, autoload_with=bind)

    taken = set(bind.execute(sa.select(dataset.c.slug)).scalars())
    rows = bind.execute(sa.select(sub.c.id, sub.c.instance_id, *(sub.c[c] for c in columns))).all()

    for row in rows:
        new_id = build(row)
        if new_id is None or new_id == row.instance_id:
            continue
        if new_id in taken:
            logger.warning(
                f"Keeping {table_name} id={row.id} as {row.instance_id!r}: {new_id!r} is already taken"
            )
            continue
        taken.add(new_id)
        bind.execute(sub.update().where(sub.c.id == row.id).values(instance_id=new_id))
        bind.execute(dataset.update().where(dataset.c.id == row.id).values(slug=new_id))


def upgrade() -> None:
    _rewrite("cmip6_dataset", _CMIP6_COLUMNS, _cmip6_new)
    for table_name in _REFERENCE_TABLES:
        _rewrite(table_name, _REFERENCE_COLUMNS, _reference_new)


def downgrade() -> None:
    _rewrite("cmip6_dataset", _CMIP6_COLUMNS, _cmip6_old)
    for table_name in _REFERENCE_TABLES:
        _rewrite(table_name, _REFERENCE_COLUMNS, _reference_old)
