"""Tests for the ``execution`` resource usage migration.

Covers the SQLite roundtrip: upgrade adds the nullable columns and downgrade drops them.

Driven in-test via ``database.alembic_config(...)`` + ``command.upgrade``/``downgrade``,
matching the pattern used by ``test_migrations_retracted_at.py``.
"""

import sqlalchemy as sa
from alembic import command

from climate_ref.database import Database

_PREVIOUS_REVISION = "a4c5d6e7f8b9"
_THIS_REVISION = "c1d2e3f4a5b6"

_COLUMNS = {
    "wall_seconds",
    "cpu_seconds",
    "peak_memory_bytes",
    "memory_source",
    "memory_limit_bytes",
    "cpu_limit",
    "resources_exclusive",
    "queue_seconds",
    "resource_context",
}


def _execution_columns(db: Database) -> set[str]:
    return {c["name"] for c in sa.inspect(db._engine).get_columns("execution")}


class TestExecutionResourcesMigration:
    """The migration adds and drops the nullable resource usage columns."""

    def test_upgrade_adds_columns(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        assert not _COLUMNS & _execution_columns(db)

        command.upgrade(alembic_cfg, _THIS_REVISION)
        assert _COLUMNS <= _execution_columns(db)

        command.upgrade(alembic_cfg, "head")

    def test_downgrade_drops_columns(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.upgrade(alembic_cfg, "head")
        assert _COLUMNS <= _execution_columns(db)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        assert not _COLUMNS & _execution_columns(db)

        command.upgrade(alembic_cfg, "head")
