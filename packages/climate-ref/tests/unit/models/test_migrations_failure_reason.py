"""Tests for the ``execution.failure_reason`` migration."""

import sqlalchemy as sa
from alembic import command

from climate_ref.database import Database

_PREVIOUS_REVISION = "d6e7f8a9b0c1"
_THIS_REVISION = "e7f8a9b0c1d2"


def _insert_execution(db: Database, execution_id: int, successful: bool | None) -> None:
    with db._engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO execution (id, output_fragment, execution_group_id, dataset_hash, retracted,"
                " successful, created_at, updated_at)"
                " VALUES (:id, 'fragment', 1, 'hash', false, :successful, :now, :now)"
            ),
            {"id": execution_id, "successful": successful, "now": "2026-01-01 00:00:00"},
        )


def _failure_reasons(db: Database) -> dict[int, str | None]:
    with db._engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT id, failure_reason FROM execution")).all()
    return dict(rows)


class TestFailureReasonMigration:
    def test_upgrade_backfills_existing_failures(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)
        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        _insert_execution(db, 1, successful=False)
        _insert_execution(db, 2, successful=True)
        _insert_execution(db, 3, successful=None)

        command.upgrade(alembic_cfg, _THIS_REVISION)

        assert _failure_reasons(db) == {1: "unknown", 2: None, 3: None}

        command.upgrade(alembic_cfg, "head")

    def test_downgrade_drops_column(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        columns = {c["name"] for c in sa.inspect(db._engine).get_columns("execution")}
        assert "failure_reason" not in columns

        command.upgrade(alembic_cfg, "head")
