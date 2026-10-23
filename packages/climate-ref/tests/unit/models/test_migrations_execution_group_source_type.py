"""Tests for the ``execution_group_source_type`` migration."""

import json

import sqlalchemy as sa
from alembic import command

from climate_ref.database import Database

_PREVIOUS_REVISION = "a9b0c1d2e3f4"
_THIS_REVISION = "b0c1d2e3f4a5"


def _insert_group(db: Database, group_id: int, selectors: dict) -> None:
    with db._engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO execution_group (id, diagnostic_id, key, diagnostic_version, dirty, selectors,"
                " created_at, updated_at)"
                " VALUES (:id, 1, :key, 1, false, :selectors, :now, :now)"
            ),
            {
                "id": group_id,
                "key": f"key-{group_id}",
                "selectors": json.dumps(selectors),
                "now": "2026-01-01 00:00:00",
            },
        )


def _links(db: Database) -> set[tuple[int, str]]:
    with db._engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT execution_group_id, source_type FROM execution_group_source_type")
        ).all()
    return {tuple(row) for row in rows}


class TestExecutionGroupSourceTypeMigration:
    def test_upgrade_backfills_from_selectors(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)
        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        _insert_group(db, 1, {"cmip6": [["source_id", "ACCESS-ESM1-5"]], "obs4mips": []})
        _insert_group(db, 2, {"obs4mips": [["source_id", "HadISST-1-1"]]})
        _insert_group(db, 3, {})

        command.upgrade(alembic_cfg, _THIS_REVISION)

        assert _links(db) == {(1, "CMIP6"), (1, "obs4MIPs"), (2, "obs4MIPs")}

        command.upgrade(alembic_cfg, "head")

    def test_downgrade_drops_table(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        assert "execution_group_source_type" not in sa.inspect(db._engine).get_table_names()

        command.upgrade(alembic_cfg, "head")
