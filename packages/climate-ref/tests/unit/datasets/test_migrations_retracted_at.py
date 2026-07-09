"""Tests for the ``dataset.retracted_at`` migration (dataset retraction).

Covers:
1. Upgrade adds the nullable ``retracted_at`` column; downgrade drops it (SQLite roundtrip).
2. A row inserted through the ORM after upgrading defaults to ``retracted_at is None``.

Driven in-test via ``database.alembic_config(...)`` + ``command.upgrade``/``downgrade``,
matching the pattern used by ``test_migrations_version_key.py``.
"""

import sqlalchemy as sa
from alembic import command

from climate_ref.database import Database
from climate_ref.models.dataset import CMIP6Dataset
from climate_ref_core.source_types import SourceDatasetType

_PREVIOUS_REVISION = "f6a7b8c9d0e1"
_THIS_REVISION = "b7c8d9e0a1f2"


class TestRetractedAtMigration:
    """The migration adds/drops the nullable ``dataset.retracted_at`` column."""

    def test_upgrade_adds_column(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        insp = sa.inspect(db._engine)
        cols = {c["name"] for c in insp.get_columns("dataset")}
        assert "retracted_at" not in cols

        command.upgrade(alembic_cfg, _THIS_REVISION)
        insp = sa.inspect(db._engine)
        cols = {c["name"] for c in insp.get_columns("dataset")}
        assert "retracted_at" in cols

        command.upgrade(alembic_cfg, "head")

    def test_downgrade_drops_column(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.upgrade(alembic_cfg, "head")
        insp = sa.inspect(db._engine)
        cols = {c["name"] for c in insp.get_columns("dataset")}
        assert "retracted_at" in cols

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        insp = sa.inspect(db._engine)
        cols = {c["name"] for c in insp.get_columns("dataset")}
        assert "retracted_at" not in cols

        command.upgrade(alembic_cfg, "head")


class TestRetractedAtDefault:
    """A freshly inserted row defaults to ``retracted_at is None`` (active)."""

    def test_insert_defaults_to_not_retracted(self, db: Database) -> None:
        dataset = CMIP6Dataset(
            slug="retracted-at-default-test",
            dataset_type=SourceDatasetType.CMIP6,
            activity_id="CMIP",
            experiment_id="historical",
            institution_id="TEST",
            source_id="TEST-MODEL",
            member_id="r1i1p1f1",
            table_id="Amon",
            variable_id="tas",
            grid_label="gn",
            version="v1",
            instance_id="retracted-at-default-test",
            variant_label="r1i1p1f1",
            finalised=True,
        )
        with db.session.begin():
            db.session.add(dataset)
        db.session.expire_all()

        with db.session.begin():
            fetched = db.session.query(CMIP6Dataset).filter_by(slug=dataset.slug).one()
            assert fetched.retracted_at is None
