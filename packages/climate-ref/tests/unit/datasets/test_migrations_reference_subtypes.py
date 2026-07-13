"""Tests for the obs4REF / ESMValTool reference subtype migration.

Covers:
1. Upgrade creates both subtype tables. Downgrade drops them (SQLite roundtrip).
2. A dataset row can be persisted through each new polymorphic subtype after upgrading.
3. ``with_polymorphic(Dataset, "*")`` works once the tables exist. This is the regression the
   migration exists for: it LEFT JOINs every mapped subtype table, so a declared-but-uncreated
   table breaks ``executor.reingest.reconstruct_execution_definition`` on every database.

Driven in-test via ``database.alembic_config(...)`` + ``command.upgrade``/``downgrade``,
matching the pattern used by the other migration tests.
"""

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.orm import with_polymorphic

from climate_ref.database import Database
from climate_ref.models.dataset import Dataset, ESMValToolReferenceDataset, Obs4REFDataset
from climate_ref_core.source_types import SourceDatasetType

_PREVIOUS_REVISION = "b7c8d9e0a1f2"
_THIS_REVISION = "e1f2a3b4c5d6"

_NEW_TABLES = {"obs4ref_dataset", "esmvaltool_reference_dataset"}


def _obs4ref(slug: str) -> Obs4REFDataset:
    return Obs4REFDataset(
        slug=slug,
        dataset_type=SourceDatasetType.obs4REF,
        activity_id="obs4REF",
        frequency="mon",
        grid="native",
        grid_label="gn",
        institution_id="TESTORG",
        long_name="Test",
        nominal_resolution="100 km",
        realm="atmos",
        product="observations",
        source_id="TEST-SRC",
        source_type="satellite",
        units="K",
        variable_id="ts",
        variant_label="v1",
        version="v1",
        vertical_levels=1,
        source_version_number="1",
        instance_id=slug,
    )


def _esmvaltool(slug: str, *, frequency: str = "mon") -> ESMValToolReferenceDataset:
    return ESMValToolReferenceDataset(
        slug=slug,
        dataset_type=SourceDatasetType.ESMValToolReference,
        project="OBS6",
        source_id="CERES-EBAF",
        variable_id="rlut",
        frequency=frequency,
        version="Ed4.2",
        instance_id=slug,
    )


class TestReferenceSubtypeMigration:
    def test_upgrade_creates_both_tables(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        tables = set(sa.inspect(db._engine).get_table_names())
        assert not (_NEW_TABLES & tables)

        command.upgrade(alembic_cfg, _THIS_REVISION)
        tables = set(sa.inspect(db._engine).get_table_names())
        assert _NEW_TABLES <= tables

        command.upgrade(alembic_cfg, "head")

    def test_downgrade_drops_both_tables(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.upgrade(alembic_cfg, "head")
        assert _NEW_TABLES <= set(sa.inspect(db._engine).get_table_names())

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        assert not (_NEW_TABLES & set(sa.inspect(db._engine).get_table_names()))

        command.upgrade(alembic_cfg, "head")

    def test_obs4ref_columns_match_obs4mips(self, db: Database) -> None:
        """The mixin-backed tables must agree, since they share ``ReferenceDatasetMixin``."""
        inspector = sa.inspect(db._engine)
        obs4ref = {c["name"] for c in inspector.get_columns("obs4ref_dataset")}
        obs4mips = {c["name"] for c in inspector.get_columns("obs4mips_dataset")}
        assert obs4ref == obs4mips


class TestReferenceSubtypePersistence:
    def test_obs4ref_row_roundtrips(self, db: Database) -> None:
        dataset = _obs4ref("obs4REF.TESTORG.TEST-SRC.mon.ts.100km.gn.v1")
        with db.session.begin():
            db.session.add(dataset)
        db.session.expire_all()

        with db.session.begin():
            fetched = db.session.query(Obs4REFDataset).filter_by(slug=dataset.slug).one()
            assert fetched.dataset_type is SourceDatasetType.obs4REF
            assert fetched.source_id == "TEST-SRC"

    def test_esmvaltool_reference_row_roundtrips(self, db: Database) -> None:
        dataset = _esmvaltool("OBS6.CERES-EBAF.Amon.rlut.Ed4.2")
        with db.session.begin():
            db.session.add(dataset)
        db.session.expire_all()

        with db.session.begin():
            fetched = db.session.query(ESMValToolReferenceDataset).filter_by(slug=dataset.slug).one()
            assert fetched.dataset_type is SourceDatasetType.ESMValToolReference
            assert fetched.project == "OBS6"
            assert fetched.frequency == "mon"
            assert fetched.tier is None

    def test_same_variable_at_two_frequencies_coexist(self, db: Database) -> None:
        """The reason ``frequency`` is stored at all.

        Monthly and daily data for one variable differ in no other column. They are distinct
        datasets, so both must persist rather than collide.
        """
        with db.session.begin():
            db.session.add(_esmvaltool("OBS6.CERES-EBAF.mon.rlut.Ed4.2", frequency="mon"))
            db.session.add(_esmvaltool("OBS6.CERES-EBAF.day.rlut.Ed4.2", frequency="day"))

        db.session.expire_all()
        with db.session.begin():
            rows = db.session.query(ESMValToolReferenceDataset).filter_by(variable_id="rlut").all()
            assert {row.frequency for row in rows} == {"mon", "day"}

    def test_frequency_is_required(self, db: Database) -> None:
        """A NULL ``frequency`` must be rejected by the schema, not silently stored."""
        dataset = ESMValToolReferenceDataset(
            slug="OBS6.CERES-EBAF.none.rlut.Ed4.2",
            dataset_type=SourceDatasetType.ESMValToolReference,
            project="OBS6",
            source_id="CERES-EBAF",
            variable_id="rlut",
            version="Ed4.2",
            instance_id="OBS6.CERES-EBAF.none.rlut.Ed4.2",
        )
        with pytest.raises(sa.exc.IntegrityError):
            with db.session.begin():
                db.session.add(dataset)
        db.session.rollback()

    def test_with_polymorphic_star_queries_every_subtype(self, db: Database) -> None:
        """Regression guard for the reason this migration is not deferrable.

        ``executor.reingest`` uses ``with_polymorphic(Dataset, "*")``, which joins every mapped
        subtype table. Without the migration this raises ``no such table: obs4ref_dataset`` even
        when no reference datasets exist.
        """
        with db.session.begin():
            db.session.add(_obs4ref("obs4REF.TESTORG.TEST-SRC.mon.ts.100km.gn.v2"))
            db.session.add(_esmvaltool("OBS6.CERES-EBAF.Amon.rsut.Ed4.2"))

        dataset_poly = with_polymorphic(Dataset, "*")
        rows = db.session.query(dataset_poly).all()
        found = {row.dataset_type for row in rows}
        assert SourceDatasetType.obs4REF in found
        assert SourceDatasetType.ESMValToolReference in found
