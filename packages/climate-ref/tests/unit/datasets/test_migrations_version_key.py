"""Tests for the ``version_key`` migration (SQL-side latest-version deduplication).

Covers:
1. Backfill of pre-existing rows across all four ``Dataset`` subclasses, keyed via
   ``version_sort_key`` (numeric, non-conforming, and non-numeric versions).
2. Downgrade dropping the column.

Driven in-test via ``database.alembic_config(...)`` + ``command.upgrade``/``downgrade``.
"""

import sqlalchemy as sa
from alembic import command

from climate_ref.database import Database
from climate_ref_core.source_types import SourceDatasetType

# The revision immediately before "add version_key to base dataset table".
_PREVIOUS_REVISION = "e5f6a7b8c9d0"
_THIS_REVISION = "f6a7b8c9d0e1"


class TestVersionKeyBackfill:
    """The migration backfills ``version_key`` for rows that pre-date the column."""

    def test_backfill_computes_correct_keys(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        # Start from just before this revision so the column doesn't exist yet.
        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        # Insert rows directly via Core (the ORM model no longer matches the downgraded schema)
        # covering: numeric version, multi-digit numeric version, and a non-conforming version,
        # across two different subclass tables.
        bind = db._engine
        metadata = sa.MetaData()
        dataset = sa.Table("dataset", metadata, autoload_with=bind)
        cmip6_dataset = sa.Table("cmip6_dataset", metadata, autoload_with=bind)
        obs4mips_dataset = sa.Table("obs4mips_dataset", metadata, autoload_with=bind)
        cmip7_dataset = sa.Table("cmip7_dataset", metadata, autoload_with=bind)
        pmp_climatology_dataset = sa.Table("pmp_climatology_dataset", metadata, autoload_with=bind)

        with bind.begin() as conn:
            base_common = {"finalised": True}
            cmip6_common = {
                "activity_id": "CMIP",
                "experiment_id": "historical",
                "institution_id": "TEST",
                "source_id": "TEST-MODEL",
                "member_id": "r1i1p1f1",
                "table_id": "Amon",
                "variable_id": "tas",
                "grid_label": "gn",
                "variant_label": "r1i1p1f1",
            }

            def _insert_cmip6(slug: str, version: str) -> int:
                base_id = conn.execute(
                    dataset.insert().values(
                        slug=slug, dataset_type=SourceDatasetType.CMIP6.value, **base_common
                    )
                ).inserted_primary_key[0]
                conn.execute(
                    cmip6_dataset.insert().values(
                        id=base_id, version=version, instance_id=slug, **cmip6_common
                    )
                )
                return base_id

            id_v2 = _insert_cmip6("backfill-v2", "v2")
            id_v10 = _insert_cmip6("backfill-v10", "v10")
            id_nonconforming = _insert_cmip6("backfill-nonconforming", "latest")

            obs_base_id = conn.execute(
                dataset.insert().values(
                    slug="backfill-obs4mips-v3", dataset_type=SourceDatasetType.obs4MIPs.value, **base_common
                )
            ).inserted_primary_key[0]
            conn.execute(
                obs4mips_dataset.insert().values(
                    id=obs_base_id,
                    version="v3",
                    instance_id="backfill-obs4mips-v3",
                    activity_id="obs4MIPs",
                    frequency="mon",
                    grid="native",
                    grid_label="gn",
                    institution_id="TEST",
                    long_name="Test",
                    nominal_resolution="100 km",
                    realm="atmos",
                    product="observations",
                    source_id="TEST-OBS",
                    source_type="satellite",
                    units="K",
                    variable_id="tas",
                    variant_label="v1",
                    vertical_levels=1,
                    source_version_number="1",
                )
            )

            # CMIP7 with a vYYYYMMDD version -- the real production format (key = 20250622).
            cmip7_base_id = conn.execute(
                dataset.insert().values(
                    slug="backfill-cmip7-date", dataset_type=SourceDatasetType.CMIP7.value, **base_common
                )
            ).inserted_primary_key[0]
            conn.execute(
                cmip7_dataset.insert().values(
                    id=cmip7_base_id,
                    version="v20250622",
                    instance_id="backfill-cmip7-date",
                    activity_id="CMIP",
                    institution_id="TEST",
                    source_id="TEST-MODEL",
                    experiment_id="historical",
                    variant_label="r1i1p1f1",
                    variable_id="tas",
                    grid_label="gn",
                    frequency="mon",
                    region="glb",
                    branding_suffix="tavg-h2m-hxy-x",
                    mip_era="CMIP7",
                )
            )

            pmp_base_id = conn.execute(
                dataset.insert().values(
                    slug="backfill-pmp-v5", dataset_type=SourceDatasetType.PMPClimatology.value, **base_common
                )
            ).inserted_primary_key[0]
            conn.execute(
                pmp_climatology_dataset.insert().values(
                    id=pmp_base_id,
                    version="v5",
                    instance_id="backfill-pmp-v5",
                    activity_id="PMP",
                    frequency="mon",
                    grid="native",
                    grid_label="gn",
                    institution_id="TEST",
                    long_name="Test",
                    nominal_resolution="100 km",
                    realm="atmos",
                    product="observations",
                    source_id="TEST-OBS",
                    source_type="satellite",
                    units="K",
                    variable_id="tas",
                    variant_label="v1",
                    vertical_levels=1,
                    source_version_number="1",
                )
            )

        # Upgrade to this revision -- adds the column and runs the Python backfill.
        command.upgrade(alembic_cfg, _THIS_REVISION)

        with bind.connect() as conn:
            dataset_reflected = sa.Table("dataset", sa.MetaData(), autoload_with=bind)
            rows = {
                row.id: row.version_key
                for row in conn.execute(sa.select(dataset_reflected.c.id, dataset_reflected.c.version_key))
            }

        assert rows[id_v2] == 2
        assert rows[id_v10] == 10
        assert rows[id_nonconforming] == -1
        assert rows[obs_base_id] == 3
        assert rows[cmip7_base_id] == 20250622
        assert rows[pmp_base_id] == 5

        # Re-apply head so the fixture teardown (which may run further migrations) doesn't fail.
        command.upgrade(alembic_cfg, "head")

    def test_downgrade_removes_column(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        insp = sa.inspect(db._engine)
        cols = {c["name"] for c in insp.get_columns("dataset")}
        assert "version_key" not in cols

        command.upgrade(alembic_cfg, "head")

        insp = sa.inspect(db._engine)
        cols = {c["name"] for c in insp.get_columns("dataset")}
        assert "version_key" in cols
