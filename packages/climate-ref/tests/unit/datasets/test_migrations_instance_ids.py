"""Tests for the migration that rewrites stored instance_ids to the identifiers ESGF publishes.

Covers:
1. CMIP6 ids naming only the primary activity of a multi-activity dataset.
2. obs4MIPs-style ids dropping the duplicated collection prefix and the
   ``nominal_resolution`` segment, across all three reference tables.
3. ``dataset.slug`` staying in step with the rewritten ``instance_id``.
4. Rows whose rebuilt id collides with a taken slug being left untouched.
5. Downgrade restoring the old format.

Driven in-test via ``database.alembic_config(...)`` + ``command.upgrade``/``downgrade``.
"""

import sqlalchemy as sa
from alembic import command

from climate_ref.database import Database
from climate_ref_core.source_types import SourceDatasetType

# The revision immediately before "rewrite stored instance_ids to the identifiers ESGF publishes".
_PREVIOUS_REVISION = "c1d2e3f4a5b6"
_THIS_REVISION = "d6e7f8a9b0c1"

_CMIP6_OLD = "CMIP6.C4MIP CDRMIP.MIROC.MIROC-ES2L.esm-1pctCO2.r1i1p1f2.Amon.tas.gn.v20200101"
_CMIP6_NEW = "CMIP6.C4MIP.MIROC.MIROC-ES2L.esm-1pctCO2.r1i1p1f2.Amon.tas.gn.v20200101"
_CMIP6_SINGLE = "CMIP6.CMIP.MIROC.MIROC-ES2L.historical.r1i1p1f2.Amon.tas.gn.v20200101"

_OBS_OLD = "obs4MIPs.obs4MIPs.NASA-LaRC.CERES-EBAF-4-2-1.mon.rsutcs.100km.gn.v20260220"
_OBS_NEW = "obs4MIPs.NASA-LaRC.CERES-EBAF-4-2-1.mon.rsutcs.gn.v20260220"
_OBS_OLD_250 = "obs4MIPs.obs4MIPs.NASA-LaRC.CERES-EBAF-4-2-1.mon.rsutcs.250km.gn.v20260220"

_REF_OLD = "obs4REF.obs4REF.MOHC.HadISST-1-1.mon.ts.250km.gn.v20250415"
_REF_NEW = "obs4REF.MOHC.HadISST-1-1.mon.ts.gn.v20250415"

_PMP_OLD = "obs4MIPs.obs4MIPs.PCMDI.ERA-5.mon.zg.250km.gn.v20250211"
_PMP_NEW = "obs4MIPs.PCMDI.ERA-5.mon.zg.gn.v20250211"


class TestInstanceIdRewrite:
    def _insert_fixture_rows(self, bind: sa.Engine) -> dict[str, int]:
        metadata = sa.MetaData()
        dataset = sa.Table("dataset", metadata, autoload_with=bind)
        cmip6_dataset = sa.Table("cmip6_dataset", metadata, autoload_with=bind)
        tables = {
            name: sa.Table(name, metadata, autoload_with=bind)
            for name in ("obs4mips_dataset", "obs4ref_dataset", "pmp_climatology_dataset")
        }

        ids: dict[str, int] = {}
        with bind.begin() as conn:

            def _insert_cmip6(key: str, slug: str, activity_id: str, experiment_id: str) -> None:
                base_id = conn.execute(
                    dataset.insert().values(
                        slug=slug, dataset_type=SourceDatasetType.CMIP6.value, finalised=True
                    )
                ).inserted_primary_key[0]
                conn.execute(
                    cmip6_dataset.insert().values(
                        id=base_id,
                        instance_id=slug,
                        activity_id=activity_id,
                        experiment_id=experiment_id,
                        institution_id="MIROC",
                        source_id="MIROC-ES2L",
                        member_id="r1i1p1f2",
                        table_id="Amon",
                        variable_id="tas",
                        grid_label="gn",
                        variant_label="r1i1p1f2",
                        version="v20200101",
                    )
                )
                ids[key] = base_id

            _insert_cmip6("cmip6_multi", _CMIP6_OLD, "C4MIP CDRMIP", "esm-1pctCO2")
            _insert_cmip6("cmip6_single", _CMIP6_SINGLE, "CMIP", "historical")

            def _insert_reference(
                key: str,
                table_name: str,
                slug: str,
                dataset_type: SourceDatasetType,
                *,
                activity_id: str,
                institution_id: str,
                source_id: str,
                variable_id: str,
                nominal_resolution: str,
                version: str,
            ) -> None:
                base_id = conn.execute(
                    dataset.insert().values(slug=slug, dataset_type=dataset_type.value, finalised=True)
                ).inserted_primary_key[0]
                conn.execute(
                    tables[table_name]
                    .insert()
                    .values(
                        id=base_id,
                        instance_id=slug,
                        activity_id=activity_id,
                        institution_id=institution_id,
                        source_id=source_id,
                        variable_id=variable_id,
                        nominal_resolution=nominal_resolution,
                        version=version,
                        frequency="mon",
                        grid="native",
                        grid_label="gn",
                        long_name="Test",
                        realm="atmos",
                        product="observations",
                        source_type="satellite",
                        units="K",
                        variant_label="v1",
                        vertical_levels=1,
                        source_version_number="1",
                    )
                )
                ids[key] = base_id

            _insert_reference(
                "obs",
                "obs4mips_dataset",
                _OBS_OLD,
                SourceDatasetType.obs4MIPs,
                activity_id="obs4MIPs",
                institution_id="NASA-LaRC",
                source_id="CERES-EBAF-4-2-1",
                variable_id="rsutcs",
                nominal_resolution="100 km",
                version="v20260220",
            )
            # Same dataset at a coarser resolution: its rebuilt id collides with the row
            # above, so the migration must leave it untouched.
            _insert_reference(
                "obs_collision",
                "obs4mips_dataset",
                _OBS_OLD_250,
                SourceDatasetType.obs4MIPs,
                activity_id="obs4MIPs",
                institution_id="NASA-LaRC",
                source_id="CERES-EBAF-4-2-1",
                variable_id="rsutcs",
                nominal_resolution="250 km",
                version="v20260220",
            )
            _insert_reference(
                "obs4ref",
                "obs4ref_dataset",
                _REF_OLD,
                SourceDatasetType.obs4REF,
                activity_id="obs4REF",
                institution_id="MOHC",
                source_id="HadISST-1-1",
                variable_id="ts",
                nominal_resolution="250 km",
                version="v20250415",
            )
            _insert_reference(
                "pmp",
                "pmp_climatology_dataset",
                _PMP_OLD,
                SourceDatasetType.PMPClimatology,
                activity_id="obs4MIPs",
                institution_id="PCMDI",
                source_id="ERA-5",
                variable_id="zg",
                nominal_resolution="250 km",
                version="v20250211",
            )
        return ids

    def _slugs_and_instance_ids(self, bind: sa.Engine, ids: dict[str, int]) -> dict[str, tuple[str, str]]:
        metadata = sa.MetaData()
        dataset = sa.Table("dataset", metadata, autoload_with=bind)
        by_table = {
            "cmip6_dataset": ("cmip6_multi", "cmip6_single"),
            "obs4mips_dataset": ("obs", "obs_collision"),
            "obs4ref_dataset": ("obs4ref",),
            "pmp_climatology_dataset": ("pmp",),
        }
        result: dict[str, tuple[str, str]] = {}
        with bind.connect() as conn:
            slugs = {row.id: row.slug for row in conn.execute(sa.select(dataset.c.id, dataset.c.slug))}
            for table_name, keys in by_table.items():
                sub = sa.Table(table_name, metadata, autoload_with=bind)
                instance_ids = {
                    row.id: row.instance_id for row in conn.execute(sa.select(sub.c.id, sub.c.instance_id))
                }
                for key in keys:
                    result[key] = (slugs[ids[key]], instance_ids[ids[key]])
        return result

    def test_upgrade_rewrites_and_downgrade_restores(self, db: Database, config) -> None:
        alembic_cfg = db.alembic_config(config)

        # Start from just before this revision, with rows in the old format.
        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)
        bind = db._engine
        ids = self._insert_fixture_rows(bind)

        command.upgrade(alembic_cfg, _THIS_REVISION)

        rewritten = self._slugs_and_instance_ids(bind, ids)
        assert rewritten["cmip6_multi"] == (_CMIP6_NEW, _CMIP6_NEW)
        assert rewritten["cmip6_single"] == (_CMIP6_SINGLE, _CMIP6_SINGLE)
        assert rewritten["obs"] == (_OBS_NEW, _OBS_NEW)
        assert rewritten["obs_collision"] == (_OBS_OLD_250, _OBS_OLD_250)
        assert rewritten["obs4ref"] == (_REF_NEW, _REF_NEW)
        assert rewritten["pmp"] == (_PMP_NEW, _PMP_NEW)

        command.downgrade(alembic_cfg, _PREVIOUS_REVISION)

        restored = self._slugs_and_instance_ids(bind, ids)
        assert restored["cmip6_multi"] == (_CMIP6_OLD, _CMIP6_OLD)
        assert restored["cmip6_single"] == (_CMIP6_SINGLE, _CMIP6_SINGLE)
        assert restored["obs"] == (_OBS_OLD, _OBS_OLD)
        assert restored["obs_collision"] == (_OBS_OLD_250, _OBS_OLD_250)
        assert restored["obs4ref"] == (_REF_OLD, _REF_OLD)
        assert restored["pmp"] == (_PMP_OLD, _PMP_OLD)

        # Clean out the fixture rows so re-applying head doesn't trip on stale data,
        # then restore head for the fixture teardown.
        metadata = sa.MetaData()
        dataset = sa.Table("dataset", metadata, autoload_with=bind)
        with bind.begin() as conn:
            for table_name in (
                "cmip6_dataset",
                "obs4mips_dataset",
                "obs4ref_dataset",
                "pmp_climatology_dataset",
            ):
                sub = sa.Table(table_name, metadata, autoload_with=bind)
                conn.execute(sub.delete())
            conn.execute(dataset.delete())
        command.upgrade(alembic_cfg, "head")
