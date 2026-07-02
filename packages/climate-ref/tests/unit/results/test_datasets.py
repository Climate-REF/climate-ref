"""Unit tests for the dataset read layer."""

import pytest

from climate_ref.models.dataset import CMIP6Dataset, DatasetFile, Obs4MIPsDataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup, execution_datasets
from climate_ref.results import DatasetFilter, Reader
from climate_ref_core.source_types import SourceDatasetType


def test_import_smoke() -> None:
    """`DatasetFilter` is importable from the curated top-level surface."""
    assert DatasetFilter is not None


def _make_cmip6_dataset(
    *,
    slug: str,
    source_id: str = "TEST-MODEL",
    experiment_id: str = "historical",
    variable_id: str = "tas",
    member_id: str = "r1i1p1f1",
    table_id: str = "Amon",
    grid_label: str = "gn",
    version: str = "v20200101",
    finalised: bool = True,
) -> CMIP6Dataset:
    return CMIP6Dataset(
        slug=slug,
        dataset_type=SourceDatasetType.CMIP6,
        activity_id="CMIP",
        experiment_id=experiment_id,
        institution_id="TEST",
        source_id=source_id,
        member_id=member_id,
        table_id=table_id,
        variable_id=variable_id,
        grid_label=grid_label,
        version=version,
        instance_id=slug,
        variant_label=member_id,
        finalised=finalised,
    )


@pytest.fixture
def db_with_datasets(db_seeded):
    """A seeded DB (CMIP6 + obs4MIPs already ingested) plus controlled version/facet variety."""
    with db_seeded.session.begin():
        v2 = _make_cmip6_dataset(
            slug="CMIP.TEST.TEST-MODEL.historical.Amon.tas.gn.v2",
            source_id="TEST-MODEL",
            variable_id="tas",
            version="v2",
        )
        v10 = _make_cmip6_dataset(
            slug="CMIP.TEST.TEST-MODEL.historical.Amon.tas.gn.v10",
            source_id="TEST-MODEL",
            variable_id="tas",
            version="v10",
        )
        unfinalised = _make_cmip6_dataset(
            slug="CMIP.TEST.OTHER-MODEL.historical.Amon.pr.gn.v1",
            source_id="OTHER-MODEL",
            variable_id="pr",
            version="v1",
            finalised=False,
        )
        db_seeded.session.add_all([v2, v10, unfinalised])
        db_seeded.session.flush()

        db_seeded.session.add(
            DatasetFile(
                dataset_id=v10.id,
                path="tas_v10.nc",
                start_time="2000-01-01",
                end_time="2000-12-31",
                tracking_id="hdl:test-tracking-id",
            )
        )

        obs_dataset = Obs4MIPsDataset(
            slug="obs4mips-test-dataset",
            dataset_type=SourceDatasetType.obs4MIPs,
            activity_id="obs4MIPs",
            frequency="mon",
            grid="native",
            grid_label="gn",
            institution_id="TEST",
            long_name="Test obs4MIPs dataset",
            nominal_resolution="100 km",
            realm="atmos",
            product="observations",
            source_id="TEST-OBS",
            source_type="satellite",
            units="K",
            variable_id="tas",
            variant_label="v1",
            version="v1",
            vertical_levels=1,
            source_version_number="1",
            instance_id="obs4mips-test-dataset",
        )
        db_seeded.session.add(obs_dataset)
        db_seeded.session.flush()

        diag = db_seeded.session.query(Diagnostic).first()
        eg = ExecutionGroup(key="key1", diagnostic_id=diag.id, selectors={})
        db_seeded.session.add(eg)
        db_seeded.session.flush()
        execution = Execution(
            execution_group_id=eg.id, successful=True, output_fragment="out1", dataset_hash="hash1"
        )
        db_seeded.session.add(execution)
        db_seeded.session.flush()

        db_seeded.session.execute(
            execution_datasets.insert(),
            [{"execution_id": execution.id, "dataset_id": v10.id}],
        )

    db_seeded.session.commit()

    db_seeded.dataset_ids = {
        "v2": v2.id,
        "v10": v10.id,
        "unfinalised": unfinalised.id,
        "obs4mips": obs_dataset.id,
    }
    db_seeded.execution_id = execution.id
    db_seeded.diagnostic_slug = diag.slug
    return db_seeded


class TestFacetFiltering:
    def test_or_within_facet(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"variable_id": ("tas", "pr")},
                latest_only=False,
            )
        )
        variable_ids = {d.facets["variable_id"] for d in coll}
        assert variable_ids <= {"tas", "pr"}
        assert len(coll) >= 2

    def test_and_across_facets(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"variable_id": ("pr",), "source_id": ("OTHER-MODEL",)},
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.datasets[0].facets["variable_id"] == "pr"
        assert coll.datasets[0].facets["source_id"] == "OTHER-MODEL"

    def test_and_across_facets_no_match(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"variable_id": ("pr",), "source_id": ("TEST-MODEL",)},
                latest_only=False,
            )
        )
        assert len(coll) == 0

    def test_unknown_facet_raises(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        with pytest.raises(ValueError, match="Unknown facet"):
            reader.datasets.datasets(
                DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"nonexistent_facet": ("x",)})
            )

    def test_unknown_facet_on_base_entity_raises(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        with pytest.raises(ValueError, match="Unknown facet"):
            reader.datasets.datasets(DatasetFilter(facets={"variable_id": ("tas",)}))

    def test_facet_on_base_entity_matches_base_column(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(DatasetFilter(facets={"dataset_type": (SourceDatasetType.obs4MIPs,)}))
        assert len(coll) > 0
        assert all(d.dataset_type == SourceDatasetType.obs4MIPs for d in coll)
        assert db_with_datasets.dataset_ids["obs4mips"] in {d.id for d in coll}


class TestFinalisedFilter:
    def test_finalised_true(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, finalised=True, latest_only=False)
        )
        assert all(d.finalised for d in coll)
        slugs = {d.slug for d in coll}
        assert "CMIP.TEST.OTHER-MODEL.historical.Amon.pr.gn.v1" not in slugs

    def test_finalised_false(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, finalised=False, latest_only=False)
        )
        assert len(coll) == 1
        assert coll.datasets[0].slug == "CMIP.TEST.OTHER-MODEL.historical.Amon.pr.gn.v1"


class TestLatestOnly:
    def test_keeps_newest_numeric_version(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            )
        )
        # v10 must win over v2 numerically, not lexically ("v10" < "v2" as strings).
        assert len(coll) == 1
        assert coll.datasets[0].id == db_with_datasets.dataset_ids["v10"]

    def test_latest_only_false_keeps_both_versions(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
                latest_only=False,
            )
        )
        assert len(coll) == 2

    def test_latest_only_noop_when_source_type_none(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(DatasetFilter(latest_only=True))
        # All datasets (across types) present -- no dataset_id_metadata to group by.
        all_coll = reader.datasets.datasets(DatasetFilter(latest_only=False))
        assert len(coll) == len(all_coll)


class TestExecutionIdJoin:
    def test_execution_id_scopes_to_linked_datasets(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                execution_id=db_with_datasets.execution_id,
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.datasets[0].id == db_with_datasets.dataset_ids["v10"]

    def test_execution_id_no_match(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, execution_id=999999, latest_only=False)
        )
        assert len(coll) == 0


class TestDiagnosticSlugJoin:
    def test_diagnostic_slug_scopes_to_linked_datasets(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                diagnostic_slug=db_with_datasets.diagnostic_slug,
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.datasets[0].id == db_with_datasets.dataset_ids["v10"]

    def test_diagnostic_slug_no_match(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                diagnostic_slug="nonexistent-diagnostic",
                latest_only=False,
            )
        )
        assert len(coll) == 0

    def test_execution_id_and_diagnostic_slug_together(self, db_with_datasets):
        # Both axes reach through ``execution_datasets``; setting both must not emit a duplicate,
        # unaliased self-join. This executes the query, so invalid SQL would raise here.
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                execution_id=db_with_datasets.execution_id,
                diagnostic_slug=db_with_datasets.diagnostic_slug,
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.datasets[0].id == db_with_datasets.dataset_ids["v10"]


class TestLimit:
    def test_limit_applied_after_latest_only(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(DatasetFilter(source_type=SourceDatasetType.CMIP6), limit=1)
        assert len(coll) == 1


class TestIncludeFiles:
    def test_include_files_true_populates_files(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            ),
            include_files=True,
        )
        assert len(coll) == 1
        ds = coll.datasets[0]
        assert len(ds.files) == 1
        assert ds.files[0].path == "tas_v10.nc"
        assert ds.files[0].start_time == "2000-01-01"
        assert ds.files[0].end_time == "2000-12-31"
        assert ds.files[0].tracking_id == "hdl:test-tracking-id"

    def test_include_files_false_leaves_files_empty(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            ),
            include_files=False,
        )
        assert len(coll) == 1
        assert coll.datasets[0].files == ()


class TestSourceTypeNonePolymorphic:
    def test_base_query_returns_all_types(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(DatasetFilter(latest_only=False))
        types = {d.dataset_type for d in coll}
        assert SourceDatasetType.CMIP6 in types
        assert SourceDatasetType.obs4MIPs in types


class TestDetachment:
    def test_dto_fields_survive_session_close(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            ),
            include_files=True,
        )
        db_with_datasets.session.expunge_all()

        assert len(coll) == 1
        ds = coll.datasets[0]
        assert ds.slug == "CMIP.TEST.TEST-MODEL.historical.Amon.tas.gn.v10"
        assert ds.facets["variable_id"] == "tas"
        assert len(ds.files) == 1


class TestToPandas:
    def test_to_pandas_columns(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.datasets(DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False))
        df = coll.to_pandas()
        for col in ("id", "slug", "dataset_type", "finalised", "created_at", "updated_at"):
            assert col in df.columns
        assert len(df) == len(coll)
