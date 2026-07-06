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
        coll = reader.datasets.list(
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
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"variable_id": ("pr",), "source_id": ("OTHER-MODEL",)},
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.items[0].facets["variable_id"] == "pr"
        assert coll.items[0].facets["source_id"] == "OTHER-MODEL"

    def test_and_across_facets_no_match(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
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
            reader.datasets.list(
                DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"nonexistent_facet": ("x",)})
            )

    def test_source_type_is_required(self):
        """``DatasetFilter`` has no default ``source_type``: a typed listing must choose a type."""
        with pytest.raises(TypeError):
            DatasetFilter()  # type: ignore[call-arg]


class TestFinalisedFilter:
    def test_finalised_true(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, finalised=True, latest_only=False)
        )
        assert all(d.finalised for d in coll)
        slugs = {d.slug for d in coll}
        assert "CMIP.TEST.OTHER-MODEL.historical.Amon.pr.gn.v1" not in slugs

    def test_finalised_false(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, finalised=False, latest_only=False)
        )
        assert len(coll) == 1
        assert coll.items[0].slug == "CMIP.TEST.OTHER-MODEL.historical.Amon.pr.gn.v1"


class TestLatestOnly:
    def test_keeps_newest_numeric_version(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            )
        )
        # v10 must win over v2 numerically, not lexically ("v10" < "v2" as strings).
        assert len(coll) == 1
        assert coll.items[0].id == db_with_datasets.dataset_ids["v10"]

    def test_latest_only_false_keeps_both_versions(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
                latest_only=False,
            )
        )
        assert len(coll) == 2

    def test_latest_only_drops_versions_vs_all(self, db_with_datasets):
        """The deduplicated listing is strictly smaller than the every-version listing."""
        reader = Reader(db_with_datasets)
        deduped = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=True))
        every_version = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False)
        )
        assert len(deduped) < len(every_version)
        assert deduped.total_count < every_version.total_count


class TestExecutionIdJoin:
    def test_execution_id_scopes_to_linked_datasets(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                execution_id=db_with_datasets.execution_id,
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.items[0].id == db_with_datasets.dataset_ids["v10"]

    def test_execution_id_no_match(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, execution_id=999999, latest_only=False)
        )
        assert len(coll) == 0


class TestDiagnosticSlugJoin:
    def test_diagnostic_slug_scopes_to_linked_datasets(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                diagnostic_slug=db_with_datasets.diagnostic_slug,
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.items[0].id == db_with_datasets.dataset_ids["v10"]

    def test_diagnostic_slug_no_match(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
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
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                execution_id=db_with_datasets.execution_id,
                diagnostic_slug=db_with_datasets.diagnostic_slug,
                latest_only=False,
            )
        )
        assert len(coll) == 1
        assert coll.items[0].id == db_with_datasets.dataset_ids["v10"]


class TestLimit:
    def test_limit_applied_after_latest_only(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6), limit=1)
        assert len(coll) == 1


class TestIncludeFiles:
    def test_include_files_true_populates_files(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            ),
            include_files=True,
        )
        assert len(coll) == 1
        ds = coll.items[0]
        assert len(ds.files) == 1
        assert ds.files[0].path == "tas_v10.nc"
        assert ds.files[0].start_time == "2000-01-01"
        assert ds.files[0].end_time == "2000-12-31"
        assert ds.files[0].tracking_id == "hdl:test-tracking-id"

    def test_include_files_false_leaves_files_empty(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            ),
            include_files=False,
        )
        assert len(coll) == 1
        assert coll.items[0].files == ()


class TestDetachment:
    def test_dto_fields_survive_session_close(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("tas",)},
            ),
            include_files=True,
        )
        db_with_datasets.session.expunge_all()

        assert len(coll) == 1
        ds = coll.items[0]
        assert ds.slug == "CMIP.TEST.TEST-MODEL.historical.Amon.tas.gn.v10"
        assert ds.facets["variable_id"] == "tas"
        assert len(ds.files) == 1


class TestToPandas:
    def test_to_pandas_columns(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False))
        df = coll.to_pandas()
        for col in ("id", "slug", "dataset_type", "finalised", "created_at", "updated_at"):
            assert col in df.columns
        assert len(df) == len(coll)

    def test_to_pandas_columns_when_empty(self, db_with_datasets):
        """An empty collection still emits the base columns, so callers can select on them."""
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("NO-SUCH-MODEL",)})
        )
        assert len(coll) == 0
        df = coll.to_pandas()
        assert len(df) == 0
        for col in ("id", "slug", "dataset_type", "finalised", "created_at", "updated_at"):
            assert col in df.columns


class TestCollectionContract:
    def test_shape_fields(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False))
        assert isinstance(coll.items, tuple)
        assert coll.offset == 0
        assert coll.limit is None
        assert coll.total_count == len(coll.items)
        assert list(coll) == list(coll.items)

    def test_total_count_reflects_dedup(self, db_with_datasets):
        """``total_count`` is the deduplicated count, not the every-version count."""
        reader = Reader(db_with_datasets)
        deduped = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("TEST-MODEL",)})
        )
        # Only v10 survives its group, so the count is 1 even though v2 exists in the DB.
        assert deduped.total_count == 1


class TestPagination:
    def test_total_count_exceeds_page(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        coll = reader.datasets.list(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False), limit=1
        )
        assert len(coll) == 1
        assert coll.limit == 1
        assert coll.total_count > 1

    def test_offset_limit_paginate_all_rows_without_overlap(self, db_with_datasets):
        """Walking the pages with offset/limit yields every row exactly once, deterministically."""
        reader = Reader(db_with_datasets)
        full = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False))
        total = full.total_count

        seen: list[int] = []
        page_size = 2
        for offset in range(0, total, page_size):
            page = reader.datasets.list(
                DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False),
                offset=offset,
                limit=page_size,
            )
            assert page.offset == offset
            seen.extend(d.id for d in page)

        assert len(seen) == total
        assert len(set(seen)) == total  # no duplicates or gaps across pages

    def test_paging_one_by_one_matches_full_order(self, db_with_datasets):
        """Paging one row at a time yields the same order as one full fetch.

        The deterministic ``(slug, id)`` ordering means the concatenation of single-row pages is
        identical to the ordered single fetch -- so a page boundary never reorders, drops, or
        repeats a row.
        """
        reader = Reader(db_with_datasets)
        full = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False))
        full_ids = [d.id for d in full]

        paged_ids: list[int] = []
        for offset in range(full.total_count):
            page = reader.datasets.list(
                DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False),
                offset=offset,
                limit=1,
            )
            paged_ids.append(page.items[0].id)

        assert paged_ids == full_ids


class TestGet:
    def test_get_hit(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        view = reader.datasets.get("CMIP.TEST.OTHER-MODEL.historical.Amon.pr.gn.v1")
        assert view is not None
        assert view.id == db_with_datasets.dataset_ids["unfinalised"]

    def test_get_miss(self, db_with_datasets):
        reader = Reader(db_with_datasets)
        assert reader.datasets.get("no-such-slug") is None

    def test_get_resolves_each_version_by_its_own_slug(self, db_with_datasets):
        """Each version has a distinct (globally unique) slug, so ``get`` resolves each precisely.

        ``get``'s ``version_key``-then-``id`` ordering is a defensive tiebreak; the unique-slug
        schema means a slug never maps to more than one row, so both versions are individually
        addressable.
        """
        reader = Reader(db_with_datasets)
        v2 = reader.datasets.get("CMIP.TEST.TEST-MODEL.historical.Amon.tas.gn.v2")
        v10 = reader.datasets.get("CMIP.TEST.TEST-MODEL.historical.Amon.tas.gn.v10")
        assert v2 is not None and v10 is not None
        assert v2.id == db_with_datasets.dataset_ids["v2"]
        assert v10.id == db_with_datasets.dataset_ids["v10"]
