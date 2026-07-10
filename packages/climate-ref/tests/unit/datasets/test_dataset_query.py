"""Tests for the SQL-side latest-version dedup in ``select_datasets``.

Covers:
1. ``select_datasets(..., latest_group_by=...)`` returns the same survivor set as the old pandas
   ``filter_latest_versions`` path (numeric ties, non-conforming versions, combined with
   finalised/facet/relationship filters -- including ``diagnostic_slug`` + ``latest_only`` together).
2. The non-nullability invariant that the SQL dedup depends on: every adapter's
   ``dataset_id_metadata`` columns are non-nullable and absent from ``columns_requiring_finalisation``.
"""

import datetime

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from climate_ref.datasets import get_dataset_adapter
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.cmip7 import CMIP7DatasetAdapter
from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter
from climate_ref.models.dataset import CMIP6Dataset, Dataset, DatasetFile
from climate_ref.models.dataset_query import DatasetFilter, select_datasets
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup, execution_datasets
from climate_ref_core.source_types import SourceDatasetType


def _make_cmip6(
    *,
    slug: str,
    source_id: str = "TEST-MODEL",
    variable_id: str = "tas",
    version: str = "v1",
    finalised: bool = True,
) -> CMIP6Dataset:
    return CMIP6Dataset(
        slug=slug,
        dataset_type=SourceDatasetType.CMIP6,
        activity_id="CMIP",
        experiment_id="historical",
        institution_id="TEST",
        source_id=source_id,
        member_id="r1i1p1f1",
        table_id="Amon",
        variable_id=variable_id,
        grid_label="gn",
        version=version,
        instance_id=slug,
        variant_label="r1i1p1f1",
        finalised=finalised,
    )


@pytest.fixture
def db_with_versions(db_seeded):
    """Add controlled version variety on top of the seeded DB: v2/v10 ties and a non-conforming version."""
    with db_seeded.session.begin():
        v2 = _make_cmip6(slug="dq.TEST-MODEL.tas.v2", version="v2")
        v10 = _make_cmip6(slug="dq.TEST-MODEL.tas.v10", version="v10")
        nonconforming = _make_cmip6(slug="dq.TEST-MODEL.pr.latest", variable_id="pr", version="latest")
        unfinalised_v1 = _make_cmip6(
            slug="dq.OTHER-MODEL.tas.v1", source_id="OTHER-MODEL", version="v1", finalised=False
        )
        db_seeded.session.add_all([v2, v10, nonconforming, unfinalised_v1])
        db_seeded.session.flush()

        db_seeded.session.add(
            DatasetFile(dataset_id=v10.id, path="dq_v10.nc", start_time="2000-01-01", end_time="2000-12-31")
        )

        diag = db_seeded.session.query(Diagnostic).first()
        eg = ExecutionGroup(key="dq-key", diagnostic_id=diag.id, selectors={})
        db_seeded.session.add(eg)
        db_seeded.session.flush()
        execution = Execution(
            execution_group_id=eg.id, successful=True, output_fragment="dq-out", dataset_hash="dq-hash"
        )
        db_seeded.session.add(execution)
        db_seeded.session.flush()
        db_seeded.session.execute(
            execution_datasets.insert(), [{"execution_id": execution.id, "dataset_id": v10.id}]
        )

    db_seeded.session.commit()
    db_seeded.dq_ids = {
        "v2": v2.id,
        "v10": v10.id,
        "nonconforming": nonconforming.id,
        "unfinalised_v1": unfinalised_v1.id,
    }
    db_seeded.dq_execution_id = execution.id
    db_seeded.dq_diagnostic_slug = diag.slug
    return db_seeded


ID_COLS = CMIP6DatasetAdapter.dataset_id_metadata


class TestSelectDatasetsLatestVersionDedup:
    def test_numeric_tie_keeps_v10_over_v2(self, db_with_versions):
        stmt = select_datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("TEST-MODEL",)}),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        ids = {r.id for r in rows}
        assert db_with_versions.dq_ids["v10"] in ids
        assert db_with_versions.dq_ids["v2"] not in ids

    def test_ties_at_max_version_both_survive(self, db_with_versions):
        """RANK (not ROW_NUMBER): two rows tied at the max ``version_key`` in a partition both survive.

        Pins the tie behaviour at the SQL layer so a future ``rank()`` -> ``row_number()`` regression is
        caught in CI, not only by the pandas parity tests.
        """
        tie_a = _make_cmip6(slug="dq.TIE-MODEL.tas.v10.a", source_id="TIE-MODEL", version="v10")
        tie_b = _make_cmip6(slug="dq.TIE-MODEL.tas.v10.b", source_id="TIE-MODEL", version="v10")
        older = _make_cmip6(slug="dq.TIE-MODEL.tas.v2", source_id="TIE-MODEL", version="v2")
        db_with_versions.session.add_all([tie_a, tie_b, older])
        db_with_versions.session.flush()
        tie_ids = {tie_a.id, tie_b.id}

        stmt = select_datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("TIE-MODEL",)}),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        assert {r.id for r in rows} == tie_ids

    def test_non_conforming_version_survives_alone_in_its_group(self, db_with_versions):
        stmt = select_datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",), "variable_id": ("pr",)},
            ),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        assert {r.id for r in rows} == {db_with_versions.dq_ids["nonconforming"]}

    def test_combined_with_finalised_filter(self, db_with_versions):
        stmt = select_datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, finalised=False),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        assert {r.id for r in rows} == {db_with_versions.dq_ids["unfinalised_v1"]}

    def test_combined_with_execution_id_and_diagnostic_slug(self, db_with_versions):
        """diagnostic_slug + latest_only together must not emit invalid SQL (duplicate self-join)."""
        stmt = select_datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                execution_id=db_with_versions.dq_execution_id,
                diagnostic_slug=db_with_versions.dq_diagnostic_slug,
            ),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        assert {r.id for r in rows} == {db_with_versions.dq_ids["v10"]}

    def test_selectinload_files_still_works(self, db_with_versions):
        """The ``id IN (subquery)`` shape keeps the outer entity a plain class, so class-bound
        loader options attached by callers (e.g. ``load_catalog``, ``reader.datasets``) still work."""
        stmt = select_datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("TEST-MODEL",)}),
            latest_group_by=ID_COLS,
        ).options(selectinload(CMIP6Dataset.files))
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        v10 = next(r for r in rows if r.id == db_with_versions.dq_ids["v10"])
        assert [f.path for f in v10.files] == ["dq_v10.nc"]

    def test_latest_only_false_ignores_latest_group_by(self, db_with_versions):
        stmt = select_datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",)},
                latest_only=False,
            ),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        ids = {r.id for r in rows}
        assert db_with_versions.dq_ids["v2"] in ids
        assert db_with_versions.dq_ids["v10"] in ids

    def test_latest_only_requires_latest_group_by(self):
        """``latest_only=True`` (the default) with no ``latest_group_by`` is rejected: the two must be
        supplied together, so a caller cannot silently receive an un-deduplicated result."""
        with pytest.raises(ValueError, match="latest_group_by"):
            select_datasets(
                DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("TEST-MODEL",)})
            )

    def test_source_type_is_required(self):
        """``DatasetFilter`` has no default ``source_type``: a typed listing must choose a type."""
        with pytest.raises(TypeError):
            DatasetFilter()  # type: ignore[call-arg]


class TestRetractedFiltering:
    """``DatasetFilter.include_retracted`` (default ``False``) excludes retracted rows."""

    def _retract(self, db, dataset_id: int) -> None:
        # ``retracted_at`` lives on the base ``dataset`` table.
        # A Core ``update(CMIP6Dataset)`` cannot reach a parent-table column on SQLite
        # (no multi-table UPDATE support), so set it through the ORM instance instead.
        ds = db.session.get(Dataset, dataset_id)
        ds.retracted_at = datetime.datetime(2026, 1, 1)
        db.session.commit()

    def test_default_excludes_retracted(self, db_with_versions):
        """Retracting the latest version removes it from the default (exclude-retracted) query, and
        the next-oldest version becomes the new survivor of the latest-version window -- the
        retracted-filter is applied before the ``RANK`` partition, not after."""
        self._retract(db_with_versions, db_with_versions.dq_ids["v10"])

        stmt = select_datasets(
            DatasetFilter(source_type=SourceDatasetType.CMIP6, facets={"source_id": ("TEST-MODEL",)}),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        ids = {r.id for r in rows}
        assert db_with_versions.dq_ids["v10"] not in ids
        assert db_with_versions.dq_ids["v2"] in ids

    def test_include_retracted_true_shows_it(self, db_with_versions):
        self._retract(db_with_versions, db_with_versions.dq_ids["v10"])

        stmt = select_datasets(
            DatasetFilter(
                source_type=SourceDatasetType.CMIP6,
                facets={"source_id": ("TEST-MODEL",)},
                include_retracted=True,
            ),
            latest_group_by=ID_COLS,
        )
        rows = db_with_versions.session.execute(stmt).scalars().unique().all()
        assert db_with_versions.dq_ids["v10"] in {r.id for r in rows}

    def test_retracted_dataset_execution_link_survives(self, db_with_versions):
        """Retracting a dataset must not touch its ``execution_datasets`` links -- only its
        eligibility for future solve-time selection changes."""
        v10_id = db_with_versions.dq_ids["v10"]
        self._retract(db_with_versions, v10_id)

        linked = (
            db_with_versions.session.execute(
                select(execution_datasets.c.dataset_id).where(
                    execution_datasets.c.execution_id == db_with_versions.dq_execution_id
                )
            )
            .scalars()
            .all()
        )
        assert v10_id in linked


class TestDatasetIdMetadataNonNullabilityInvariant:
    """Pin the assumption the SQL dedup depends on: every adapter's ``dataset_id_metadata`` column
    is non-nullable on the model AND absent from ``columns_requiring_finalisation``.

    An actual NULL insert would raise ``IntegrityError`` (NOT NULL constraint), so this is pinned
    structurally instead of by attempting a NULL insert.
    """

    @pytest.mark.parametrize(
        "adapter_cls",
        [CMIP6DatasetAdapter, CMIP7DatasetAdapter, Obs4MIPsDatasetAdapter, PMPClimatologyDatasetAdapter],
    )
    def test_dataset_id_metadata_columns_are_non_nullable(self, adapter_cls):
        adapter = adapter_cls()
        entity: type[Dataset] = adapter.dataset_cls
        for column_name in adapter.dataset_id_metadata:
            column = entity.__mapper__.columns[column_name]
            assert not column.nullable, (
                f"{entity.__name__}.{column_name} is nullable but is a dataset_id_metadata "
                "partition column for the SQL latest-version window; a NULL value would silently "
                "break RANK partitioning."
            )

    @pytest.mark.parametrize(
        "adapter_cls",
        [CMIP6DatasetAdapter, CMIP7DatasetAdapter, Obs4MIPsDatasetAdapter, PMPClimatologyDatasetAdapter],
    )
    def test_dataset_id_metadata_disjoint_from_finalisation_columns(self, adapter_cls):
        adapter = adapter_cls()
        overlap = set(adapter.dataset_id_metadata) & adapter.columns_requiring_finalisation
        assert not overlap, (
            f"{adapter_cls.__name__}: dataset_id_metadata columns {overlap} require finalisation, "
            "so they may be NA pre-finalisation, which would break the SQL latest-version window."
        )


class TestVersionKeyOrmOnlyInvariant:
    """Pin the documented ORM-only invariant for ``version_key``.

    The ``_sync_version_key`` mapper event only fires on ORM inserts/updates. A Core-level
    ``UPDATE`` to ``version`` bypasses it, leaving ``version_key`` stale. This is a known
    limitation documented on the ``version``/``version_key`` column docstrings; the test
    exists to pin that behaviour so a future change is a conscious one.
    """

    def test_orm_insert_syncs_version_key(self, db_seeded):
        """Baseline: an ORM insert does keep ``version_key`` in sync."""
        with db_seeded.session.begin():
            ds = _make_cmip6(slug="vk.orm-insert.v10", version="v10")
            db_seeded.session.add(ds)
            db_seeded.session.flush()
            assert ds.version_key == 10

    def test_core_update_leaves_version_key_stale(self, db_seeded):
        """A Core-level ``UPDATE`` to ``version`` does NOT re-sync ``version_key``.

        This pins the known limitation: the mapper event never fires for a Core update, so
        ``version_key`` keeps its old value (2) even though ``version`` is now ``v10``.
        """
        with db_seeded.session.begin():
            ds = _make_cmip6(slug="vk.core-update.v2", version="v2")
            db_seeded.session.add(ds)
            db_seeded.session.flush()
            assert ds.version_key == 2
            ds_id = ds.id

        # Core-level UPDATE on the subclass table, bypassing the ORM mapper event.
        with db_seeded.session.begin():
            db_seeded.session.execute(
                update(CMIP6Dataset).where(CMIP6Dataset.id == ds_id).values(version="v10")
            )

        db_seeded.session.expire_all()
        refreshed = db_seeded.session.get(CMIP6Dataset, ds_id)
        assert refreshed.version == "v10"
        # version_key is stale: still 2, not the 10 an ORM write would have produced.
        assert refreshed.version_key == 2


def test_get_dataset_adapter_covers_all_four_subclasses():
    """Sanity check that all four Dataset subclasses are reachable via the adapter registry
    (keeps the parametrised tests above from silently under-covering a fifth subclass added later)."""
    for source_type in (
        SourceDatasetType.CMIP6,
        SourceDatasetType.CMIP7,
        SourceDatasetType.obs4MIPs,
        SourceDatasetType.PMPClimatology,
    ):
        adapter = get_dataset_adapter(source_type.value)
        assert adapter.dataset_id_metadata
