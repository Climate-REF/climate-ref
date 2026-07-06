"""
Typed, detached read surface for datasets.

[DatasetsReader][climate_ref.results.datasets.DatasetsReader] is reached via
[Reader.datasets][climate_ref.results.values.Reader.datasets]. It executes the shared
[select_datasets][climate_ref.models.dataset_query.select_datasets] query (also used by
``adapter.load_catalog``) and maps the rows into detached DTOs that outlive the session.

The filter and query builder live in the models layer
([climate_ref.models.dataset_query][climate_ref.models.dataset_query]); ``DatasetFilter`` is
re-exported here so callers construct it from ``climate_ref.results``.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session, selectin_polymorphic, selectinload

from climate_ref.database import Database
from climate_ref.datasets import get_dataset_adapter
from climate_ref.models.dataset import Dataset
from climate_ref.models.dataset_query import DatasetFilter, select_datasets
from climate_ref_core.source_types import SourceDatasetType

__all__ = ["DatasetFilter", "select_datasets"]


@attrs.frozen(kw_only=True)
class DatasetFileView:
    """A single dataset file, detached from the ORM. Times are raw stored strings (no cftime)."""

    path: str
    """Path to the file."""

    start_time: str | None
    """Start of the file's time range, as a raw stored string."""

    end_time: str | None
    """End of the file's time range, as a raw stored string."""

    tracking_id: str | None
    """The file's tracking ID, when present."""


@attrs.frozen(kw_only=True)
class DatasetView:
    """A single dataset, detached from the ORM."""

    id: int
    """Primary key of the underlying ``Dataset`` row."""

    slug: str
    """The dataset's slug."""

    dataset_type: SourceDatasetType
    """The dataset's source type (e.g. CMIP6, obs4MIPs)."""

    finalised: bool
    """Whether the dataset was registered via the complete (netCDF-opening) parser."""

    created_at: Any
    """Timestamp the dataset was created."""

    updated_at: Any
    """Timestamp the dataset was last updated."""

    facets: Mapping[str, object]
    """Dataset-specific metadata, keyed by the adapter's ``dataset_specific_metadata`` fields."""

    files: tuple[DatasetFileView, ...]
    """The dataset's files; empty unless requested with ``include_files=True``."""


@attrs.frozen(kw_only=True)
class DatasetCollection:
    """
    An immutable page of datasets plus collection-level metadata.

    ``total_count`` is the number of datasets matching the filter *before* pagination, so a caller
    can tell there are more rows than the returned page. ``offset``/``limit`` echo back the
    pagination applied to produce ``items`` (``limit`` is ``None`` when the whole result was
    returned).
    """

    items: tuple[DatasetView, ...]
    """The datasets on this page."""

    total_count: int
    """Total datasets matching the filter before ``offset``/``limit``."""

    offset: int
    """Rows skipped before this page."""

    limit: int | None
    """Page size requested, or ``None`` when the whole result was returned."""

    def __iter__(self) -> Iterator[DatasetView]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def to_pandas(self) -> pd.DataFrame:
        """
        DataFrame with one row per dataset; columns are base fields plus the facet dict expanded.

        The base columns (``id, slug, dataset_type, finalised, created_at, updated_at``) are
        emitted explicitly even when the collection is empty, so callers can select columns /
        build an empty table without special-casing. Facet columns are dynamic (they depend on
        the source type queried) and so are only present when at least one row has them.
        """
        base_columns = ["id", "slug", "dataset_type", "finalised", "created_at", "updated_at"]
        records = []
        for ds in self.items:
            rec: dict[str, Any] = {
                "id": ds.id,
                "slug": ds.slug,
                "dataset_type": ds.dataset_type.value,
                "finalised": ds.finalised,
                "created_at": ds.created_at,
                "updated_at": ds.updated_at,
            }
            rec.update(ds.facets)
            records.append(rec)
        return pd.DataFrame.from_records(records, columns=base_columns if not records else None)


def _dataset_subtypes() -> list[type[Dataset]]:
    """Every concrete ``Dataset`` subclass, for polymorphic eager-loading."""
    return [m.class_ for m in Dataset.__mapper__.polymorphic_map.values()]


class DatasetsReader:
    """
    Dataset read domain.

    Constructed from a [Database][climate_ref.database.Database], which owns the session.

    All read methods return detached DTOs that outlive the session.

    ``list`` requires a ``DatasetFilter`` (with its required ``source_type``), so unlike the other
    readers -- whose ``filters`` argument is optional and defaults to "everything" -- there is no
    useful all-datasets default here. This is a deliberate, documented divergence from that shared
    contract: dataset facet columns are per-type, so a typed listing has to choose the type. ``get``
    keeps taking a bare slug, which is globally unique and needs no ``source_type``.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._db.session

    def _to_view(self, dataset: Dataset, *, include_files: bool) -> DatasetView:
        adapter = get_dataset_adapter(dataset.dataset_type.value)
        facets = {k: getattr(dataset, k) for k in adapter.dataset_specific_metadata if hasattr(dataset, k)}
        files = (
            tuple(
                DatasetFileView(
                    path=f.path, start_time=f.start_time, end_time=f.end_time, tracking_id=f.tracking_id
                )
                for f in dataset.files  # type: ignore[attr-defined]
            )
            if include_files
            else ()
        )
        return DatasetView(
            id=dataset.id,
            slug=dataset.slug,
            dataset_type=dataset.dataset_type,
            finalised=dataset.finalised,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            facets=facets,
            files=files,
        )

    def _base_statement(self, filter: DatasetFilter) -> Select[Any]:  # noqa: A002
        """
        Build the (unpaginated, unordered) ``SELECT`` over the concrete entity for the filter.

        Deduplication to the latest version happens in SQL via a ``RANK`` window keyed off the
        concrete adapter's ``dataset_id_metadata`` (inert when ``filter.latest_only`` is ``False``).
        """
        adapter = get_dataset_adapter(filter.source_type.value)
        return select_datasets(filter, latest_group_by=adapter.dataset_id_metadata)

    def list(
        self,
        filters: DatasetFilter,
        *,
        offset: int = 0,
        limit: int | None = None,
        include_files: bool = False,
    ) -> DatasetCollection:
        """
        Query one source type's datasets, optionally scoped to an execution or diagnostic.

        ``filters`` is required (its ``source_type`` picks the concrete type and hence the facet
        columns). Deduplication to the latest version (``filters.latest_only``, the default) happens
        in SQL via a ``RANK`` window keyed off the adapter's ``dataset_id_metadata``, so the default
        call returns exactly one row per dataset rather than every version.

        Pagination (``offset``/``limit``) is applied in SQL after dedup, over rows ordered by
        ``(slug, id)`` so paging is deterministic even when two datasets share a slug. ``total_count``
        is computed from a separate unpaged count over the same deduplicated statement.
        """
        base_stmt = self._base_statement(filters)
        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        total_count = self.session.execute(count_stmt).scalar_one()

        stmt = base_stmt.order_by(None).order_by(Dataset.slug, Dataset.id)
        if include_files:
            stmt = stmt.options(selectinload(Dataset.files))  # type: ignore[attr-defined]
        if limit is not None:
            stmt = stmt.offset(offset).limit(limit)
        elif offset:
            stmt = stmt.offset(offset)

        rows = list(self.session.execute(stmt).scalars().unique().all())
        items = tuple(self._to_view(r, include_files=include_files) for r in rows)

        return DatasetCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def get(self, slug: str) -> DatasetView | None:
        """
        Fetch one dataset by slug, or ``None`` when no dataset has that slug.

        When multiple rows share a slug (different versions), the latest is returned, ranked by
        ``version_key`` then ``id`` so the choice is deterministic on a version tie.
        """
        stmt = (
            select(Dataset)
            .where(Dataset.slug == slug)
            .order_by(Dataset.version_key.desc(), Dataset.id.desc())
            .options(selectin_polymorphic(Dataset, _dataset_subtypes()))
        )
        dataset = self.session.execute(stmt).scalars().first()
        return self._to_view(dataset, include_files=False) if dataset is not None else None
