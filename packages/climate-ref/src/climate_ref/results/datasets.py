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
from sqlalchemy.orm import selectinload

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
    """An immutable collection of datasets."""

    datasets: tuple[DatasetView, ...]
    """The datasets in this collection."""

    def __iter__(self) -> Iterator[DatasetView]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)

    def to_pandas(self) -> pd.DataFrame:
        """DataFrame with one row per dataset; columns are base fields plus the facet dict expanded."""
        records = []
        for ds in self.datasets:
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
        return pd.DataFrame.from_records(records)


class DatasetsReader:
    """
    Dataset read domain.

    Constructed from a [Database][climate_ref.database.Database], which owns the session.

    All read methods return detached DTOs that outlive the session.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

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

    def datasets(
        self,
        filter: DatasetFilter | None = None,  # noqa: A002
        *,
        limit: int | None = None,
        include_files: bool = False,
    ) -> DatasetCollection:
        """
        Query datasets, optionally scoped to a source type, execution or diagnostic.

        Deduplication to the latest version (when ``filter.latest_only``) happens in SQL via a
        ``RANK`` window, keyed off the adapter's ``dataset_id_metadata``. ``limit`` is pushed into
        the same statement, so it is applied after dedup, over the ordered, latest datasets --
        matching ``adapter.load_catalog``'s dedup-then-limit ordering, and only fetching the rows
        actually returned instead of the whole table.
        """
        filter = filter or DatasetFilter()  # noqa: A001

        adapter = get_dataset_adapter(filter.source_type.value) if filter.source_type else None
        entity: type[Dataset] = adapter.dataset_cls if adapter else Dataset

        latest_group_by = adapter.dataset_id_metadata if adapter is not None else None
        stmt = select_datasets(filter, latest_group_by=latest_group_by)
        if include_files:
            stmt = stmt.options(selectinload(entity.files))  # type: ignore[attr-defined]
        if limit is not None:
            stmt = stmt.limit(limit)

        session = self._db.session
        rows = list(session.execute(stmt).scalars().unique().all())

        return DatasetCollection(datasets=tuple(self._to_view(r, include_files=include_files) for r in rows))
