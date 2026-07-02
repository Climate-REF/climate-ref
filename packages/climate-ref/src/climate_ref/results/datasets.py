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
    start_time: str | None
    end_time: str | None
    tracking_id: str | None


@attrs.frozen(kw_only=True)
class DatasetView:
    """A single dataset, detached from the ORM."""

    id: int
    slug: str
    dataset_type: SourceDatasetType
    finalised: bool
    created_at: Any
    updated_at: Any
    facets: Mapping[str, object]
    files: tuple[DatasetFileView, ...]


@attrs.frozen(kw_only=True)
class DatasetCollection:
    """An immutable collection of datasets."""

    datasets: tuple[DatasetView, ...]

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

        ``limit`` (when given) is applied after ``latest_only`` filtering, over the ordered, latest
        datasets, so it caps returned datasets -- matching ``adapter.load_catalog``'s dedup-then-limit
        ordering.
        """
        filter = filter or DatasetFilter()  # noqa: A001

        entity: type[Dataset] = (
            get_dataset_adapter(filter.source_type.value).dataset_cls if filter.source_type else Dataset
        )
        stmt = select_datasets(filter)
        if include_files:
            stmt = stmt.options(selectinload(entity.files))  # type: ignore[attr-defined]

        session = self._db.session
        rows = list(session.execute(stmt).scalars().unique().all())

        adapter = get_dataset_adapter(filter.source_type.value) if filter.source_type else None
        if filter.latest_only and adapter is not None and adapter.dataset_id_metadata:
            id_cols = adapter.dataset_id_metadata
            version_col = adapter.version_metadata
            # Reuse the adapter's own latest-version policy (its short-circuits + numeric compare) so
            # there is a single definition of "latest version": feed it a minimal id/version frame
            # indexed by dataset id and keep the survivors.
            versions = pd.DataFrame(
                [{**{c: getattr(r, c) for c in id_cols}, version_col: getattr(r, version_col)} for r in rows],
                index=[r.id for r in rows],
            )
            surviving_ids = set(adapter.filter_latest_versions(versions).index)
            rows = [r for r in rows if r.id in surviving_ids]

        if limit is not None:
            rows = rows[:limit]

        return DatasetCollection(datasets=tuple(self._to_view(r, include_files=include_files) for r in rows))
