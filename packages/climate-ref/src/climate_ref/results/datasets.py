"""
Which datasets are in the database?

Use this to find the model, observation or reanalysis datasets that REF has ingested,
for example all CMIP6 datasets for a given model, or the files of one dataset.
[DatasetsReader][climate_ref.results.datasets.DatasetsReader] is reached via
[Reader.datasets][climate_ref.results.reader.Reader.datasets].
It lists one source type at a time (CMIP6, CMIP7, obs4MIPs and so on).

It returns:

* [DatasetCollection][climate_ref.results.datasets.DatasetCollection]: a list of datasets,
  with ``to_pandas()`` for a table.
* [DatasetView][climate_ref.results.datasets.DatasetView]: one dataset.
* [DatasetFileView][climate_ref.results.datasets.DatasetFileView]: one file of a dataset.

``DatasetFilter`` chooses which datasets to keep.
Import it from ``climate_ref.results``.
It filters datasets rather than executions,
and shares only the diagnostic and provider fields with the other filters in
[climate_ref.query.filters][climate_ref.query.filters].

Examples
--------
All CMIP6 datasets in the database, as a table:

```python
from climate_ref.results import DatasetFilter, Reader
from climate_ref_core.source_types import SourceDatasetType

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    cmip6 = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6))
    df = cmip6.to_pandas()
```
"""

import functools
from collections.abc import Mapping
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectin_polymorphic, selectinload

from climate_ref.database import Database
from climate_ref.datasets import get_dataset_adapter
from climate_ref.models.dataset import Dataset
from climate_ref.query import DatasetFilter, count, paginate, select_datasets
from climate_ref.results._page import Page
from climate_ref_core.source_types import SourceDatasetType


@attrs.frozen(kw_only=True)
class DatasetFileView:
    """
    One netCDF file of a dataset.

    Times are the strings stored in the database, not parsed dates.
    Get these from ``DatasetView.files``, which is only filled by ``list(..., include_files=True)``.

    Examples
    --------
    The files of every CMIP6 ``tas`` dataset for ACCESS-CM2:

    ```python
    filters = DatasetFilter(
        source_type=SourceDatasetType.CMIP6,
        facets={"source_id": "ACCESS-CM2", "variable_id": "tas"},
    )
    for dataset in reader.datasets.list(filters, include_files=True):
        for file in dataset.files:
            print(file.path, file.start_time, file.end_time)
    ```
    """

    path: str
    """Path to the file on disk."""

    start_time: str | None
    """Start of the file's time range, as the text stored in the database."""

    end_time: str | None
    """End of the file's time range, as the text stored in the database."""

    tracking_id: str | None
    """The file's ``tracking_id`` global attribute, when present."""


@attrs.frozen(kw_only=True)
class DatasetView:
    """
    One dataset, such as one CMIP6 model, experiment, member, table, variable and grid at one version.

    Returned by [DatasetsReader.get][climate_ref.results.datasets.DatasetsReader.get]
    and held in a [DatasetCollection][climate_ref.results.datasets.DatasetCollection].

    Examples
    --------
    Look up one dataset and print its model and experiment:

    ```python
    with Database.from_config(config, read_only=True) as db:
        dataset = Reader(db).datasets.get(slug)
        if dataset is not None:
            print(dataset.facets["source_id"], dataset.facets["experiment_id"])
    ```
    """

    id: int
    """The dataset's id in the database."""

    slug: str
    """
    The dataset's unique name.

    For example ``"CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Omon.tos.gn.v20210318"``.
    """

    dataset_type: SourceDatasetType
    """The dataset's source type, e.g. ``SourceDatasetType.CMIP6`` or ``SourceDatasetType.obs4MIPs``."""

    finalised: bool
    """
    ``True`` if REF read the metadata from inside every file.

    ``False`` means it used only the file and folder names.
    """

    retracted_at: Any
    """When the dataset was retracted, or ``None`` if it has not been."""

    created_at: Any
    """When the dataset was first ingested, as a ``datetime``."""

    updated_at: Any
    """When the dataset's record was last changed, as a ``datetime``."""

    facets: Mapping[str, object]
    """
    The dataset's facets by name, e.g. ``source_id``, ``experiment_id`` and ``member_id`` for CMIP6.

    The names depend on ``dataset_type``.
    """

    files: tuple[DatasetFileView, ...]
    """The dataset's files. This is empty unless the dataset came from ``list(..., include_files=True)``."""


@attrs.frozen(kw_only=True)
class DatasetCollection(Page[DatasetView]):
    """
    The datasets that ``reader.datasets.list`` found.

    Loop over it, or use ``len()``, to get the [DatasetView][climate_ref.results.datasets.DatasetView]
    objects on this page.
    Call ``to_pandas()`` to get a table.
    ``total_count`` is the number of matching datasets before ``offset`` and ``limit``.

    Examples
    --------
    The first 20 CMIP6 datasets, and how many there are in all:

    ```python
    with Database.from_config(config, read_only=True) as db:
        page = Reader(db).datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6), limit=20)
        print(len(page), "of", page.total_count)
        for dataset in page:
            print(dataset.slug)
    ```
    """

    def to_pandas(self) -> pd.DataFrame:
        """
        Put the datasets in a pandas table with one row per dataset.

        The columns are ``id``, ``slug``, ``dataset_type``, ``finalised``, ``retracted_at``,
        ``created_at`` and ``updated_at``, followed by one column per facet.
        ``dataset_type`` holds the string value, such as ``"cmip6"``.
        An empty collection still has the base columns, so you can select from it without a special case.
        Facet columns depend on the source type, so an empty collection has none.

        Examples
        --------
        Pick out the model and experiment of each dataset:

        ```python
        df = page.to_pandas()
        df[["slug", "source_id", "experiment_id"]]
        ```
        """
        base_columns = ["id", "slug", "dataset_type", "finalised", "retracted_at", "created_at", "updated_at"]
        records = []
        for ds in self.items:
            rec: dict[str, Any] = {
                "id": ds.id,
                "slug": ds.slug,
                "dataset_type": ds.dataset_type.value,
                "finalised": ds.finalised,
                "retracted_at": ds.retracted_at,
                "created_at": ds.created_at,
                "updated_at": ds.updated_at,
            }
            rec.update(ds.facets)
            records.append(rec)
        return pd.DataFrame.from_records(records, columns=base_columns if not records else None)


@functools.cache
def _dataset_specific_metadata(source_type: str) -> tuple[str, ...]:
    # Building an adapter loads the default config, too slow to repeat for every row.
    return tuple(get_dataset_adapter(source_type).dataset_specific_metadata)


def _dataset_subtypes() -> list[type[Dataset]]:
    """Every concrete ``Dataset`` subclass, for polymorphic eager-loading."""
    return [m.class_ for m in Dataset.__mapper__.polymorphic_map.values()]


class DatasetsReader:
    """
    Which datasets has REF ingested?

    Reach it as ``Reader(db).datasets``.
    Use ``list`` to search by source type and facets, and ``get`` to fetch one dataset by its slug.
    The results can be used after the database is closed.

    ``list`` needs a ``DatasetFilter`` with a ``source_type``, unlike the other readers,
    whose ``filters`` argument is optional.
    Each source type has different facets, so a listing has to choose one.
    ``get`` takes only a slug, because a slug identifies a dataset of any source type.

    Examples
    --------
    Describe which datasets you want with a [DatasetFilter][climate_ref.query.filters.DatasetFilter].
    Only ``source_type`` is required, and this does not need a database.
    A single facet value can be a string, and several values are alternatives:

    >>> from climate_ref.results import DatasetFilter
    >>> from climate_ref_core.source_types import SourceDatasetType
    >>> filters = DatasetFilter(
    ...     source_type=SourceDatasetType.CMIP6,
    ...     facets={"source_id": "ACCESS-ESM1-5", "experiment_id": ["historical", "ssp126"]},
    ...     slug_contains="Omon",
    ... )
    >>> filters.latest_only
    True
    >>> filters.facets
    {'source_id': ('ACCESS-ESM1-5',), 'experiment_id': ('historical', 'ssp126')}

    By default ``latest_only`` is ``True``, so you get the newest version of each dataset.
    Pass the filter to ``list`` to get the matching datasets and their files:

    ```python
    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        page = reader.datasets.list(filters, limit=50, include_files=True)
        for dataset in page:
            print(dataset.slug, [f.path for f in dataset.files])
    ```

    Every version of each dataset, including retracted ones (``include_retracted``):

    ```python
    everything = DatasetFilter(source_type=SourceDatasetType.CMIP6, latest_only=False, include_retracted=True)
    ```

    The datasets that execution 42 used:

    ```python
    used = reader.datasets.list(DatasetFilter(source_type=SourceDatasetType.CMIP6, execution_ids=42))
    ```
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._session if self._session is not None else self._db.session

    def _to_view(self, dataset: Dataset, *, include_files: bool) -> DatasetView:
        fields = _dataset_specific_metadata(dataset.dataset_type.value)
        facets = {k: getattr(dataset, k) for k in fields if hasattr(dataset, k)}
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
            retracted_at=dataset.retracted_at,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            facets=facets,
            files=files,
        )

    def _base_statement(self, filter: DatasetFilter) -> Select[Any]:  # noqa: A002
        """
        Build the unpaginated ``SELECT`` over the dataset class for the filter's source type.

        Keeping only the latest version is done in SQL, partitioned by the adapter's ``dataset_id_metadata``.
        It does nothing when ``filter.latest_only`` is ``False``.
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
        with_count: bool = True,
    ) -> DatasetCollection:
        """
        Which datasets of one source type are in the database?

        The ``source_type`` of ``filters`` picks the type, and so the facets you can filter on.
        By default only the latest version of each dataset is returned.
        Set ``latest_only=False`` on the filter to get every version.

        Datasets are sorted by slug, so paging is stable.
        ``offset`` and ``limit`` apply after older versions are dropped.

        Parameters
        ----------
        filters
            Which datasets to keep, e.g. a source type plus ``facets={"source_id": "ACCESS-CM2"}``.
            Only ``source_type`` is required.
        offset
            Number of datasets to skip.
        limit
            Maximum number of datasets to return, or ``None`` for all of them.
        include_files
            Also fill ``DatasetView.files`` with each dataset's files.
        with_count
            Fill ``total_count`` with the number of matches before ``offset`` and ``limit``.
            Pass ``False`` to skip counting on a large database, which leaves ``total_count`` as ``None``.

        Returns
        -------
        DatasetCollection
            The datasets on this page.

        Raises
        ------
        ValueError
            If a name in ``filters.facets`` is not a facet of the source type.

        Examples
        --------
        Which CMIP6 ``tas`` datasets for ACCESS-CM2 are in the database?

        ```python
        with Database.from_config(config, read_only=True) as db:
            found = Reader(db).datasets.list(
                DatasetFilter(
                    source_type=SourceDatasetType.CMIP6,
                    facets={"source_id": "ACCESS-CM2", "variable_id": "tas"},
                )
            )
            df = found.to_pandas()
            df[["experiment_id", "member_id", "version"]]
        ```
        """
        base_stmt = self._base_statement(filters)
        total_count = count(self.session, base_stmt) if with_count else None

        stmt = base_stmt.order_by(None).order_by(Dataset.slug, Dataset.id)
        if include_files:
            stmt = stmt.options(selectinload(Dataset.files))  # type: ignore[attr-defined]

        rows = list(self.session.execute(paginate(stmt, offset, limit)).scalars().unique().all())
        items = tuple(self._to_view(r, include_files=include_files) for r in rows)

        return DatasetCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def get(self, slug: str) -> DatasetView | None:
        """
        Is this dataset in the database?

        Returns the dataset with this slug, or ``None`` when there is none.
        If several versions share the slug, the newest is returned.
        The files are not loaded, so ``DatasetView.files`` is empty.
        Use ``list(..., include_files=True)`` when you need them.

        Parameters
        ----------
        slug
            The dataset's unique name.

        Returns
        -------
        DatasetView or None
            The dataset, or ``None`` if no dataset has that slug.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            dataset = Reader(db).datasets.get(
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Omon.tos.gn.v20210318"
            )
        ```
        """
        stmt = (
            select(Dataset)
            .where(Dataset.slug == slug)
            .order_by(Dataset.version_key.desc(), Dataset.id.desc())
            .options(selectin_polymorphic(Dataset, _dataset_subtypes()))
        )
        dataset = self.session.execute(stmt).scalars().first()
        return self._to_view(dataset, include_files=False) if dataset is not None else None
