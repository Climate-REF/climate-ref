"""
Read metric values through ``Reader(db).values``.

A metric value is a number, or a series of numbers, that a diagnostic produced for a model or a reference.
Each value has labels such as ``source_id``, ``experiment_id``, ``metric`` and ``statistic``.
These labels are the CV dimensions, and the values present for each label are its facets.

[Reader][climate_ref.results.reader.Reader] wraps an open REF database.
Its [values][climate_ref.results.reader.Reader.values] reader,
[ValuesReader][climate_ref.results.values.ValuesReader], gets scalar and series values.
A read returns a collection that you can loop over or turn into a DataFrame with ``to_pandas()``.

Pass a [MetricValueFilter][climate_ref.query.filters.MetricValueFilter] to choose which values you get,
and an [OutlierPolicy][climate_ref.results.outliers.OutlierPolicy] to flag unusual scalar values.

Examples
--------
Get the scalar values and the time series of one diagnostic:

```python
from climate_ref.results import MetricValueFilter, Reader

filters = MetricValueFilter(provider_slug="example", diagnostic_slug="global-mean-timeseries")

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    scalars = reader.values.scalar_values(filters).to_pandas()
    series = reader.values.series_values(filters).to_pandas()
```
"""

from collections.abc import Mapping, Sequence
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session, joinedload

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, MetricValueType, ScalarMetricValue, SeriesMetricValue
from climate_ref.query import (
    MetricValueFilter,
    count,
    paginate,
    select_value_ids,
)
from climate_ref.results._page import Page, check_paging
from climate_ref.results.frames import collect_facets, scalar_values_to_frame, series_values_to_frame
from climate_ref.results.outliers import OutlierPolicy, select_outlier_flags


def _select_page(
    entity: type[MetricValue], value_ids: Select[Any], offset: int, limit: int | None
) -> Select[Any]:
    """
    Select a page of ``entity`` by paging ``value_ids`` first, then loading only those rows.

    SQLite would otherwise filter through ``ix_metric_value_type``, which reads every value of that type.
    """
    ids = value_ids.subquery("value_ids")
    page = paginate(select(ids.c.id).order_by(ids.c.id), offset, limit).subquery("value_page")
    return select(entity).where(entity.id.in_(select(page.c.id))).order_by(entity.id)


def _split_kind(dimensions: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """Split the model/reference ``kind`` (defaults to ``"model"``) from the other CV dimensions."""
    rest = dict(dimensions)
    return rest.pop("kind", "model"), rest


@attrs.frozen(kw_only=True)
class Facet:
    """
    The values that one label takes across a result, such as the models in it.

    Use these to see what you can filter on.

    Examples
    --------
    >>> from climate_ref.results.values import Facet
    >>> Facet(key="source_id", values=("CanESM5", "MIROC6")).values
    ('CanESM5', 'MIROC6')
    """

    key: str
    """Name of the label, such as ``"source_id"``."""

    values: tuple[str, ...]
    """The distinct values of the label, such as the ``source_id`` of each model."""


@attrs.frozen(kw_only=True)
class ScalarValue:
    """
    One number that a diagnostic produced, such as the bias of one model for one metric.

    Read with [ValuesReader.scalar_values][climate_ref.results.values.ValuesReader.scalar_values].
    The labels of the value, such as the model and the metric, are in ``dimensions``.

    Examples
    --------
    >>> from climate_ref.results.values import ScalarValue
    >>> value = ScalarValue(
    ...     id=1,
    ...     execution_id=2,
    ...     execution_group_id=3,
    ...     value=0.5,
    ...     kind="model",
    ...     dimensions={"source_id": "CanESM5", "statistic": "mean"},
    ...     attributes={},
    ... )
    >>> value.dimensions["statistic"]
    'mean'
    """

    id: int
    """Id of the value in the REF database."""

    execution_id: int
    """The execution that produced the value."""

    execution_group_id: int
    """The execution group of that execution."""

    value: float | None
    """The scalar value, which can be ``None`` or NaN."""

    kind: str
    """Whether the value is for a model or a reference. This is ``"model"`` if the value has no such label."""

    dimensions: Mapping[str, str]
    """
    The labels of the value, e.g. ``source_id``, ``experiment_id``, ``metric`` or ``statistic``.

    ``kind`` is not repeated here.
    """

    attributes: Mapping[str, Any]
    """Extra notes stored with the value that are not labels you can filter on."""

    is_outlier: bool | None = None
    """Whether the value was flagged as an outlier, or ``None`` if outlier detection was off."""

    verification_status: str | None = None
    """``"unverified"`` for a flagged value, ``"verified"`` otherwise, or ``None`` if detection was off."""

    diagnostic_slug: str | None = None
    """Slug of the diagnostic that produced the value, set only with ``include_context=True``."""

    provider_slug: str | None = None
    """Slug of the provider of that diagnostic, set only with ``include_context=True``."""


@attrs.frozen(kw_only=True)
class SeriesValue:
    """
    One series of numbers that a diagnostic produced, such as a global-mean time series of one model.

    Read with [ValuesReader.series_values][climate_ref.results.values.ValuesReader.series_values].
    The labels of the series, such as the model, are in ``dimensions``.

    Examples
    --------
    >>> from climate_ref.results.values import SeriesValue
    >>> series = SeriesValue(
    ...     id=1,
    ...     execution_id=2,
    ...     execution_group_id=3,
    ...     values=[280.1, 280.4],
    ...     index=[2000, 2001],
    ...     index_name="time",
    ...     reference_id=None,
    ...     kind="model",
    ...     dimensions={"source_id": "CanESM5"},
    ...     attributes={},
    ... )
    >>> series.is_reference
    False
    """

    id: int
    """Database id of the value."""

    execution_id: int
    """The execution that produced this value."""

    execution_group_id: int
    """The execution group the producing execution belongs to."""

    values: Sequence[float | int]
    """The numbers of the series, in the same order as ``index``."""

    index: Sequence[float | int | str] | None
    """What the series is plotted against, such as times or latitudes."""

    index_name: str | None
    """Name of the index axis, e.g. ``"time"`` or ``"lat"``."""

    reference_id: str | None
    """Id of the observational dataset for a reference series, and ``None`` for a model series."""

    kind: str
    """Model/reference kind, resolved from the ``kind`` CV dimension (defaults to ``"model"``)."""

    dimensions: Mapping[str, str]
    """
    CV dimension values (e.g. ``source_id``, ``statistic``, ``metric``) for this row.

    ``kind`` is excluded here and surfaced as the dedicated ``kind`` field instead.
    """

    attributes: Mapping[str, Any]
    """Free-form, non-CV attributes attached to the row."""

    diagnostic_slug: str | None = None
    """Slug of the diagnostic that produced the value, set only with ``include_context=True``."""

    provider_slug: str | None = None
    """Slug of the provider of that diagnostic, set only with ``include_context=True``."""

    @property
    def is_reference(self) -> bool:
        """Whether this is a reference series, which means ``reference_id`` is set."""
        return self.reference_id is not None


@attrs.frozen(kw_only=True)
class ScalarValueCollection(Page[ScalarValue]):
    """
    The scalar values from a read, with the models and other labels present and any outlier counts.

    Returned by [ValuesReader.scalar_values][climate_ref.results.values.ValuesReader.scalar_values].
    Loop over it for the [ScalarValue][climate_ref.results.values.ScalarValue] objects,
    or call ``to_pandas()`` for a DataFrame.
    ``total_count`` does not include values removed as outliers.

    Examples
    --------
    See which models have results, then get the first 100 values as a table:

    ```python
    with Database.from_config(config, read_only=True) as db:
        collection = Reader(db).values.scalar_values(limit=100)

    print(collection.total_count, len(collection))
    print(collection.facets_dict()["source_id"])
    df = collection.to_pandas()
    ```

    >>> from climate_ref.results.values import Facet, ScalarValue, ScalarValueCollection
    >>> value = ScalarValue(
    ...     id=1,
    ...     execution_id=2,
    ...     execution_group_id=3,
    ...     value=0.5,
    ...     kind="model",
    ...     dimensions={"source_id": "CanESM5"},
    ...     attributes={},
    ... )
    >>> collection = ScalarValueCollection(
    ...     items=(value,),
    ...     total_count=1,
    ...     facets=(Facet(key="source_id", values=("CanESM5",)),),
    ...     offset=0,
    ...     limit=None,
    ...     had_outliers=False,
    ...     outlier_count=0,
    ... )
    >>> collection.facets_dict()
    {'source_id': ['CanESM5']}
    >>> collection.to_pandas()[["source_id", "value"]]
      source_id  value
    0   CanESM5    0.5
    """

    facets: tuple[Facet, ...]
    """The values of each label over all matching results, such as every ``source_id``. Ignores ``limit``."""

    had_outliers: bool
    """Whether outlier detection ran and flagged at least one value."""

    outlier_count: int
    """Number of values flagged as outliers, or ``0`` if detection did not run."""

    def facets_dict(self) -> dict[str, list[str]]:
        """Return ``facets`` as a plain ``{label: [values]}`` dict, e.g. ``{"source_id": ["CanESM5"]}``."""
        return {f.key: list(f.values) for f in self.facets}

    def to_pandas(self) -> pd.DataFrame:
        """
        Return the values as a DataFrame with one row per value.

        The columns are one per label (such as ``source_id`` or ``metric``),
        then ``id``, ``execution_id``, ``execution_group_id``, ``kind`` and ``value``.
        ``is_outlier`` and ``verification_status`` are added if outlier detection was on.
        ``diagnostic_slug`` and ``provider_slug`` are added if the read used ``include_context=True``.
        """
        detection_ran = any(v.is_outlier is not None for v in self.items)
        return scalar_values_to_frame(self.items, detection_ran=detection_ran)


@attrs.frozen(kw_only=True)
class SeriesValueCollection(Page[SeriesValue]):
    """
    The series values from a read, with the models and other labels present.

    Returned by [ValuesReader.series_values][climate_ref.results.values.ValuesReader.series_values].
    Loop over it for the [SeriesValue][climate_ref.results.values.SeriesValue] objects,
    or call ``to_pandas()`` for a DataFrame.

    Examples
    --------
    Get the observational series, as one row per point and as one row per series:

    ```python
    with Database.from_config(config, read_only=True) as db:
        collection = Reader(db).values.series_values(MetricValueFilter(reference_only=True))

    long_df = collection.to_pandas()
    wide_df = collection.to_pandas(explode=False)
    ```
    """

    facets: tuple[Facet, ...]
    """The values of each label over all matching results, such as every ``source_id``. Ignores ``limit``."""

    def facets_dict(self) -> dict[str, list[str]]:
        """Return ``facets`` as a plain ``{label: [values]}`` dict, e.g. ``{"source_id": ["CanESM5"]}``."""
        return {f.key: list(f.values) for f in self.facets}

    def to_pandas(self, *, explode: bool = True) -> pd.DataFrame:
        """
        Return the series as a DataFrame.

        With ``explode=True`` (the default) there is one row per point, with columns ``index`` and ``value``.
        This is the form to plot or group with pandas, and it matches the CSV of the REF API.
        With ``explode=False`` there is one row per series, and ``values`` and ``index`` hold lists.
        """
        return series_values_to_frame(self.items, explode=explode)


def _to_facets(facet_map: Mapping[str, Sequence[Any]]) -> tuple[Facet, ...]:
    return tuple(Facet(key=k, values=tuple(v)) for k, v in facet_map.items())


class ValuesReader:
    """
    Gets the metric values that diagnostics produced, to compare models or plot against observations.

    Use it as ``Reader(db).values``.
    ``scalar_values`` gets single numbers, such as a bias per model and metric.
    ``series_values`` gets series, such as time series.
    Both return a collection that you can turn into a DataFrame and keep after the database is closed.

    Pass ``session`` only if you manage the database session yourself.

    Examples
    --------
    ```python
    with Database.from_config(config, read_only=True) as db:
        values = Reader(db).values
        scalars = values.scalar_values(MetricValueFilter(provider_slug="pmp"), limit=50)
        series = values.series_values(MetricValueFilter(diagnostic_slug="enso_tel"))
    ```
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The database session the reads use."""
        return self._session if self._session is not None else self._db.session

    def _facets(self, value_ids: Any) -> tuple[Facet, ...]:
        return _to_facets(collect_facets(self.session, value_ids, MetricValue))

    def scalar_values(  # noqa: PLR0913
        self,
        filters: MetricValueFilter | None = None,
        *,
        outliers: OutlierPolicy | None = None,
        include_unverified: bool = False,
        offset: int = 0,
        limit: int | None = None,
        with_facets: bool = True,
        with_count: bool = True,
        include_context: bool = False,
    ) -> ScalarValueCollection:
        """
        Get single-number metric values, to compare models in a table.

        Call it with a [MetricValueFilter][climate_ref.query.filters.MetricValueFilter]
        to choose the diagnostic, models, experiments or metric.
        With the default ``latest_only=False``, a rerun group gives values from each non-retracted execution,
        so the same value can appear more than once.
        For large results, set ``limit`` to read them in pages.

        Parameters
        ----------
        filters
            Which values to get.
            ``None`` uses the default filter: promoted versions only, with retracted executions left out.
        outliers
            Flags unusual values, e.g. ``OutlierPolicy(method="iqr")``. It is off by default.
        include_unverified
            Keep the values flagged as outliers. By default they are dropped,
            and not counted in ``total_count``.
            This only matters if ``outliers`` is set.
        offset
            Number of values to skip, to read the next page.
        limit
            Maximum number of values to return. ``None`` returns all of them.
        with_facets
            Fill ``facets`` with the values of each label, such as the models present.
        with_count
            Fill ``total_count``. It is ``None`` otherwise.
        include_context
            Fill ``diagnostic_slug`` and ``provider_slug`` on each value.

        Returns
        -------
        ScalarValueCollection
            The values. Call ``to_pandas()`` on it for a DataFrame.

        Notes
        -----
        Outlier flags compare each value with the other models in its cell, across all matching values.
        A cell is one diagnostic and one combination of labels apart from the model ones,
        as described in [climate_ref.results.outliers][].
        So ``offset`` and ``limit`` do not change which values are flagged.
        ``facets`` cover all matching values, including those dropped as outliers or left out by ``limit``.

        Examples
        --------
        Compare two models on the mean of one diagnostic, and flag outliers:

        ```python
        from climate_ref.results import MetricValueFilter, OutlierPolicy

        filters = MetricValueFilter(
            provider_slug="example",
            diagnostic_slug="global-mean-timeseries",
            dimensions={"statistic": "mean", "source_id": ["ACCESS-ESM1-5", "CanESM5"]},
        )
        with Database.from_config(config, read_only=True) as db:
            collection = Reader(db).values.scalar_values(
                filters,
                outliers=OutlierPolicy(method="iqr", factor=10.0),
                include_unverified=True,
                include_context=True,
            )
        print(collection.outlier_count)
        df = collection.to_pandas()
        ```
        """
        check_paging(offset, limit)
        filters = filters or MetricValueFilter()
        policy = outliers or OutlierPolicy(method="off")

        if policy.enabled:
            return self._scalar_values_with_outliers(
                filters,
                policy,
                include_unverified=include_unverified,
                offset=offset,
                limit=limit,
                with_facets=with_facets,
                with_count=with_count,
                include_context=include_context,
            )

        # No outlier detection: page and count in SQL so we never materialise the whole table.
        value_ids = select_value_ids(MetricValueType.SCALAR, filters)
        facets = self._facets(value_ids) if with_facets else ()
        total_count = count(self.session, value_ids) if with_count else None
        load_stmt = _select_page(ScalarMetricValue, value_ids, offset, limit).options(
            self._scalar_loader(include_context)
        )
        rows = list(self.session.execute(load_stmt).scalars().all())
        items = tuple(self._to_scalar_dto(r, include_context) for r in rows)
        return ScalarValueCollection(
            items=items,
            total_count=total_count,
            facets=facets,
            offset=offset,
            limit=limit,
            had_outliers=False,
            outlier_count=0,
        )

    def _scalar_values_with_outliers(  # noqa: PLR0913
        self,
        filters: MetricValueFilter,
        policy: OutlierPolicy,
        *,
        include_unverified: bool,
        offset: int,
        limit: int | None,
        with_facets: bool,
        with_count: bool,
        include_context: bool,
    ) -> ScalarValueCollection:
        """
        Flag every matching value in the database so the bounds do not depend on the page, then page.

        Only the values on the page are loaded.
        The counts ride along on each row of the page, so detection runs once unless the page is empty.
        """
        value_ids = select_value_ids(MetricValueType.SCALAR, filters)
        flags = select_outlier_flags(value_ids, policy).subquery("outlier_flags")
        facets = self._facets(value_ids) if with_facets else ()
        flagged_sum = func.sum(case((flags.c.is_outlier, 1), else_=0))
        counted = select(
            flags.c.id,
            flags.c.is_outlier,
            func.count().over().label("matched"),
            flagged_sum.over().label("flagged"),
        ).subquery("outlier_counted")
        page_ids = select(counted).order_by(counted.c.id)
        if not include_unverified:
            page_ids = page_ids.where(~counted.c.is_outlier)
        page = paginate(page_ids, offset, limit).subquery("outlier_page")
        page_stmt = (
            select(ScalarMetricValue, page.c.is_outlier, page.c.matched, page.c.flagged)
            .join(page, page.c.id == ScalarMetricValue.id)
            .options(self._scalar_loader(include_context))
            .order_by(page.c.id)
        )
        rows = self.session.execute(page_stmt).all()
        if rows:
            matched, flagged = rows[0].matched, rows[0].flagged
        else:
            matched, flagged = self.session.execute(select(func.count(), func.coalesce(flagged_sum, 0))).one()
        total_count = (matched if include_unverified else matched - flagged) if with_count else None
        items = tuple(
            attrs.evolve(
                self._to_scalar_dto(row, include_context),
                is_outlier=bool(is_outlier),
                verification_status="unverified" if is_outlier else "verified",
            )
            for row, is_outlier, _, _ in rows
        )
        return ScalarValueCollection(
            items=items,
            total_count=total_count,
            facets=facets,
            offset=offset,
            limit=limit,
            had_outliers=flagged > 0,
            outlier_count=flagged,
        )

    def series_values(  # noqa: PLR0913
        self,
        filters: MetricValueFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        with_facets: bool = True,
        with_count: bool = True,
        include_context: bool = False,
    ) -> SeriesValueCollection:
        """
        Get series of values, such as time series, to plot.

        Call it with a [MetricValueFilter][climate_ref.query.filters.MetricValueFilter]
        to choose the diagnostic and models.
        With the default ``latest_only=False``, a rerun group gives series from each non-retracted execution,
        so the same series can appear more than once.
        Set ``reference_only`` on the filter to get the observational series or the model series.
        For large results, set ``limit`` to read them in pages.

        Parameters
        ----------
        filters
            Which series to get.
            ``None`` uses the default filter: promoted versions only, with retracted executions left out.
        offset
            Number of series to skip, to read the next page.
        limit
            Maximum number of series to return. ``None`` returns all of them.
        with_facets
            Fill ``facets`` with the values of each label, such as the models present.
        with_count
            Fill ``total_count``. It is ``None`` otherwise.
        include_context
            Fill ``diagnostic_slug`` and ``provider_slug`` on each series.

        Returns
        -------
        SeriesValueCollection
            The series, each with its ``index`` and ``index_name``.
            ``facets`` cover all matching series, including those left out by ``limit``.

        Examples
        --------
        Get the model series of one diagnostic for plotting:

        ```python
        from climate_ref.results import MetricValueFilter

        filters = MetricValueFilter(provider_slug="pmp", diagnostic_slug="enso_tel", reference_only=False)
        with Database.from_config(config, read_only=True) as db:
            collection = Reader(db).values.series_values(filters, limit=20)
        df = collection.to_pandas()
        ```
        """
        check_paging(offset, limit)
        filters = filters or MetricValueFilter()

        value_ids = select_value_ids(MetricValueType.SERIES, filters)
        facets = self._facets(value_ids) if with_facets else ()
        total_count = count(self.session, value_ids) if with_count else None

        load_stmt = _select_page(SeriesMetricValue, value_ids, offset, limit).options(
            self._series_loader(include_context)
        )
        rows = list(self.session.execute(load_stmt).scalars().unique().all())

        items = tuple(self._to_series_dto(r, include_context) for r in rows)
        return SeriesValueCollection(
            items=items, total_count=total_count, facets=facets, offset=offset, limit=limit
        )

    @staticmethod
    def _scalar_loader(include_context: bool) -> Any:
        exec_load = joinedload(ScalarMetricValue.execution)
        if include_context:
            return (
                exec_load.joinedload(Execution.execution_group)
                .joinedload(ExecutionGroup.diagnostic)
                .joinedload(Diagnostic.provider)
            )
        return exec_load

    @staticmethod
    def _series_loader(include_context: bool) -> Any:
        exec_load = joinedload(SeriesMetricValue.execution)
        if include_context:
            return (
                exec_load.joinedload(Execution.execution_group)
                .joinedload(ExecutionGroup.diagnostic)
                .joinedload(Diagnostic.provider)
            )
        return exec_load

    @staticmethod
    def _context_slugs(execution: Execution, include_context: bool) -> tuple[str | None, str | None]:
        if not include_context:
            return None, None
        diagnostic = execution.execution_group.diagnostic
        return diagnostic.slug, diagnostic.provider.slug

    def _to_scalar_dto(self, row: ScalarMetricValue, include_context: bool) -> ScalarValue:
        diagnostic_slug, provider_slug = self._context_slugs(row.execution, include_context)
        kind, dims = _split_kind(row.dimensions)
        return ScalarValue(
            id=row.id,
            execution_id=row.execution_id,
            execution_group_id=row.execution.execution_group_id,
            value=row.value,
            kind=kind,
            dimensions=dims,
            attributes=dict(row.attributes or {}),
            diagnostic_slug=diagnostic_slug,
            provider_slug=provider_slug,
        )

    def _to_series_dto(self, row: SeriesMetricValue, include_context: bool) -> SeriesValue:
        diagnostic_slug, provider_slug = self._context_slugs(row.execution, include_context)
        kind, dims = _split_kind(row.dimensions)
        return SeriesValue(
            id=row.id,
            execution_id=row.execution_id,
            execution_group_id=row.execution.execution_group_id,
            values=list(row.values or []),
            index=list(row.index) if row.index is not None else None,
            index_name=row.index_name,
            reference_id=row.reference_id,
            kind=kind,
            dimensions=dims,
            attributes=dict(row.attributes or {}),
            diagnostic_slug=diagnostic_slug,
            provider_slug=provider_slug,
        )
