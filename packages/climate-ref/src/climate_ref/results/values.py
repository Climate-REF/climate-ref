"""
Typed, ORM-free surface for metric-value results.

[Reader][climate_ref.results.values.Reader] is the intent-named entry point that notebooks
and (eventually) the API use, via its [values][climate_ref.results.values.Reader.values]
sub-reader.
[ValuesReader][climate_ref.results.values.ValuesReader]'s read methods return frozen,
detached value objects wrapped in collections that offer ``to_pandas()``.
The objects are fully materialised, so they remain valid after the originating session closes.
A notebook can build a DataFrame inside a ``with Database(...)`` block and keep using it afterwards.

Everything here is a thin layer over [climate_ref.results._query][]
(the shared ``Select`` builders) and [climate_ref.results.outliers][].
power users who need the raw ``Select`` or ORM rows use those modules directly.
"""

import contextlib
import functools
import gc
import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import pandas as pd
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, MetricValueType, SeriesIndex
from climate_ref.models.provider import Provider
from climate_ref.results._page import Page
from climate_ref.results._query import (
    MetricValueFilter,
    count_values,
    paginate,
    select_value_ids,
)
from climate_ref.results.executions import ExecutionsReader
from climate_ref.results.frames import (
    collect_facets,
    facets_from_columns,
    finish_value_frame,
    scalar_values_to_frame,
    series_values_to_frame,
)
from climate_ref.results.outliers import OutlierPolicy, flag_outliers

if TYPE_CHECKING:
    from climate_ref.results.artifacts import ArtifactsReader
    from climate_ref.results.datasets import DatasetsReader
    from climate_ref.results.diagnostics import DiagnosticsReader
    from climate_ref.results.resources import ResourcesReader


_ID_CHUNK = 900
"""Ids per ``IN`` lookup, below the 999 bound parameters SQLite allowed before 3.32."""


def _scalar_columns() -> list[Any]:
    """Return the columns a ``ScalarValue`` is built from, in the order ``_scalar_dtos`` unpacks them."""
    table = MetricValue.__table__.c
    return [
        table.id,
        table.execution_id,
        table.value,
        table.attributes,
        *(table[d] for d in MetricValue._cv_dimensions),
    ]


def _series_columns() -> list[Any]:
    """Return the columns a ``SeriesValue`` is built from, in the order ``_series_dtos`` unpacks them."""
    table = MetricValue.__table__.c
    return [
        table.id,
        table.execution_id,
        table["values"],
        table.index_id,
        table.reference_id,
        table.attributes,
        *(table[d] for d in MetricValue._cv_dimensions),
    ]


@contextlib.contextmanager
def _gc_paused() -> Iterator[None]:
    """Pause the cyclic garbage collector, which otherwise rescans every row object a large read creates."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _detection_dims(policy: OutlierPolicy) -> list[str]:
    """Return the CV dimensions outlier detection reads: the policy's groups plus ``source_id``."""
    keys = {*policy.group_by, "source_id"}
    return [d for d in MetricValue._cv_dimensions if d in keys]


def _split_kind(dimensions: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    """Split the model/reference ``kind`` (defaults to ``"model"``) from the other CV dimensions."""
    rest = dict(dimensions)
    return rest.pop("kind", "model"), rest


@attrs.frozen(kw_only=True)
class Facet:
    """Distinct values observed for one CV dimension across a filtered query."""

    key: str
    """The CV dimension name."""

    values: tuple[str, ...]
    """Distinct values observed for the dimension."""


@attrs.frozen(kw_only=True)
class ScalarValue:
    """A single scalar metric value, detached from the ORM."""

    id: int
    """Primary key of the underlying ``ScalarMetricValue`` row."""

    execution_id: int
    """The execution that produced this value."""

    execution_group_id: int
    """The execution group the producing execution belongs to."""

    value: float | None
    """The scalar value itself."""

    kind: str
    """Model/reference kind, resolved from the ``kind`` CV dimension (defaults to ``"model"``)."""

    dimensions: Mapping[str, str]
    """
    CV dimension values (e.g. ``source_id``, ``statistic``, ``metric``) for this row.

    ``kind`` is excluded here and surfaced as the dedicated ``kind`` field instead.
    """

    attributes: Mapping[str, Any]
    """Free-form, non-CV attributes attached to the row."""

    is_outlier: bool | None = None
    """Per-row outlier verdict; ``None`` unless outlier detection ran."""

    verification_status: str | None = None
    """``"verified"``/``"unverified"`` status paired with ``is_outlier``; ``None`` unless detection ran."""

    diagnostic_slug: str | None = None
    """Owning diagnostic's slug, populated only when ``include_context=True``."""

    provider_slug: str | None = None
    """Owning provider's slug, populated only when ``include_context=True``."""


@attrs.frozen(kw_only=True)
class SeriesValue:
    """A 1-d series metric value, detached from the ORM. The index is snapshotted from the shared axis."""

    id: int
    """Primary key of the underlying ``SeriesMetricValue`` row."""

    execution_id: int
    """The execution that produced this value."""

    execution_group_id: int
    """The execution group the producing execution belongs to."""

    values: Sequence[float | int]
    """The series' y-values, in index order."""

    index: Sequence[float | int | str] | None
    """The shared index axis values, snapshotted at read time."""

    index_name: str | None
    """Name of the shared index axis (e.g. a time or latitude label)."""

    reference_id: str | None
    """Set for observation/reference series; ``None`` for model series."""

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
    """Owning diagnostic's slug, populated only when ``include_context=True``."""

    provider_slug: str | None = None
    """Owning provider's slug, populated only when ``include_context=True``."""

    @property
    def is_reference(self) -> bool:
        """True for observation/reference series (``reference_id`` is set)."""
        return self.reference_id is not None


@attrs.frozen(kw_only=True)
class ScalarValueCollection(Page[ScalarValue]):
    """
    An immutable page of scalar values plus collection-level, outlier-aware metadata.

    ``total_count`` is taken after outlier removal, if any.
    """

    facets: tuple[Facet, ...]
    """Distinct CV dimension values observed over the full filtered set, before pagination."""

    had_outliers: bool
    """Whether outlier detection ran and flagged at least one value."""

    outlier_count: int
    """Number of values flagged as outliers, or ``0`` when detection did not run."""

    def facets_dict(self) -> dict[str, list[str]]:
        """Facets as a plain ``{dimension: [values]}`` mapping."""
        return {f.key: list(f.values) for f in self.facets}

    def to_pandas(self) -> pd.DataFrame:
        """Tidy DataFrame: one row per value, one column per CV dimension present, plus metadata."""
        detection_ran = any(v.is_outlier is not None for v in self.items)
        return scalar_values_to_frame(self.items, detection_ran=detection_ran)


@attrs.frozen(kw_only=True)
class SeriesValueCollection(Page[SeriesValue]):
    """An immutable page of series values plus collection-level metadata."""

    facets: tuple[Facet, ...]
    """Distinct CV dimension values observed over the full filtered set, before pagination."""

    def facets_dict(self) -> dict[str, list[str]]:
        """Facets as a plain ``{dimension: [values]}`` mapping."""
        return {f.key: list(f.values) for f in self.facets}

    def to_pandas(self, *, explode: bool = True) -> pd.DataFrame:
        """
        DataFrame of the series values.

        With ``explode=True`` (default) the result is long-form: one row per (series, index point),
        matching the API's CSV shape. With ``explode=False`` each series is one row with list-valued
        ``values``/``index`` cells.
        """
        return series_values_to_frame(self.items, explode=explode)


def _to_facets(facet_map: Mapping[str, Sequence[Any]]) -> tuple[Facet, ...]:
    return tuple(Facet(key=k, values=tuple(v)) for k, v in facet_map.items())


class ValuesReader:
    """
    Metric-value read domain: scalar/series values and their facets.

    Constructed from a [Database][climate_ref.database.Database], which owns the session and the
    read-only story. All read methods return detached collections that outlive the session.

    Pass ``session`` to read through a caller-owned session instead,
    for example one opened by [Database.session_scope][climate_ref.database.Database.session_scope].
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
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
        Query scalar values, returning an outlier-aware collection.

        When outlier detection is disabled (the default) pagination and counting happen in SQL,
        so cost scales with the requested page rather than the whole result set. When ``outliers``
        is enabled, detection runs over the FULL filtered set so IQR bounds are globally
        consistent; with ``include_unverified`` False, flagged values are then removed before
        pagination and excluded from ``total_count``. ``facets`` are always computed over the full
        filtered set (before any outlier removal and pagination).
        """
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

        # No detection: page and count in SQL so we never materialise the whole table.
        value_ids = select_value_ids(MetricValueType.SCALAR, filters)
        facets = self._facets(value_ids) if with_facets else ()
        total_count = count_values(self.session, value_ids) if with_count else None
        page_stmt = value_ids.with_only_columns(*_scalar_columns()).order_by(MetricValue.__table__.c.id)
        rows = self._fetch(paginate(page_stmt, offset, limit))
        items = self._scalar_dtos(rows, include_context)
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
        Run detection over the full filtered set so IQR bounds are globally consistent, then filter and page.

        Only the columns detection needs are loaded for every row, and only the returned page becomes DTOs.
        """
        dims = MetricValue._cv_dimensions
        scalar_rows = self._whole_set(MetricValueType.SCALAR, filters, ["id", "execution_id", "value", *dims])
        _, _, values, *dim_columns = zip(*scalar_rows) if scalar_rows else [()] * (3 + len(dims))
        facets = _to_facets(facets_from_columns(dims, dim_columns)) if with_facets else ()
        detection_dims = _detection_dims(policy)
        frame = pd.DataFrame(
            {
                "value": values,
                **{k: column for k, column in zip(dims, dim_columns) if k in detection_dims},
            }
        )

        is_outlier = flag_outliers(frame, policy)
        outlier_count = int(is_outlier.sum())
        kept = np.arange(len(scalar_rows)) if include_unverified else np.flatnonzero(~is_outlier)

        total_count = len(kept) if with_count else None
        page = kept[offset : offset + limit] if limit is not None else kept[offset:]
        verdicts = {scalar_rows[i][0]: bool(is_outlier[i]) for i in page.tolist()}
        items = self._scalar_dtos(
            self._fetch_by_id(_scalar_columns(), list(verdicts)), include_context, verdicts
        )
        return ScalarValueCollection(
            items=items,
            total_count=total_count,
            facets=facets,
            offset=offset,
            limit=limit,
            had_outliers=outlier_count > 0,
            outlier_count=outlier_count,
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
        Query series values with SQL-level pagination.

        The shared index axis is resolved so ``index``/``index_name`` are populated. ``facets`` are
        computed over the full filtered set before pagination.
        """
        filters = filters or MetricValueFilter()

        value_ids = select_value_ids(MetricValueType.SERIES, filters)
        facets = self._facets(value_ids) if with_facets else ()
        total_count = count_values(self.session, value_ids) if with_count else None

        page_stmt = value_ids.with_only_columns(*_series_columns()).order_by(MetricValue.__table__.c.id)
        items = self._series_dtos(self._fetch(paginate(page_stmt, offset, limit)), include_context)
        return SeriesValueCollection(
            items=items, total_count=total_count, facets=facets, offset=offset, limit=limit
        )

    # -- frames ----------------------------------------------------------------
    # Whole-set reads for notebooks, built straight from columns without a value object per row.
    @_gc_paused()
    def scalar_frame(
        self,
        filters: MetricValueFilter | None = None,
        *,
        outliers: OutlierPolicy | None = None,
        include_unverified: bool = False,
        include_context: bool = False,
    ) -> pd.DataFrame:
        """
        Every scalar value matching ``filters`` as a DataFrame, ordered by id.

        Holds the same rows and columns as ``scalar_values(...).to_pandas()`` for the same arguments,
        but is several times faster on a large set because no ``ScalarValue`` is built.
        The CV dimension columns come first, in registration order.
        """
        filters = filters or MetricValueFilter()
        policy = outliers or OutlierPolicy(method="off")
        names = ["id", "execution_id", "value", *MetricValue._cv_dimensions]
        frame = pd.DataFrame.from_records(
            self._whole_set(MetricValueType.SCALAR, filters, names), columns=names
        )

        outlier_columns = []
        if policy.enabled:
            is_outlier = flag_outliers(frame[["value", *_detection_dims(policy)]], policy)
            frame["is_outlier"] = is_outlier
            frame["verification_status"] = np.where(is_outlier, "unverified", "verified")
            if not include_unverified:
                frame = frame[~is_outlier]
            outlier_columns = ["is_outlier", "verification_status"]

        frame = self._add_context(frame, include_context)
        return finish_value_frame(frame, ["value", *outlier_columns])

    @_gc_paused()
    def series_frame(
        self,
        filters: MetricValueFilter | None = None,
        *,
        explode: bool = True,
        include_context: bool = False,
    ) -> pd.DataFrame:
        """
        Every series value matching ``filters`` as a DataFrame, ordered by id.

        Holds the same rows and columns as ``series_values(...).to_pandas(explode=explode)``,
        but is several times faster on a large set because no ``SeriesValue`` is built.
        The CV dimension columns come first, in registration order.
        """
        filters = filters or MetricValueFilter()
        names = ["id", "execution_id", "values", "index_id", "reference_id", *MetricValue._cv_dimensions]
        frame = pd.DataFrame.from_records(
            self._whole_set(MetricValueType.SERIES, filters, names), columns=names
        )
        if explode:
            # A series with no points has no exploded rows, so it must not keep a dimension column.
            frame = frame.loc[np.array([bool(v) for v in frame["values"]], dtype=bool)]
        frame = self._add_context(frame, include_context)

        axes = self._index_axes(frame["index_id"].dropna().astype(int).tolist())
        frame["index_name"] = [(axes[i][0] if i in axes else None) or "index" for i in frame["index_id"]]
        index = [axes[i][1] if i in axes else None for i in frame.pop("index_id")]
        values = [v or [] for v in frame.pop("values")]
        # Finish before exploding, so it works on one row per series rather than one per point.
        frame = finish_value_frame(frame, ["index_name", "reference_id"])
        if not explode:
            frame["values"] = values
            # A fresh list per row, so editing one cell cannot change the others sharing the axis.
            frame["index"] = pd.Series([None if idx is None else list(idx) for idx in index], dtype=object)
            return frame

        lengths = [len(v) for v in values]
        frame = frame.iloc[np.repeat(np.arange(len(frame)), lengths)].reset_index(drop=True)
        frame["value"] = list(itertools.chain.from_iterable(values))
        # Points past the end of the axis fall back to their position, as ``to_pandas`` does.
        frame["index"] = list(
            itertools.chain.from_iterable(
                range(n) if idx is None else idx if len(idx) == n else [*idx[:n], *range(len(idx), n)]
                for idx, n in zip(index, lengths)
            )
        )
        return frame

    def _whole_set(
        self, value_type: MetricValueType, filters: MetricValueFilter, names: Sequence[str]
    ) -> list[tuple[Any, ...]]:
        """Fetch the named ``metric_value`` columns of every value of ``value_type`` matching ``filters``."""
        table = MetricValue.__table__.c
        return self._fetch(
            select_value_ids(value_type, filters)
            .with_only_columns(*(table[name] for name in names))
            .order_by(table.id)
        )

    def _add_context(self, frame: pd.DataFrame, include_context: bool) -> pd.DataFrame:
        """Join ``execution_group_id`` (and the slugs) by execution, dropping rows whose execution is gone."""
        names = ["execution_group_id", "diagnostic_slug", "provider_slug"][: 3 if include_context else 1]
        context = self._execution_context(frame["execution_id"].unique().tolist(), include_context)
        by_execution = pd.DataFrame.from_dict(
            {execution_id: row[: len(names)] for execution_id, row in context.items()},
            orient="index",
            columns=names,
        )
        return frame.join(by_execution, on="execution_id", how="inner").reset_index(drop=True)

    # -- materialisers ---------------------------------------------------------
    # Rows are fetched as plain columns, because building ORM objects dominates a whole-diagnostic read.
    def _fetch(self, stmt: Select[Any]) -> list[tuple[Any, ...]]:
        """Run a column ``Select`` on the session's connection, flushing first as the ORM would."""
        if self.session.autoflush:
            self.session.flush()
        return list(self.session.connection().execute(stmt).tuples().all())

    def _fetch_in(self, stmt: Select[Any], column: Any, ids: Sequence[int]) -> list[tuple[Any, ...]]:
        """Rows of ``stmt`` whose ``column`` is in ``ids``, one ``IN`` lookup per chunk of ids."""
        rows = []
        for start in range(0, len(ids), _ID_CHUNK):
            rows += self._fetch(stmt.where(column.in_(ids[start : start + _ID_CHUNK])))
        return rows

    def _fetch_by_id(self, columns: Sequence[Any], ids: Sequence[int]) -> list[tuple[Any, ...]]:
        """Rows of ``columns`` for the metric values ``ids``, in ``ids`` order, skipping deleted rows."""
        rows = self._fetch_in(select(*columns), MetricValue.__table__.c.id, ids)
        position = {row_id: i for i, row_id in enumerate(ids)}
        return sorted(rows, key=lambda row: position[row[0]])

    def _execution_context(
        self, execution_ids: Iterable[int], include_context: bool
    ) -> dict[int, tuple[int, str | None, str | None]]:
        """``(execution_group_id, diagnostic_slug, provider_slug)`` per execution id."""
        stmt = select(Execution.id, Execution.execution_group_id)
        if include_context:
            stmt = (
                stmt.add_columns(Diagnostic.slug, Provider.slug)
                .join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
                .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
                .join(Provider, Diagnostic.provider_id == Provider.id)
            )
        rows = self._fetch_in(stmt, Execution.id, sorted(set(execution_ids)))
        return {execution_id: (group_id, *(slugs or (None, None))) for execution_id, group_id, *slugs in rows}

    def _index_axes(self, index_ids: Iterable[int | None]) -> dict[int, tuple[str | None, list[Any]]]:
        """``(name, values)`` per index axis id, so each shared axis is decoded once."""
        stmt = select(SeriesIndex.id, SeriesIndex.name, SeriesIndex.values)
        rows = self._fetch_in(stmt, SeriesIndex.id, sorted({i for i in index_ids if i is not None}))
        return {axis_id: (name, values) for axis_id, name, values in rows}

    def _scalar_dtos(
        self,
        rows: Sequence[tuple[Any, ...]],
        include_context: bool,
        verdicts: Mapping[int, bool] | None = None,
    ) -> tuple[ScalarValue, ...]:
        """DTOs for rows of ``_scalar_columns()``, with outlier verdicts by id when detection ran."""
        dims = MetricValue._cv_dimensions
        context = self._execution_context((row[1] for row in rows), include_context)
        items = []
        for row_id, execution_id, value, attributes, *dim_values in rows:
            if execution_id not in context:
                # Deleted by another transaction since its values were read.
                continue
            group_id, diagnostic_slug, provider_slug = context[execution_id]
            kind, dimensions = _split_kind({k: v for k, v in zip(dims, dim_values) if v is not None})
            flagged = verdicts[row_id] if verdicts is not None else None
            items.append(
                ScalarValue(
                    id=row_id,
                    execution_id=execution_id,
                    execution_group_id=group_id,
                    value=value,
                    kind=kind,
                    dimensions=dimensions,
                    attributes=dict(attributes or {}),
                    is_outlier=flagged,
                    verification_status=None if flagged is None else "unverified" if flagged else "verified",
                    diagnostic_slug=diagnostic_slug,
                    provider_slug=provider_slug,
                )
            )
        return tuple(items)

    def _series_dtos(self, rows: Sequence[tuple[Any, ...]], include_context: bool) -> tuple[SeriesValue, ...]:
        """DTOs for rows of ``_series_columns()``."""
        dims = MetricValue._cv_dimensions
        context = self._execution_context((row[1] for row in rows), include_context)
        axes = self._index_axes(row[3] for row in rows)
        items = []
        for row_id, execution_id, values, index_id, reference_id, attributes, *dim_values in rows:
            if execution_id not in context:
                # Deleted by another transaction since its values were read.
                continue
            group_id, diagnostic_slug, provider_slug = context[execution_id]
            index_name, index = axes[index_id] if index_id is not None else (None, None)
            kind, dimensions = _split_kind({k: v for k, v in zip(dims, dim_values) if v is not None})
            items.append(
                SeriesValue(
                    id=row_id,
                    execution_id=execution_id,
                    execution_group_id=group_id,
                    values=list(values or []),
                    index=list(index) if index is not None else None,
                    index_name=index_name,
                    reference_id=reference_id,
                    kind=kind,
                    dimensions=dimensions,
                    attributes=dict(attributes or {}),
                    diagnostic_slug=diagnostic_slug,
                    provider_slug=provider_slug,
                )
            )
        return tuple(items)


class Reader:
    """
    Typed entry point to REF query results.

    Constructed from a [Database][climate_ref.database.Database], which owns the session and the
    read-only story. This is a thin entry point: it exposes per-domain sub-readers as properties,
    [values][climate_ref.results.values.Reader.values] for metric-value reads,
    [executions][climate_ref.results.values.Reader.executions] for execution-group reads,
    [datasets][climate_ref.results.values.Reader.datasets] for dataset reads,
    [diagnostics][climate_ref.results.values.Reader.diagnostics] for diagnostic reads,
    [resources][climate_ref.results.values.Reader.resources] for per-execution resource
    measurements, and
    [artifacts][climate_ref.results.values.Reader.artifacts] for output path resolution
    (only available when a ``results`` root is supplied).

    By default every sub-reader queries through the database's long-lived session.
    Pass ``session`` to read through a caller-owned session instead,
    for example one opened by [Database.session_scope][climate_ref.database.Database.session_scope].
    The sub-readers are cached, so a reader built this way must not outlive the scope it was given.
    """

    def __init__(
        self, database: Database, results: Path | None = None, *, session: Session | None = None
    ) -> None:
        self._db = database
        self._results = results
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._session if self._session is not None else self._db.session

    @functools.cached_property
    def values(self) -> ValuesReader:
        """Metric-value reads (scalar/series/facets)."""
        return ValuesReader(self._db, session=self._session)

    @functools.cached_property
    def executions(self) -> ExecutionsReader:
        """Execution-group and execution reads."""
        return ExecutionsReader(self._db, session=self._session)

    @functools.cached_property
    def datasets(self) -> "DatasetsReader":
        """Dataset reads."""
        from climate_ref.results.datasets import DatasetsReader  # noqa: PLC0415

        return DatasetsReader(self._db, session=self._session)

    @functools.cached_property
    def diagnostics(self) -> "DiagnosticsReader":
        """Diagnostic reads."""
        from climate_ref.results.diagnostics import DiagnosticsReader  # noqa: PLC0415

        return DiagnosticsReader(self._db, session=self._session)

    @functools.cached_property
    def resources(self) -> "ResourcesReader":
        """Per-execution resource measurement reads."""
        from climate_ref.results.resources import ResourcesReader  # noqa: PLC0415

        return ResourcesReader(self._db, session=self._session)

    @functools.cached_property
    def artifacts(self) -> "ArtifactsReader":
        """
        Output path resolution.

        Raises ``ValueError`` when no ``results`` root was supplied to the constructor.
        """
        if self._results is None:
            raise ValueError(
                "reader.artifacts requires a results root; construct "
                "Reader(database, results=config.paths.results)."
            )
        from climate_ref.results.artifacts import ArtifactsReader  # noqa: PLC0415

        return ArtifactsReader(self._results)
