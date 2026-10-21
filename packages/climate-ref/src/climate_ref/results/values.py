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

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, MetricValueType, ScalarMetricValue, SeriesMetricValue
from climate_ref.models.provider import Provider
from climate_ref.results._page import Page
from climate_ref.results._query import (
    MetricValueFilter,
    count_values,
    paginate,
    select_scalar_values,
    select_series_values,
    select_value_ids,
)
from climate_ref.results.executions import ExecutionsReader
from climate_ref.results.frames import (
    collect_facets,
    facets_from_columns,
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
        base_stmt = select_scalar_values(filters)
        value_ids = select_value_ids(MetricValueType.SCALAR, filters)
        facets = self._facets(value_ids) if with_facets else ()
        total_count = count_values(self.session, value_ids) if with_count else None
        load_stmt = paginate(base_stmt.options(self._scalar_loader(include_context)), offset, limit)
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
        Run detection over the full filtered set so IQR bounds are globally consistent, then filter and page.

        Only the columns detection needs are loaded for every row, and only the returned page becomes DTOs.
        """
        dims = ScalarMetricValue._cv_dimensions
        table = MetricValue.__table__.c
        detect_stmt = (
            select_value_ids(MetricValueType.SCALAR, filters)
            .with_only_columns(table.id, table.execution_id, table.value, *(table[d] for d in dims))
            .order_by(table.id)
        )
        # Core execution, because the ORM result layer adds a third to this fetch, so flush as it would.
        if self.session.autoflush:
            self.session.flush()
        scalar_rows = self.session.connection().execute(detect_stmt).tuples().all()
        _, _, values, *dim_columns = zip(*scalar_rows) if scalar_rows else [()] * (3 + len(dims))
        facets = _to_facets(facets_from_columns(dims, dim_columns)) if with_facets else ()
        detection_dims = {*policy.group_by, "source_id"}
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
        items = self._scalar_page(scalar_rows, is_outlier, page.tolist(), include_context)
        return ScalarValueCollection(
            items=items,
            total_count=total_count,
            facets=facets,
            offset=offset,
            limit=limit,
            had_outliers=outlier_count > 0,
            outlier_count=outlier_count,
        )

    def _scalar_page(
        self, rows: Sequence[Any], is_outlier: np.ndarray, positions: list[int], include_context: bool
    ) -> tuple[ScalarValue, ...]:
        """DTOs for the detection rows at ``positions``, fetching the columns detection did not need."""
        dims = ScalarMetricValue._cv_dimensions
        columns: list[Any] = [
            ScalarMetricValue.id,
            Execution.execution_group_id,
            ScalarMetricValue.attributes,
        ]
        if include_context:
            columns += [Diagnostic.slug.label("diagnostic_slug"), Provider.slug.label("provider_slug")]
        extra_stmt = select(*columns).join(Execution, ScalarMetricValue.execution_id == Execution.id)
        if include_context:
            extra_stmt = (
                extra_stmt.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
                .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
                .join(Provider, Diagnostic.provider_id == Provider.id)
            )
        extras: dict[int, Any] = {}
        ids = [rows[i][0] for i in positions]
        for start in range(0, len(ids), _ID_CHUNK):
            chunk = ids[start : start + _ID_CHUNK]
            for row in self.session.execute(extra_stmt.where(ScalarMetricValue.id.in_(chunk))):
                extras[row.id] = row

        items = []
        for i in positions:
            row_id, execution_id, value, *dim_values = rows[i]
            extra = extras.get(row_id)
            if extra is None:
                # Deleted by another transaction since detection read it.
                continue
            kind, dimensions = _split_kind({k: v for k, v in zip(dims, dim_values) if v is not None})
            flagged = bool(is_outlier[i])
            items.append(
                ScalarValue(
                    id=row_id,
                    execution_id=execution_id,
                    execution_group_id=extra.execution_group_id,
                    value=value,
                    kind=kind,
                    dimensions=dimensions,
                    attributes=dict(extra.attributes or {}),
                    is_outlier=flagged,
                    verification_status="unverified" if flagged else "verified",
                    diagnostic_slug=extra.diagnostic_slug if include_context else None,
                    provider_slug=extra.provider_slug if include_context else None,
                )
            )
        return tuple(items)

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

        base_stmt = select_series_values(filters)
        value_ids = select_value_ids(MetricValueType.SERIES, filters)
        facets = self._facets(value_ids) if with_facets else ()
        total_count = count_values(self.session, value_ids) if with_count else None

        load_stmt = paginate(base_stmt.options(self._series_loader(include_context)), offset, limit)
        rows = list(self.session.execute(load_stmt).scalars().unique().all())

        items = tuple(self._to_series_dto(r, include_context) for r in rows)
        return SeriesValueCollection(
            items=items, total_count=total_count, facets=facets, offset=offset, limit=limit
        )

    # -- loaders / materialisers ------------------------------------------------
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
