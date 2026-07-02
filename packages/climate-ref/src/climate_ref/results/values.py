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
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import pandas as pd
from sqlalchemy.orm import Session, joinedload

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import ScalarMetricValue, SeriesMetricValue
from climate_ref.results._query import (
    MetricValueFilter,
    count_values,
    select_scalar_values,
    select_series_values,
)
from climate_ref.results.executions import ExecutionsReader
from climate_ref.results.frames import collect_facets
from climate_ref.results.outliers import OutlierPolicy, detect_scalar_outliers

if TYPE_CHECKING:
    from climate_ref.results.artifacts import ArtifactsReader
    from climate_ref.results.datasets import DatasetsReader
    from climate_ref.results.diagnostics import DiagnosticsReader


def _kind_of(dimensions: Mapping[str, str]) -> str:
    """Resolve the model/reference ``kind`` from the CV dimensions (defaults to ``"model"``)."""
    return dimensions.get("kind", "model")


@attrs.frozen(kw_only=True)
class Facet:
    """Distinct values observed for one CV dimension across a filtered query."""

    key: str
    values: tuple[str, ...]


@attrs.frozen(kw_only=True)
class ScalarValue:
    """A single scalar metric value, detached from the ORM."""

    id: int
    execution_id: int
    execution_group_id: int
    value: float | None
    kind: str
    dimensions: Mapping[str, str]
    attributes: Mapping[str, Any]
    # per-row outlier verdict (populated when detection ran)
    is_outlier: bool | None = None
    verification_status: str | None = None
    # optional provenance, only when include_context=True
    diagnostic_slug: str | None = None
    provider_slug: str | None = None


@attrs.frozen(kw_only=True)
class SeriesValue:
    """A 1-d series metric value, detached from the ORM. The index is snapshotted from the shared axis."""

    id: int
    execution_id: int
    execution_group_id: int
    values: Sequence[float | int]
    index: Sequence[float | int | str] | None
    index_name: str | None
    reference_id: str | None
    kind: str
    dimensions: Mapping[str, str]
    attributes: Mapping[str, Any]
    diagnostic_slug: str | None = None
    provider_slug: str | None = None

    @property
    def is_reference(self) -> bool:
        """True for observation/reference series (``reference_id`` is set)."""
        return self.reference_id is not None


@attrs.frozen(kw_only=True)
class ScalarValueCollection:
    """An immutable page of scalar values plus collection-level, outlier-aware metadata."""

    items: tuple[ScalarValue, ...]
    total_count: int
    facets: tuple[Facet, ...]
    offset: int
    limit: int | None
    had_outliers: bool
    outlier_count: int

    def __iter__(self) -> Iterator[ScalarValue]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def facets_dict(self) -> dict[str, list[str]]:
        """Facets as a plain ``{dimension: [values]}`` mapping."""
        return {f.key: list(f.values) for f in self.facets}

    def to_pandas(self) -> pd.DataFrame:
        """Tidy DataFrame: one row per value, one column per CV dimension present, plus metadata."""
        detection_ran = any(v.is_outlier is not None for v in self.items)
        records = []
        for v in self.items:
            rec: dict[str, Any] = dict(v.dimensions)
            rec.update(
                id=v.id,
                execution_id=v.execution_id,
                execution_group_id=v.execution_group_id,
                kind=v.kind,
                value=v.value,
            )
            if detection_ran:
                rec.update(is_outlier=v.is_outlier, verification_status=v.verification_status)
            if v.diagnostic_slug is not None:
                rec.update(diagnostic_slug=v.diagnostic_slug, provider_slug=v.provider_slug)
            records.append(rec)
        return pd.DataFrame.from_records(records)


@attrs.frozen(kw_only=True)
class SeriesValueCollection:
    """An immutable page of series values plus collection-level metadata."""

    items: tuple[SeriesValue, ...]
    total_count: int
    facets: tuple[Facet, ...]
    offset: int
    limit: int | None

    def __iter__(self) -> Iterator[SeriesValue]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

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
        records = []
        for v in self.items:
            base: dict[str, Any] = dict(v.dimensions)
            base.update(
                id=v.id,
                execution_id=v.execution_id,
                execution_group_id=v.execution_group_id,
                kind=v.kind,
                index_name=v.index_name or "index",
                reference_id=v.reference_id,
            )
            if explode:
                idx = v.index
                for i, value in enumerate(v.values):
                    rec = dict(base)
                    rec.update(value=value, index=idx[i] if idx is not None and i < len(idx) else i)
                    records.append(rec)
            else:
                rec = dict(base)
                rec.update(values=list(v.values), index=list(v.index) if v.index is not None else None)
                records.append(rec)
        return pd.DataFrame.from_records(records)


class ValuesReader:
    """
    Metric-value read domain: scalar/series values and their facets.

    Constructed from a [Database][climate_ref.database.Database], which owns the session and the
    read-only story. All read methods return detached collections that outlive the session.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._db.session

    def _facets(self, base_stmt: Any, entity: Any) -> tuple[Facet, ...]:
        facet_map = collect_facets(self.session, base_stmt, entity)
        return tuple(Facet(key=k, values=tuple(v)) for k, v in facet_map.items())

    def scalar_values(  # noqa: PLR0913
        self,
        filters: MetricValueFilter | None = None,
        *,
        outliers: OutlierPolicy | None = None,
        include_unverified: bool = False,
        offset: int = 0,
        limit: int | None = None,
        with_facets: bool = True,
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

        base_stmt = select_scalar_values(filters)
        facets = self._facets(base_stmt, ScalarMetricValue) if with_facets else ()

        if not policy.enabled:
            # No detection: page and count in SQL so we never materialise the whole table.
            total_count = count_values(self.session, base_stmt)
            load_stmt = base_stmt.options(self._scalar_loader(include_context))
            if limit is not None:
                load_stmt = load_stmt.offset(offset).limit(limit)
            elif offset:
                load_stmt = load_stmt.offset(offset)
            rows = list(self.session.execute(load_stmt).scalars().all())
            items = tuple(
                self._to_scalar_dto(r, False, "verified", include_context, detection_ran=False) for r in rows
            )
            return ScalarValueCollection(
                items=items,
                total_count=total_count,
                facets=facets,
                offset=offset,
                limit=limit,
                had_outliers=False,
                outlier_count=0,
            )

        # Detection enabled: materialise the full filtered set so IQR bounds are globally
        # consistent, then filter and paginate in Python.
        load_stmt = base_stmt.options(self._scalar_loader(include_context))
        rows = list(self.session.execute(load_stmt).scalars().all())

        annotated, outlier_count = detect_scalar_outliers(rows, policy)
        had_outliers = outlier_count > 0
        if not include_unverified:
            annotated = [a for a in annotated if not a.is_outlier]

        total_count = len(annotated)
        page = annotated[offset : offset + limit] if limit is not None else annotated[offset:]
        items = tuple(
            self._to_scalar_dto(
                a.value, a.is_outlier, a.verification_status, include_context, detection_ran=True
            )
            for a in page
        )
        return ScalarValueCollection(
            items=items,
            total_count=total_count,
            facets=facets,
            offset=offset,
            limit=limit,
            had_outliers=had_outliers,
            outlier_count=outlier_count,
        )

    def series_values(
        self,
        filters: MetricValueFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        with_facets: bool = True,
        include_context: bool = False,
    ) -> SeriesValueCollection:
        """
        Query series values with SQL-level pagination.

        The shared index axis is resolved so ``index``/``index_name`` are populated. ``facets`` are
        computed over the full filtered set before pagination.
        """
        filters = filters or MetricValueFilter()

        base_stmt = select_series_values(filters)
        facets = self._facets(base_stmt, SeriesMetricValue) if with_facets else ()
        total_count = count_values(self.session, base_stmt)

        load_stmt = base_stmt.options(self._series_loader(include_context))
        if limit is not None:
            load_stmt = load_stmt.offset(offset).limit(limit)
        elif offset:
            load_stmt = load_stmt.offset(offset)
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

    def _to_scalar_dto(
        self,
        row: ScalarMetricValue,
        is_outlier: bool,
        verification_status: str,
        include_context: bool,
        detection_ran: bool,
    ) -> ScalarValue:
        diagnostic_slug, provider_slug = self._context_slugs(row.execution, include_context)
        dims = dict(row.dimensions)
        return ScalarValue(
            id=row.id,
            execution_id=row.execution_id,
            execution_group_id=row.execution.execution_group_id,
            value=row.value,
            kind=_kind_of(dims),
            dimensions=dims,
            attributes=dict(row.attributes or {}),
            is_outlier=is_outlier if detection_ran else None,
            verification_status=verification_status if detection_ran else None,
            diagnostic_slug=diagnostic_slug,
            provider_slug=provider_slug,
        )

    def _to_series_dto(self, row: SeriesMetricValue, include_context: bool) -> SeriesValue:
        diagnostic_slug, provider_slug = self._context_slugs(row.execution, include_context)
        dims = dict(row.dimensions)
        return SeriesValue(
            id=row.id,
            execution_id=row.execution_id,
            execution_group_id=row.execution.execution_group_id,
            values=list(row.values or []),
            index=list(row.index) if row.index is not None else None,
            index_name=row.index_name,
            reference_id=row.reference_id,
            kind=_kind_of(dims),
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
    [diagnostics][climate_ref.results.values.Reader.diagnostics] for diagnostic reads, and
    [artifacts][climate_ref.results.values.Reader.artifacts] for output path resolution
    (only available when a ``results`` root is supplied).
    """

    def __init__(self, database: Database, results: Path | None = None) -> None:
        self._db = database
        self._results = results

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._db.session

    @functools.cached_property
    def values(self) -> ValuesReader:
        """Metric-value reads (scalar/series/facets)."""
        return ValuesReader(self._db)

    @functools.cached_property
    def executions(self) -> ExecutionsReader:
        """Execution-group and execution reads."""
        return ExecutionsReader(self._db)

    @functools.cached_property
    def datasets(self) -> "DatasetsReader":
        """Dataset reads."""
        from climate_ref.results.datasets import DatasetsReader  # noqa: PLC0415

        return DatasetsReader(self._db)

    @functools.cached_property
    def diagnostics(self) -> "DiagnosticsReader":
        """Diagnostic reads."""
        from climate_ref.results.diagnostics import DiagnosticsReader  # noqa: PLC0415

        return DiagnosticsReader(self._db)

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
