"""
Typed, detached read surface for diagnostic metadata.

[DiagnosticsReader][climate_ref.results.diagnostics.DiagnosticsReader] is reached via
[Reader.diagnostics][climate_ref.results.values.Reader.diagnostics].
It executes the [select_diagnostics][climate_ref.results.diagnostics.select_diagnostics] query
(joining ``Diagnostic -> Provider`` and counting execution groups per diagnostic)
and maps the rows into detached DTOs that outlive the session.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import ColumnElement, Select, func, literal, or_, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.functions import FunctionElement

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, MetricValueType
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_str_tuple
from climate_ref.results._query import ExecutionScope, InputDatasetFilter, apply_execution_scope
from climate_ref.results._stats import rows_to_execution_stats
from climate_ref.results.executions import ExecutionStats, select_execution_statistics


@attrs.frozen(kw_only=True)
class DiagnosticFilter:
    """
    Declarative filter over diagnostics.

    Every field is optional; ``None`` means "do not constrain on this axis".
    ``diagnostic_contains``/``provider_contains`` are case-insensitive substring matches
    (OR-combined within each field), matching the semantics used by
    [ExecutionGroupFilter][climate_ref.results.executions.ExecutionGroupFilter].
    """

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on provider slug (OR-combined)."""

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on diagnostic slug (OR-combined)."""

    input_datasets: InputDatasetFilter | None = None
    """
    Keep only execution groups with an execution that ran against matching datasets.

    This scopes the counts as well, and drops diagnostics left with no groups.
    """


@attrs.frozen(kw_only=True)
class DiagnosticView:
    """
    A single diagnostic with information about the currently promoted version and execution-group counts.

    ``execution_group_count`` counts execution groups across every diagnostic version,
    while ``successful``/``inflight``/``total`` are scoped to the currently ``promoted_version``.
    """

    provider_slug: str
    """Owning provider's slug."""

    slug: str
    """The diagnostic's own slug, unique within its provider."""

    name: str
    """Human-readable diagnostic name."""

    promoted_version: int
    """Diagnostic version currently promoted for production."""

    execution_group_count: int
    """Execution groups for the diagnostic across every version."""

    successful: int
    """Promoted-version execution groups whose latest execution succeeded."""

    inflight: int
    """Promoted-version execution groups whose latest execution is still running (outcome not yet known)."""

    total: int
    """Execution groups at the promoted version."""


@attrs.frozen(kw_only=True)
class DiagnosticCollection:
    """
    An immutable page of diagnostics plus collection-level metadata.

    ``total_count`` is the number of diagnostics matching the filter *before* pagination, so a
    caller can tell there are more rows than the returned page. ``offset``/``limit`` echo back the
    pagination applied to produce ``items`` (``limit`` is ``None`` when the whole result was
    returned).
    """

    items: tuple[DiagnosticView, ...]
    """The diagnostics on this page."""

    total_count: int
    """Total diagnostics matching the filter before ``offset``/``limit``."""

    offset: int
    """Rows skipped before this page."""

    limit: int | None
    """Page size requested, or ``None`` when the whole result was returned."""

    def __iter__(self) -> Iterator[DiagnosticView]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def to_pandas(self) -> pd.DataFrame:
        """
        DataFrame mirroring the ``diagnostics list`` CLI columns.

        Columns are emitted explicitly (``provider, diagnostic, name, promoted_version,
        execution_group_count, successful, inflight, total``) even when the collection is empty, so
        callers can select columns / build an empty table without special-casing.
        """
        columns = [
            "provider",
            "diagnostic",
            "name",
            "promoted_version",
            "execution_group_count",
            "successful",
            "inflight",
            "total",
        ]
        records = [
            {
                "provider": d.provider_slug,
                "diagnostic": d.slug,
                "name": d.name,
                "promoted_version": d.promoted_version,
                "execution_group_count": d.execution_group_count,
                "successful": d.successful,
                "inflight": d.inflight,
                "total": d.total,
            }
            for d in self.items
        ]
        return pd.DataFrame.from_records(records, columns=columns)


def select_diagnostics(filters: DiagnosticFilter | None = None) -> Select[Any]:
    """
    Build the ``Select`` for diagnostics joined to their provider, with an execution-group count.

    ``execution_group_count`` counts every ``ExecutionGroup`` row for the diagnostic
    (all versions, not scoped to ``promoted_version``),
    so it reflects the diagnostic's full execution history.
    With ``filters.input_datasets`` it counts only matching groups, and diagnostics without one are dropped.
    Ordered by ``(Provider.slug, Diagnostic.slug, Diagnostic.id)``.
    The diagnostic primary key is the final tiebreak so SQL pagination is deterministic across pages.
    """
    filters = filters or DiagnosticFilter()

    group_count_stmt = select(
        ExecutionGroup.diagnostic_id,
        func.count(ExecutionGroup.id).label("execution_group_count"),
    ).group_by(ExecutionGroup.diagnostic_id)
    if filters.input_datasets:
        group_count_stmt = group_count_stmt.where(filters.input_datasets.execution_group_clause())
    group_count_subquery = group_count_stmt.subquery()

    stmt = (
        select(
            Provider.slug.label("provider_slug"),
            Diagnostic.slug.label("slug"),
            Diagnostic.name.label("name"),
            Diagnostic.promoted_version.label("promoted_version"),
            func.coalesce(group_count_subquery.c.execution_group_count, 0).label("execution_group_count"),
        )
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .join(
            group_count_subquery,
            Diagnostic.id == group_count_subquery.c.diagnostic_id,
            isouter=filters.input_datasets is None,
        )
        .order_by(Provider.slug, Diagnostic.slug, Diagnostic.id)
    )

    return _apply_filters(stmt, filters)


def _apply_filters(stmt: Select[Any], filters: DiagnosticFilter) -> Select[Any]:
    if filters.provider_contains:
        stmt = stmt.where(or_(*(Provider.slug.ilike(f"%{s.lower()}%") for s in filters.provider_contains)))
    if filters.diagnostic_contains:
        stmt = stmt.where(
            or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in filters.diagnostic_contains))
        )
    return stmt


class _Unindexed(FunctionElement[Any]):
    """A column that SQLite must not use an index on, written ``+column`` in SQLite."""

    inherit_cache = True

    def __init__(self, column: ColumnElement[Any]) -> None:
        super().__init__(column)
        self.type = column.type


@compiles(_Unindexed)
def _compile_unindexed(element: _Unindexed, compiler: SQLCompiler, **kw: Any) -> str:
    prefix = "+" if compiler.dialect.name == "sqlite" else ""
    return prefix + compiler.process(element.clauses, **kw)


def select_diagnostics_with_values(
    value_type: MetricValueType,
    filters: DiagnosticFilter | None = None,
    *,
    execution_group_where: ColumnElement[bool] | None = None,
) -> Select[Any]:
    """
    Build the ``Select`` for diagnostics with at least one value of ``value_type`` at the promoted version.

    Values from retracted executions do not count, matching
    [Reader.values][climate_ref.results.values.Reader.values].

    Selects ``id``, ``provider_slug`` and ``slug``, ordered like
    [select_diagnostics][climate_ref.results.diagnostics.select_diagnostics].
    ``execution_group_where`` further restricts which execution groups count.
    """
    filters = filters or DiagnosticFilter()

    has_values = (
        select(literal(1))
        .select_from(ExecutionGroup)
        .join(Execution, Execution.execution_group_id == ExecutionGroup.id)
        .join(MetricValue, MetricValue.execution_id == Execution.id)
        .where(
            ExecutionGroup.diagnostic_id == Diagnostic.id,
            ExecutionGroup.diagnostic_version == Diagnostic.promoted_version,
            # Without table statistics SQLite picks ix_metric_value_type and reads every series row.
            _Unindexed(MetricValue.type.expression) == value_type,
        )
    )
    has_values = apply_execution_scope(has_values, ExecutionScope(include_retracted=False))
    if execution_group_where is not None:
        has_values = has_values.where(execution_group_where)
    if filters.input_datasets:
        has_values = has_values.where(filters.input_datasets.execution_clause())

    stmt = (
        select(
            Diagnostic.id.label("id"),
            Provider.slug.label("provider_slug"),
            Diagnostic.slug.label("slug"),
        )
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .where(has_values.exists())
        .order_by(Provider.slug, Diagnostic.slug, Diagnostic.id)
    )
    return _apply_filters(stmt, filters)


class DiagnosticsReader:
    """
    Diagnostic read domain.

    Constructed from a [Database][climate_ref.database.Database], which owns the session.
    All read methods return detached DTOs that outlive the session.
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._session if self._session is not None else self._db.session

    def _to_view(self, row: Any, stats_by_key: Mapping[tuple[str, str], ExecutionStats]) -> DiagnosticView:
        stat = stats_by_key.get((row.provider_slug, row.slug))
        return DiagnosticView(
            provider_slug=row.provider_slug,
            slug=row.slug,
            name=row.name,
            promoted_version=row.promoted_version,
            execution_group_count=row.execution_group_count,
            successful=stat.successful if stat is not None else 0,
            inflight=stat.running if stat is not None else 0,
            total=stat.total if stat is not None else 0,
        )

    def list(
        self,
        filters: DiagnosticFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> DiagnosticCollection:
        """
        Query diagnostics, with their provider, execution-group count, and promoted-version stats.

        Pagination is applied in SQL (``offset``/``limit``), with ``total_count`` computed from a
        separate unpaged count query over the same filtered statement.
        Promoted-version status counts (``successful``/``inflight``/``total``) are merged in from
        [stats][climate_ref.results.diagnostics.DiagnosticsReader.stats],
        and keyed by ``(provider, diagnostic)`` so status-count logic stays defined once.

        Diagnostics with no execution groups at the promoted version get zeros.
        """
        filters = filters or DiagnosticFilter()

        base_stmt = select_diagnostics(filters)
        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        total_count = self.session.execute(count_stmt).scalar_one()

        stmt = base_stmt
        if limit is not None:
            stmt = stmt.offset(offset).limit(limit)
        elif offset:
            stmt = stmt.offset(offset)

        rows = self.session.execute(stmt).all()

        stats = self.stats(
            provider_contains=filters.provider_contains,
            diagnostic_contains=filters.diagnostic_contains,
            input_datasets=filters.input_datasets,
        )
        stats_by_key = {(s.provider, s.diagnostic): s for s in stats}
        items = tuple(self._to_view(row, stats_by_key) for row in rows)

        return DiagnosticCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def with_values(
        self, value_type: MetricValueType, filters: DiagnosticFilter | None = None
    ) -> frozenset[tuple[str, str]]:
        """
        Find the ``(provider, diagnostic)`` pairs with a value of ``value_type`` at the promoted version.

        See [select_diagnostics_with_values][climate_ref.results.diagnostics.select_diagnostics_with_values].
        """
        rows = self.session.execute(select_diagnostics_with_values(value_type, filters)).all()
        return frozenset((row.provider_slug, row.slug) for row in rows)

    def stats(
        self,
        *,
        diagnostic_contains: Sequence[str] | None = None,
        provider_contains: Sequence[str] | None = None,
        input_datasets: InputDatasetFilter | None = None,
    ) -> tuple[ExecutionStats, ...]:
        """
        Per-(provider, diagnostic) execution status counts.

        Reuses [select_execution_statistics][climate_ref.results.executions.select_execution_statistics]
        rather than duplicating the aggregate SQL, so status-count logic stays defined once.
        This has the same signature and behaviour as
        [ExecutionsReader.statistics][climate_ref.results.executions.ExecutionsReader.statistics],
        just reachable under the diagnostics domain.
        """
        stmt = select_execution_statistics(
            diagnostic_contains=diagnostic_contains,
            provider_contains=provider_contains,
            input_datasets=input_datasets,
        )
        return rows_to_execution_stats(self.session.execute(stmt).all())
