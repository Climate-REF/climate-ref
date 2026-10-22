"""
Typed, detached read surface for diagnostic metadata.

[DiagnosticsReader][climate_ref.results.diagnostics.DiagnosticsReader] is reached via
[Reader.diagnostics][climate_ref.results.values.Reader.diagnostics].
It executes the [select_diagnostics][climate_ref.results.diagnostics.select_diagnostics] query
(joining ``Diagnostic -> Provider`` and counting execution groups per diagnostic)
and maps the rows into detached DTOs that outlive the session.
"""

from collections.abc import Mapping
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import ColumnElement, Select, func, literal, select
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, MetricValueType
from climate_ref.models.provider import Provider
from climate_ref.results._page import Page
from climate_ref.results._query import (
    _Unindexed,
    apply_execution_scope,
    count_values,
    paginate,
)
from climate_ref.results._stats import rows_to_execution_stats
from climate_ref.results.executions import ExecutionStats, select_execution_statistics
from climate_ref.results.filters import DiagnosticFilter, ExecutionScope, promoted_version_clause


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
class DiagnosticCollection(Page[DiagnosticView]):
    """An immutable page of diagnostics plus collection-level metadata."""

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

    return stmt.where(*filters.diagnostic_clauses())


def select_diagnostics_with_values(
    value_type: MetricValueType,
    filters: DiagnosticFilter | None = None,
    *,
    execution_group_where: ColumnElement[bool] | None = None,
    scope: ExecutionScope = ExecutionScope(include_retracted=False),
) -> Select[Any]:
    """
    Build the ``Select`` for diagnostics with at least one value of ``value_type`` at the promoted version.

    ``scope`` picks which executions count.
    By default retracted executions do not count, matching
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
            promoted_version_clause(),
            # Without table statistics SQLite picks ix_metric_value_type and reads every series row.
            _Unindexed(MetricValue.type.expression) == value_type,
        )
    )
    has_values = apply_execution_scope(has_values, scope)
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
    return stmt.where(*filters.diagnostic_clauses())


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
        with_count: bool = True,
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
        total_count = count_values(self.session, base_stmt) if with_count else None

        rows = self.session.execute(paginate(base_stmt, offset, limit)).all()

        stats = self.stats(filters)
        stats_by_key = {(s.provider, s.diagnostic): s for s in stats}
        items = tuple(self._to_view(row, stats_by_key) for row in rows)

        return DiagnosticCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def with_values(
        self,
        value_type: MetricValueType,
        filters: DiagnosticFilter | None = None,
        *,
        scope: ExecutionScope = ExecutionScope(include_retracted=False),
    ) -> frozenset[tuple[str, str]]:
        """
        Find the ``(provider, diagnostic)`` pairs with a value of ``value_type`` at the promoted version.

        See [select_diagnostics_with_values][climate_ref.results.diagnostics.select_diagnostics_with_values].
        """
        rows = self.session.execute(select_diagnostics_with_values(value_type, filters, scope=scope)).all()
        return frozenset((row.provider_slug, row.slug) for row in rows)

    def stats(self, filters: DiagnosticFilter | None = None) -> tuple[ExecutionStats, ...]:
        """
        Per-(provider, diagnostic) execution status counts.

        Reuses [select_execution_statistics][climate_ref.results.executions.select_execution_statistics]
        rather than duplicating the aggregate SQL, so status-count logic stays defined once.
        This has the same signature and behaviour as
        [ExecutionsReader.statistics][climate_ref.results.executions.ExecutionsReader.statistics],
        just reachable under the diagnostics domain.
        """
        stmt = select_execution_statistics(filters)
        return rows_to_execution_stats(self.session.execute(stmt).all())
