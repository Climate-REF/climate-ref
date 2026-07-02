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
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionGroup
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_str_tuple
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


def select_diagnostics(filter: DiagnosticFilter | None = None) -> Select[Any]:  # noqa: A002
    """
    Build the ``Select`` for diagnostics joined to their provider, with an execution-group count.

    ``execution_group_count`` counts every ``ExecutionGroup`` row for the diagnostic
    (all versions, not scoped to ``promoted_version``),
    so it reflects the diagnostic's full execution history.
    Ordered by ``(Provider.slug, Diagnostic.slug)`` for stable output.
    """
    filter = filter or DiagnosticFilter()  # noqa: A001

    group_count_subquery = (
        select(
            ExecutionGroup.diagnostic_id,
            func.count(ExecutionGroup.id).label("execution_group_count"),
        )
        .group_by(ExecutionGroup.diagnostic_id)
        .subquery()
    )

    stmt = (
        select(
            Provider.slug.label("provider_slug"),
            Diagnostic.slug.label("slug"),
            Diagnostic.name.label("name"),
            Diagnostic.promoted_version.label("promoted_version"),
            func.coalesce(group_count_subquery.c.execution_group_count, 0).label("execution_group_count"),
        )
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .outerjoin(group_count_subquery, Diagnostic.id == group_count_subquery.c.diagnostic_id)
        .order_by(Provider.slug, Diagnostic.slug)
    )

    if filter.provider_contains:
        stmt = stmt.where(or_(*(Provider.slug.ilike(f"%{s.lower()}%") for s in filter.provider_contains)))
    if filter.diagnostic_contains:
        stmt = stmt.where(or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in filter.diagnostic_contains)))

    return stmt


class DiagnosticsReader:
    """
    Diagnostic read domain.

    Constructed from a [Database][climate_ref.database.Database], which owns the session.
    All read methods return detached DTOs that outlive the session.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._db.session

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
        filter: DiagnosticFilter | None = None,  # noqa: A002
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
        filter = filter or DiagnosticFilter()  # noqa: A001

        base_stmt = select_diagnostics(filter)
        count_stmt = select(func.count()).select_from(base_stmt.subquery())
        total_count = self.session.execute(count_stmt).scalar_one()

        stmt = base_stmt
        if limit is not None:
            stmt = stmt.offset(offset).limit(limit)
        elif offset:
            stmt = stmt.offset(offset)

        rows = self.session.execute(stmt).all()

        stats = self.stats(
            provider_contains=filter.provider_contains,
            diagnostic_contains=filter.diagnostic_contains,
        )
        stats_by_key = {(s.provider, s.diagnostic): s for s in stats}
        items = tuple(self._to_view(row, stats_by_key) for row in rows)

        return DiagnosticCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def stats(
        self,
        *,
        diagnostic_contains: Sequence[str] | None = None,
        provider_contains: Sequence[str] | None = None,
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
            diagnostic_contains=diagnostic_contains, provider_contains=provider_contains
        )
        rows = self.session.execute(stmt).all()
        return tuple(
            ExecutionStats(
                provider=row.provider,
                diagnostic=row.diagnostic,
                running=row.running,
                failed=row.failed,
                successful=row.successful,
                not_started=row.not_started,
                dirty=row.dirty,
                total=row.total,
            )
            for row in rows
        )
