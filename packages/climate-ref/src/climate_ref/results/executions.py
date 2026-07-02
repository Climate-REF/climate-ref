"""
Typed, ORM-free surface for execution-group and execution results.

[ExecutionsReader][climate_ref.results.executions.ExecutionsReader] is reached via
[Reader.executions][climate_ref.results.values.Reader.executions].
It wraps the group+latest query
([get_execution_group_and_latest_filtered][climate_ref.models.execution.get_execution_group_and_latest_filtered])
and the per-(provider, diagnostic) status aggregate mapping ORM rows to frozen DTOs (Data Transfer Objects).
These DTOs are detached from the database so outlive the database session.

Two write/recovery paths deliberately stay on the ORM helper instead of this reader:

- ``cli/executions.py::delete_groups``
    (needs live ORM objects to cascade-delete related rows and remove output directories)
- ``executor/reingest.py::get_executions_for_reingest``
    (needs ``include_superseded=True`` plus the *oldest* execution in a group,
    not this reader's "latest" definition).
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
    get_execution_group_and_latest_filtered,
)
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_str_tuple
from climate_ref.results._query import latest_execution_for_group


def _as_facets(
    value: Mapping[str, Sequence[str]] | None,
) -> Mapping[str, tuple[str, ...]] | None:
    """Copy a facets mapping into an immutable ``dict`` of immutable ``tuple`` values."""
    if value is None:
        return None
    return {k: tuple(v) for k, v in value.items()}


@attrs.frozen(kw_only=True)
class ExecutionGroupFilter:
    """
    Declarative filter over execution groups.

    Every field is optional; ``None`` means "do not constrain on this axis".
    This mirrors exactly what [get_execution_group_and_latest_filtered]
    [climate_ref.models.execution.get_execution_group_and_latest_filtered] supports today --
    ``diagnostic_contains``/``provider_contains`` are case-insensitive substring matches (OR-combined),
    ``facets`` is the selector-facet map consumed by the helper's Python-side matching.

    Exact-match ``diagnostic_slug``/``provider_slug`` are a documented follow-up,
    as the underlying helper only does substring matching today,
    so adding exact match means extending the helper first.
    """

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    dirty: bool | None = None
    successful: bool | None = None
    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    include_superseded: bool = False


@attrs.frozen(kw_only=True)
class ExecutionView:
    """A single execution, detached from the ORM."""

    id: int
    execution_group_id: int
    successful: bool | None
    dataset_hash: str
    retracted: bool
    output_fragment: str
    path: str | None
    provider_version: str | None
    created_at: Any
    updated_at: Any


@attrs.frozen(kw_only=True)
class ExecutionGroupView:
    """An execution group, detached from the ORM, with its per-group latest execution (if any)."""

    id: int
    key: str
    diagnostic_slug: str
    provider_slug: str
    dirty: bool
    diagnostic_version: int
    selectors: Mapping[str, Any]
    created_at: Any
    updated_at: Any
    # Reflects the helper's `max(created_at)` outer join
    # (models/execution.py::get_execution_group_and_latest), which can duplicate a group on an
    # exact `created_at` tie. This may differ from `ExecutionsReader.latest_execution()`, which
    # tie-breaks by `created_at DESC, id DESC` (`_query.latest_execution_for_group`). Both
    # behaviours are intentional for their respective consumers (`groups()` vs `latest_execution()`)
    # -- do not unify them.
    latest: ExecutionView | None

    @property
    def successful(self) -> bool | None:
        """Convenience mirror of ``latest.successful`` (matches the CLI's ``successful`` column)."""
        return self.latest.successful if self.latest else None


@attrs.frozen(kw_only=True)
class ExecutionStats:
    """Per-(provider, diagnostic) execution status counts, detached from the ORM."""

    provider: str
    diagnostic: str
    running: int
    failed: int
    successful: int
    not_started: int
    dirty: int
    total: int


@attrs.frozen(kw_only=True)
class OutputView:
    """A single registered execution output, detached from the ORM."""

    execution_id: int
    output_type: str
    filename: str | None
    short_name: str | None
    long_name: str | None
    description: str | None
    dimensions: Mapping[str, str]


@attrs.frozen(kw_only=True)
class ExecutionGroupCollection:
    """An immutable page of execution groups plus collection-level metadata."""

    items: tuple[ExecutionGroupView, ...]
    total_count: int
    offset: int
    limit: int | None

    def __iter__(self) -> Iterator[ExecutionGroupView]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def to_pandas(self) -> pd.DataFrame:
        """
        DataFrame mirroring the ``list-groups`` CLI columns.

        Columns are emitted explicitly (``id, key, provider, diagnostic, dirty, successful,
        created_at, updated_at, selectors``) even when the collection is empty, so callers can
        select columns / build an empty table without special-casing.
        """
        columns = [
            "id",
            "key",
            "provider",
            "diagnostic",
            "dirty",
            "successful",
            "created_at",
            "updated_at",
            "selectors",
        ]
        records = [
            {
                "id": eg.id,
                "key": eg.key,
                "provider": eg.provider_slug,
                "diagnostic": eg.diagnostic_slug,
                "dirty": eg.dirty,
                "successful": eg.successful,
                "created_at": eg.created_at,
                "updated_at": eg.updated_at,
                "selectors": dict(eg.selectors),
            }
            for eg in self.items
        ]
        return pd.DataFrame.from_records(records, columns=columns)


def select_execution_statistics(
    *,
    diagnostic_contains: Sequence[str] | None = None,
    provider_contains: Sequence[str] | None = None,
) -> Select[Any]:
    """
    Build the per-(provider, diagnostic) status-count aggregate ``Select``.

    Moved out of ``cli/executions.py::stats`` so the aggregate is reusable plumbing.
    Only ``diagnostic_contains``/``provider_contains`` are accepted
    (not the full [ExecutionGroupFilter][climate_ref.results.executions.ExecutionGroupFilter])
    so ``dirty`` / ``successful`` / ``facets`` are never silently ignored
    on an aggregate that cannot honour them.

    Status definitions (unchanged from the CLI):

    * ``running`` -- latest execution exists and ``successful IS NULL``.
    * ``failed`` -- latest execution exists and ``successful IS False``.
    * ``successful`` -- latest execution exists and ``successful IS True``.
    * ``not_started`` -- no execution exists for the group.
    * ``dirty`` -- the group's ``dirty`` flag is set.

    Only execution groups whose ``diagnostic_version`` matches the diagnostic's
    ``promoted_version`` are counted (same promoted-version gate as
    [get_execution_group_and_latest_filtered][climate_ref.models.execution.get_execution_group_and_latest_filtered]).
    """
    latest_exec_subquery = (
        select(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("latest_created_at"),
        )
        .group_by(Execution.execution_group_id)
        .subquery()
    )

    stmt = (
        select(
            Provider.slug.label("provider"),
            Diagnostic.slug.label("diagnostic"),
            func.count().label("total"),
            func.sum(
                case(
                    (Execution.successful.is_(None) & Execution.id.isnot(None), 1),
                    else_=0,
                )
            ).label("running"),
            func.sum(case((Execution.successful.is_(False), 1), else_=0)).label("failed"),
            func.sum(case((Execution.successful.is_(True), 1), else_=0)).label("successful"),
            func.sum(case((Execution.id.is_(None), 1), else_=0)).label("not_started"),
            func.sum(case((ExecutionGroup.dirty.is_(True), 1), else_=0)).label("dirty"),
        )
        .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
        .where(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .outerjoin(latest_exec_subquery, ExecutionGroup.id == latest_exec_subquery.c.execution_group_id)
        .outerjoin(
            Execution,
            (Execution.execution_group_id == ExecutionGroup.id)
            & (Execution.created_at == latest_exec_subquery.c.latest_created_at),
        )
        .group_by(Provider.slug, Diagnostic.slug)
        .order_by(Provider.slug, Diagnostic.slug)
    )

    if diagnostic_contains:
        stmt = stmt.where(or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in diagnostic_contains)))
    if provider_contains:
        stmt = stmt.where(or_(*(Provider.slug.ilike(f"%{s.lower()}%") for s in provider_contains)))

    return stmt


def select_execution_outputs(execution_id: int) -> Select[Any]:
    """
    Build the ``Select`` for one execution's registered ``ExecutionOutput`` rows.

    Ordered by ``(output_type, id)`` for stable, grouped output -- there is no prior consumer
    ordering to preserve, since ``inspect`` did not list these rows before.
    """
    return (
        select(ExecutionOutput)
        .where(ExecutionOutput.execution_id == execution_id)
        .order_by(ExecutionOutput.output_type, ExecutionOutput.id)
    )


class ExecutionsReader:
    """
    Execution-group and execution read domain.

    Constructed from a [Database][climate_ref.database.Database],
    which owns the session and the read-only story.
    All read methods return detached DTOs that outlive the session.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._db.session

    def _to_execution_view(self, execution: Execution) -> ExecutionView:
        return ExecutionView(
            id=execution.id,
            execution_group_id=execution.execution_group_id,
            successful=execution.successful,
            dataset_hash=execution.dataset_hash,
            retracted=execution.retracted,
            output_fragment=execution.output_fragment,
            path=execution.path,
            provider_version=execution.provider_version,
            created_at=execution.created_at,
            updated_at=execution.updated_at,
        )

    def _to_group_view(self, eg: ExecutionGroup, latest: Execution | None) -> ExecutionGroupView:
        return ExecutionGroupView(
            id=eg.id,
            key=eg.key,
            diagnostic_slug=eg.diagnostic.slug,
            provider_slug=eg.diagnostic.provider.slug,
            dirty=eg.dirty,
            diagnostic_version=eg.diagnostic_version,
            selectors=dict(eg.selectors),
            created_at=eg.created_at,
            updated_at=eg.updated_at,
            latest=self._to_execution_view(latest) if latest is not None else None,
        )

    def groups(
        self,
        filters: ExecutionGroupFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> ExecutionGroupCollection:
        """
        Query execution groups with their per-group latest execution.

        Wraps [get_execution_group_and_latest_filtered]
        [climate_ref.models.execution.get_execution_group_and_latest_filtered] exactly: the full
        filtered list is materialised, ``total_count`` is its length, and the page is a Python
        slice (``all[offset:offset + limit]``). This preserves today's ``list-groups`` pagination
        and ordering exactly -- no SQL-level pagination or additional ``order_by`` is introduced
        here.
        """
        filters = filters or ExecutionGroupFilter()

        all_results = get_execution_group_and_latest_filtered(
            self.session,
            diagnostic_filters=list(filters.diagnostic_contains) if filters.diagnostic_contains else None,
            provider_filters=list(filters.provider_contains) if filters.provider_contains else None,
            facet_filters={k: list(v) for k, v in filters.facets.items()} if filters.facets else None,
            dirty=filters.dirty,
            successful=filters.successful,
            include_superseded=filters.include_superseded,
        )

        total_count = len(all_results)
        page = all_results[offset : offset + limit] if limit is not None else all_results[offset:]
        items = tuple(self._to_group_view(eg, latest) for eg, latest in page)

        return ExecutionGroupCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def statistics(
        self,
        *,
        diagnostic_contains: Sequence[str] | None = None,
        provider_contains: Sequence[str] | None = None,
    ) -> tuple[ExecutionStats, ...]:
        """Execute the ``select_execution_statistics`` builder and map rows to ``ExecutionStats``."""
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

    def latest_execution(self, execution_group_id: int) -> ExecutionView | None:
        """
        Return the most recent execution for a group.

        Wraps [latest_execution_for_group][climate_ref.results._query.latest_execution_for_group],
        which tie-breaks by ``created_at DESC, id DESC``. See the note on
        [ExecutionGroupView.latest][climate_ref.results.executions.ExecutionGroupView] for how this
        differs from the per-group latest returned by ``groups()`` on an exact timestamp tie.
        """
        execution = latest_execution_for_group(self.session, execution_group_id)
        return self._to_execution_view(execution) if execution is not None else None

    def outputs(self, execution_id: int) -> tuple[OutputView, ...]:
        """Execute the ``select_execution_outputs`` builder and map rows to ``OutputView``."""
        stmt = select_execution_outputs(execution_id)
        rows = self.session.execute(stmt).scalars().all()
        return tuple(
            OutputView(
                execution_id=row.execution_id,
                output_type=row.output_type.value,
                filename=row.filename,
                short_name=row.short_name,
                long_name=row.long_name,
                description=row.description,
                dimensions=dict(row.dimensions),
            )
            for row in rows
        )
