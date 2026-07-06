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

A common set of ordering (``created_at DESC, id DESC``) is used to ensure consistent ordering and tie breaks.

"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.dataset_query import _as_facets
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
from climate_ref.results._stats import rows_to_execution_stats


@attrs.frozen(kw_only=True)
class ExecutionGroupFilter:
    """
    Declarative filter over execution groups.

    Every field is optional; ``None`` means "do not constrain on this axis".
    This mirrors exactly what [get_execution_group_and_latest_filtered]
    [climate_ref.models.execution.get_execution_group_and_latest_filtered] supports --
    ``diagnostic_contains``/``provider_contains`` are case-insensitive substring matches (OR-combined),
    ``facets`` is the selector-facet map consumed by the helper's Python-side matching,
    and ``successful`` vs ``latest_successful`` expose the helper's post-rank / pre-rank success filters.

    Exact-match ``diagnostic_slug``/``provider_slug`` are a documented follow-up,
    as the underlying helper only does substring matching today,
    so adding exact match means extending the helper first.
    """

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on diagnostic slug (OR-combined)."""

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on provider slug (OR-combined)."""

    dirty: bool | None = None
    """Constrain on the group's ``dirty`` flag."""

    successful: bool | None = None
    """
    Post-rank filter on the *winning* execution: keep a group only if its latest execution matches.

    ``True`` keeps groups whose latest execution succeeded.
    ``False`` keeps groups whose latest execution failed, is in progress, or does not exist yet.
    This does not change which execution is considered latest -- contrast ``latest_successful``.
    """

    latest_successful: bool | None = None
    """
    Pre-rank population filter: change which execution is chosen as "latest" before ranking.

    ``True`` ranks only over successful executions,
    so a group's ``latest`` becomes its most recent *successful* run
    (surfacing an earlier success even when a later run failed).
    ``False`` ranks only over unsuccessful / in-progress runs.

    ``None`` (default) ranks over all executions.
    Composes with ``successful`` but answers a different question.
    """

    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """Selector-facet map matched against each group's ``selectors``."""

    include_superseded: bool = False
    """Include execution groups whose ``diagnostic_version`` is not the diagnostic's promoted version."""


@attrs.frozen(kw_only=True)
class ExecutionView:
    """A single execution, detached from the ORM."""

    id: int
    """Primary key of the underlying ``Execution`` row."""

    execution_group_id: int
    """The execution group this execution belongs to."""

    successful: bool | None
    """``True``/``False`` once the execution has finished, ``None`` while still running."""

    dataset_hash: str
    """Hash of the input datasets used for this execution."""

    retracted: bool
    """Whether this execution has been retracted."""

    output_fragment: str
    """Relative directory storing this execution's output, once moved to the final output directory."""

    path: str | None
    """Path to the output bundle, relative to the diagnostic execution result output directory."""

    provider_version: str | None
    """Version of the diagnostic provider that produced this execution."""

    created_at: Any
    """Timestamp the execution was created."""

    updated_at: Any
    """Timestamp the execution was last updated."""


@attrs.frozen(kw_only=True)
class ExecutionGroupView:
    """An execution group, detached from the ORM, with its per-group latest execution (if any)."""

    id: int
    """Primary key of the underlying ``ExecutionGroup`` row."""

    key: str
    """Stable key identifying the group's input-dataset selection."""

    diagnostic_slug: str
    """Owning diagnostic's slug."""

    provider_slug: str
    """Owning provider's slug."""

    dirty: bool
    """Whether the group is flagged for re-execution (e.g. new data has arrived)."""

    diagnostic_version: int
    """Diagnostic version this group was executed against."""

    selectors: Mapping[str, Any]
    """The selector-facet values used to build this group's input-dataset selection."""

    created_at: Any
    """Timestamp the group was created."""

    updated_at: Any
    """Timestamp the group was last updated."""

    latest: ExecutionView | None
    """
    The group's most recent execution, or ``None`` if it has never been executed.

    These groups are ranked by ``created_at DESC, id DESC``.
    This matches ``ExecutionsReader.latest_execution()`` and the ``statistics()`` aggregates.
    """

    @property
    def successful(self) -> bool | None:
        """Convenience mirror of ``latest.successful`` (matches the CLI's ``successful`` column)."""
        return self.latest.successful if self.latest else None


@attrs.frozen(kw_only=True)
class ExecutionStats:
    """Per-(provider, diagnostic) execution status counts, detached from the ORM."""

    provider: str
    """Provider slug."""

    diagnostic: str
    """Diagnostic slug."""

    running: int
    """Groups whose latest execution exists with ``successful IS NULL``."""

    failed: int
    """Groups whose latest execution exists with ``successful IS False``."""

    successful: int
    """Groups whose latest execution exists with ``successful IS True``."""

    not_started: int
    """Groups with no execution yet."""

    dirty: int
    """Groups flagged for re-execution."""

    total: int
    """Total execution groups counted (at the diagnostic's promoted version)."""


@attrs.frozen(kw_only=True)
class OutputView:
    """A single registered execution output, detached from the ORM."""

    execution_id: int
    """The execution that registered this output."""

    output_type: str
    """The output's type (e.g. plot, data, HTML), which determines how it is displayed."""

    filename: str | None
    """Path to the output, relative to the diagnostic execution result output directory."""

    short_name: str | None
    """Short key of the output, unique for a given result and output type."""

    long_name: str | None
    """Human-readable name for the output."""

    description: str | None
    """Free-text description of the output."""

    dimensions: Mapping[str, str]
    """CV dimension values associated with this output."""


@attrs.frozen(kw_only=True)
class ExecutionGroupCollection:
    """An immutable page of execution groups plus collection-level metadata."""

    items: tuple[ExecutionGroupView, ...]
    """The execution groups on this page."""

    total_count: int
    """Total execution groups matching the filter before ``offset``/``limit``."""

    offset: int
    """Rows skipped before this page."""

    limit: int | None
    """Page size requested, or ``None`` when the whole result was returned."""

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
    # Resolve to a single latest execution id per group, tie-broken by max id (matching the
    # created_at DESC, id DESC tie-break used elsewhere -- see `_query.latest_execution_for_group`).
    # Joining on `execution_group_id` + `created_at` alone (without this tie-break) can match two
    # rows for a group whose executions share an exact `created_at`, double-counting that group in
    # every aggregate column below.
    latest_exec_subquery = (
        select(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("latest_created_at"),
        )
        .group_by(Execution.execution_group_id)
        .subquery()
    )
    latest_exec_id_subquery = (
        select(func.max(Execution.id).label("latest_execution_id"))
        .join(
            latest_exec_subquery,
            (Execution.execution_group_id == latest_exec_subquery.c.execution_group_id)
            & (Execution.created_at == latest_exec_subquery.c.latest_created_at),
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
        .outerjoin(
            Execution,
            Execution.id.in_(select(latest_exec_id_subquery.c.latest_execution_id))
            & (Execution.execution_group_id == ExecutionGroup.id),
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
        [climate_ref.models.execution.get_execution_group_and_latest_filtered],
        which returns a materialised list.
        The ``facets`` axis is matched in Python (against each group's ``selectors``),
        and the helper materialises unconditionally,
        so pagination here is a Python slice over the fully materialised list rather than SQL.

        ``total_count`` is the post-filter length,
        and the results are ordered by ``id`` ascending (the group primary key)
        before slicing so paging is deterministic even when two groups share a ``created_at``.
        """
        # Pushing SQL pagination through the helper would be wrong whenever ``facets`` is set,
        # since the Python facet filter runs after the SQL query.
        filters = filters or ExecutionGroupFilter()

        all_results = get_execution_group_and_latest_filtered(
            self.session,
            diagnostic_filters=list(filters.diagnostic_contains) if filters.diagnostic_contains else None,
            provider_filters=list(filters.provider_contains) if filters.provider_contains else None,
            facet_filters={k: list(v) for k, v in filters.facets.items()} if filters.facets else None,
            dirty=filters.dirty,
            successful=filters.successful,
            latest_successful=filters.latest_successful,
            include_superseded=filters.include_superseded,
        )
        all_results = sorted(all_results, key=lambda pair: pair[0].id)

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
        return rows_to_execution_stats(self.session.execute(stmt).all())

    def latest_execution(self, execution_group_id: int) -> ExecutionView | None:
        """
        Return the most recent execution for a group.

        Wraps [latest_execution_for_group][climate_ref.results._query.latest_execution_for_group],
        which tie-breaks using the same ranking ``groups()`` and ``statistics()`` use
        (``created_at DESC, id DESC``).
        """
        execution = latest_execution_for_group(self.session, execution_group_id)
        return self._to_execution_view(execution) if execution is not None else None

    def group(self, execution_group_id: int) -> ExecutionGroupView | None:
        """
        Fetch one execution group by id, with its latest execution resolved.

        Returns ``None`` when no group has that id.
        """
        eg = self.session.get(ExecutionGroup, execution_group_id)
        if eg is None:
            return None
        latest = latest_execution_for_group(self.session, execution_group_id)
        return self._to_group_view(eg, latest)

    def execution(self, execution_id: int) -> ExecutionView | None:
        """Fetch one execution by id, or ``None`` when no execution has that id."""
        execution = self.session.get(Execution, execution_id)
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
