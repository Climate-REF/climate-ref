"""
Builders over the resources executions used.
"""

from typing import Any, Literal

from sqlalchemy import Select, func, null, or_, select

from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.provider import Provider
from climate_ref.query.executions import _scope_executions
from climate_ref.query.filters import ResourceFilter

GroupBy = Literal["diagnostic", "provider"]
"""How resource summaries are grouped: one row per diagnostic, or per provider."""


def select_execution_resources(filters: ResourceFilter | None = None) -> Select[Any]:
    """
    Build the ``Select`` over the resource columns on ``Execution``.

    Rows are kept when they carry a measurement or when they failed.
    A failed execution with no measurement is what an out-of-memory kill looks like,
    and that count is needed to report the bias it introduces.

    Ordered by ``created_at, id`` ascending so SQL pagination is deterministic across pages.

    Parameters
    ----------
    filters
        Restricts the executions considered. ``None`` means no restriction.

    Returns
    -------
    :
        A ``Select`` yielding one row per execution.
    """
    filters = filters or ResourceFilter()

    stmt = (
        select(
            Execution.id.label("execution_id"),
            Provider.slug.label("provider_slug"),
            Diagnostic.slug.label("diagnostic_slug"),
            Execution.successful,
            Execution.wall_seconds,
            Execution.cpu_seconds,
            Execution.peak_memory_bytes,
            Execution.memory_source,
            Execution.memory_limit_bytes,
            Execution.cpu_limit,
            Execution.resources_exclusive,
            Execution.queue_seconds,
            Execution.created_at,
        )
        .where(
            or_(
                Execution.wall_seconds.is_not(None),
                Execution.peak_memory_bytes.is_not(None),
                Execution.successful.is_(False),
            )
        )
        .order_by(Execution.created_at, Execution.id)
    )
    return _apply_resource_filters(stmt, filters)


def _apply_resource_filters(stmt: Select[Any], filters: ResourceFilter) -> Select[Any]:
    """Join an ``Execution`` select through to its diagnostic and provider, then apply the filter."""
    stmt = (
        stmt.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
        .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
        .join(Provider, Diagnostic.provider_id == Provider.id)
    )
    return _scope_executions(stmt, filters, latest_only=filters.latest_only)


_SUMMARY_AGGREGATES = (
    func.count(Execution.wall_seconds).label("timed_execution_count"),
    func.sum(Execution.wall_seconds).label("wall_seconds_total"),
    func.avg(Execution.wall_seconds).label("wall_seconds_mean"),
    func.max(Execution.wall_seconds).label("wall_seconds_max"),
    func.sum(Execution.cpu_seconds).label("cpu_seconds_total"),
    func.avg(Execution.cpu_seconds).label("cpu_seconds_mean"),
    func.min(Execution.peak_memory_bytes).label("peak_memory_bytes_min"),
    func.max(Execution.peak_memory_bytes).label("peak_memory_bytes_max"),
)


def select_resource_summary(
    filters: ResourceFilter | None = None, *, group_by: GroupBy | None = None
) -> Select[Any]:
    """
    Build the ``Select`` that aggregates resource usage in SQL.

    Every execution matching ``filters`` is aggregated, including ones without a measurement,
    because the aggregates skip missing values.

    Parameters
    ----------
    filters
        Restricts the executions considered. ``None`` means no restriction.
    group_by
        ``diagnostic`` or ``provider`` for one row per group, ordered by slug.
        ``None`` for a single row over every matching execution.

    Returns
    -------
    :
        A ``Select`` yielding ``provider_slug``, ``diagnostic_slug`` and the aggregate columns.
    """
    provider: Any = Provider.slug if group_by is not None else null()
    diagnostic: Any = Diagnostic.slug if group_by == "diagnostic" else null()
    stmt = select(
        provider.label("provider_slug"), diagnostic.label("diagnostic_slug"), *_SUMMARY_AGGREGATES
    ).select_from(Execution)
    stmt = _apply_resource_filters(stmt, filters or ResourceFilter())
    if group_by is not None:
        keys = [provider, diagnostic] if group_by == "diagnostic" else [provider]
        stmt = stmt.group_by(*keys).order_by(*keys)
    return stmt
