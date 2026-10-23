"""
Builders over diagnostics, with counts of their execution groups and executions.
"""

from typing import Any

from sqlalchemy import Select, case, func, literal, select

from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import MetricValue, MetricValueType
from climate_ref.models.provider import Provider
from climate_ref.query.executions import _join_latest_execution, _scope_executions
from climate_ref.query.filters import DiagnosticFilter
from climate_ref.query.values import _Unindexed

_SUCCESSFUL_COUNT = func.sum(case((Execution.successful.is_(True), 1), else_=0))
_RUNNING_COUNT = func.sum(case((Execution.successful.is_(None) & Execution.id.isnot(None), 1), else_=0))
_FAILED_COUNT = func.sum(case((Execution.successful.is_(False), 1), else_=0))
_NOT_STARTED_COUNT = func.sum(case((Execution.id.is_(None), 1), else_=0))
_DIRTY_COUNT = func.sum(case((ExecutionGroup.dirty.is_(True), 1), else_=0))


def select_group_counts(filters: DiagnosticFilter | None = None) -> Select[Any]:
    """
    Build the ``Select`` for each diagnostic's group counts, without counting executions.

    Selects ``provider_slug``, ``diagnostic_slug``, ``group_count``, ``successful``, ``running``, ``failed``,
    ``not_started`` and ``dirty``, counted as in
    [select_diagnostics][climate_ref.query.diagnostics.select_diagnostics].
    Diagnostics with no groups are kept at zero, unless ``filters.input_datasets`` is set.
    Ordered by ``(Provider.slug, Diagnostic.slug, Diagnostic.id)``.
    """
    filters = filters or DiagnosticFilter()
    count_stmt = select(
        ExecutionGroup.diagnostic_id,
        func.count(ExecutionGroup.id).label("group_count"),
        _SUCCESSFUL_COUNT.label("successful"),
        _RUNNING_COUNT.label("running"),
        _FAILED_COUNT.label("failed"),
        _NOT_STARTED_COUNT.label("not_started"),
        _DIRTY_COUNT.label("dirty"),
    ).select_from(ExecutionGroup)
    counts = _join_latest_execution(count_stmt, filters).group_by(ExecutionGroup.diagnostic_id).subquery()
    stmt = (
        select(
            Provider.slug.label("provider_slug"),
            Diagnostic.slug.label("diagnostic_slug"),
            *(func.coalesce(c, 0).label(c.name) for c in counts.c if c.name != "diagnostic_id"),
        )
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .join(counts, Diagnostic.id == counts.c.diagnostic_id, isouter=not filters.input_datasets)
        .order_by(Provider.slug, Diagnostic.slug, Diagnostic.id)
    )
    return stmt.where(*filters.diagnostic_clauses())


def select_diagnostics(filters: DiagnosticFilter | None = None) -> Select[Any]:
    """
    Build the ``Select`` for diagnostics joined to their provider, with group and execution counts.

    ``group_count`` counts the diagnostic's groups that the filter keeps,
    the same groups as [select_execution_groups][climate_ref.query.executions.select_execution_groups].
    ``successful``, ``failed`` and ``running`` count those whose newest execution succeeded, failed or has not
    finished.
    ``not_started`` counts those with no execution and ``dirty`` those marked to be rerun.
    ``execution_count`` counts every execution of those diagnostics that the filter keeps,
    the same executions as
    [select_execution_resources][climate_ref.query.resources.select_execution_resources].
    With ``filters.input_datasets`` diagnostics with no matching group are dropped.
    Ordered by ``(Provider.slug, Diagnostic.slug, Diagnostic.id)``, so paging is stable.
    """
    filters = filters or DiagnosticFilter()

    execution_count_stmt = (
        select(ExecutionGroup.diagnostic_id, func.count(Execution.id).label("execution_count"))
        .select_from(Execution)
        .join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
    )
    execution_count_subquery = (
        _scope_executions(execution_count_stmt, filters).group_by(ExecutionGroup.diagnostic_id).subquery()
    )
    return (
        select_group_counts(filters)
        .add_columns(
            Diagnostic.name.label("name"),
            Diagnostic.promoted_version.label("promoted_version"),
            func.coalesce(execution_count_subquery.c.execution_count, 0).label("execution_count"),
        )
        .outerjoin(execution_count_subquery, Diagnostic.id == execution_count_subquery.c.diagnostic_id)
    )


def select_diagnostics_with_values(
    value_type: MetricValueType, filters: DiagnosticFilter | None = None
) -> Select[Any]:
    """
    Build the ``Select`` for diagnostics with at least one value of ``value_type``.

    The executions that count are those [Reader.values][climate_ref.results.reader.Reader.values] reads
    for the same filter, so by default retracted executions and superseded versions do not count.

    Selects ``id``, ``provider_slug`` and ``slug``, ordered like
    [select_diagnostics][climate_ref.query.diagnostics.select_diagnostics].
    """
    filters = filters or DiagnosticFilter()

    has_values = (
        select(literal(1))
        .select_from(ExecutionGroup)
        .join(Execution, Execution.execution_group_id == ExecutionGroup.id)
        .join(MetricValue, MetricValue.execution_id == Execution.id)
        .where(
            ExecutionGroup.diagnostic_id == Diagnostic.id,
            # Without table statistics SQLite picks ix_metric_value_type and reads every series row.
            _Unindexed(MetricValue.type.expression) == value_type,
        )
    )
    has_values = _scope_executions(has_values, filters)

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
