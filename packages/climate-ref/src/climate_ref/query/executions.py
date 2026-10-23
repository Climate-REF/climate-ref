"""
Builders over executions and execution groups, and the definition of each group's latest execution.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import ColumnElement, Select, func, or_, select
from sqlalchemy.orm import joinedload

from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup, ExecutionOutput
from climate_ref.query.filters import (
    DiagnosticFilter,
    ExecutionGroupFilter,
    MetricValueFilter,
    ResourceFilter,
    split_selector_key,
)

_ScopedFilter = MetricValueFilter | ResourceFilter | ExecutionGroupFilter | DiagnosticFilter


# There are two forms to select for the "latest" execution.
# The grouped max is fast across many groups, the ordered limit-1 lookup is fast for one group or a page.
def _select_latest_execution(*where: ColumnElement[bool]) -> Select[tuple[Execution]]:
    """Select the latest ``Execution`` matching ``where``."""
    return select(Execution).where(*where).order_by(Execution.created_at.desc(), Execution.id.desc()).limit(1)


def select_latest_execution(execution_group_id: int) -> Select[tuple[Execution]]:
    """
    Build the ``Select`` for one group's latest ``Execution``, which returns no row if it has none.

    The latest is chosen by ``created_at``, then id, and retracted executions count.
    """
    return _select_latest_execution(Execution.execution_group_id == execution_group_id)


def latest_execution_id(among_executions: Sequence[ColumnElement[bool]] = ()) -> Any:
    """
    Correlated subquery of the outer ``ExecutionGroup``'s latest execution id, or ``NULL`` if it has none.

    ``among_executions`` works as in
    [select_latest_executions][climate_ref.query.executions.select_latest_executions].
    """
    return (
        _select_latest_execution(Execution.execution_group_id == ExecutionGroup.id, *among_executions)
        .with_only_columns(Execution.id)
        .correlate(ExecutionGroup)
        .scalar_subquery()
    )


def select_latest_executions(
    among_executions: Sequence[ColumnElement[bool]] = (),
) -> Select[tuple[int, int]]:
    """
    Build a ``Select`` of the latest execution in each execution group.

    "Latest" is the greatest ``created_at``, with the greatest ``id`` breaking a tie,
    the same ordering as [select_latest_execution][climate_ref.query.executions.select_latest_execution].
    Retracted executions are not skipped, so a retracted latest execution still wins its group.

    Parameters
    ----------
    among_executions
        Predicates on ``Execution`` applied *before* picking the latest,
        so the winner is the newest execution *among those matching*.
        For example ``[Execution.successful.is_(True)]`` picks each group's latest successful execution.
        Group-level filters (diagnostic, provider, dirty, version) belong on the outer query instead,
        where they narrow which groups are returned without changing which execution wins.

    Returns
    -------
        A ``Select`` with ``execution_group_id`` and ``execution_id`` columns,
        one row per group that has at least one matching execution.
        Call ``.subquery()`` to join or filter against it.
    """
    # Grouped max rather than row_number: with a predicate, SQLite 3.40 rescans a filtered window
    # subquery for every outer row when it is left-joined to execution_group.
    newest = (
        select(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("created_at"),
        )
        .where(*among_executions)
        .group_by(Execution.execution_group_id)
        .subquery()
    )
    return (
        select(
            Execution.execution_group_id.label("execution_group_id"),
            func.max(Execution.id).label("execution_id"),
        )
        .join(
            newest,
            (Execution.execution_group_id == newest.c.execution_group_id)
            & (Execution.created_at == newest.c.created_at),
        )
        .where(*among_executions)
        .group_by(Execution.execution_group_id)
    )


def _where_group_in_scope(stmt: Select[Any], filters: _ScopedFilter) -> Select[Any]:
    """Keep rows whose ``ExecutionGroup`` is at a kept version, of a matching diagnostic and source type."""
    stmt = stmt.where(*filters.version_clauses(), *filters.group_clauses())
    if filters.matches_any:
        # Matched against diagnostics first, so SQLite reaches the groups by index rather than scanning them.
        stmt = stmt.where(ExecutionGroup.diagnostic_id.in_(filters.diagnostic_ids()))
    return stmt


def _scope_executions(stmt: Select[Any], filters: _ScopedFilter, *, latest_only: bool = False) -> Select[Any]:
    """
    Keep the executions the filter's shared fields keep, for a select joined to ``Execution``.

    The select must also be joined to ``ExecutionGroup``.
    Retracted executions are dropped unless ``include_retracted`` is set,
    and with ``latest_only`` only each group's newest kept execution is left.
    """
    stmt = _where_group_in_scope(stmt, filters)
    if filters.input_datasets:
        stmt = stmt.where(filters.input_datasets.execution_clause(filters.source_types))
    among = filters.execution_clauses()
    stmt = stmt.where(*among)
    if not filters.include_retracted:
        stmt = stmt.where(Execution.retracted.is_(False))
    if latest_only:
        latest = select_latest_executions(among).subquery()
        stmt = stmt.where(Execution.id.in_(select(latest.c.execution_id)))
    return stmt


def select_execution_outputs(execution_id: int) -> Select[Any]:
    """
    Build the ``Select`` for one execution's registered ``ExecutionOutput`` rows.

    Ordered by ``(output_type, id)`` so outputs of one type are together and the order is stable.
    """
    return (
        select(ExecutionOutput)
        .where(ExecutionOutput.execution_id == execution_id)
        .order_by(ExecutionOutput.output_type, ExecutionOutput.id)
    )


def _successful_clause(successful: bool) -> ColumnElement[bool]:
    """Match succeeded executions for ``True``, and failed, running or missing ones for ``False``."""
    if successful:
        return Execution.successful.is_(True)
    return or_(Execution.successful.is_(False), Execution.successful.is_(None))


def _join_latest_execution(
    stmt: Select[Any],
    filters: ExecutionGroupFilter | DiagnosticFilter,
    among_executions: Sequence[ColumnElement[bool]] = (),
) -> Select[Any]:
    """
    Outer-join each group's newest ``Execution`` and keep the groups the filter's shared fields keep.

    This is the one definition of which groups count, used by every reader over groups or diagnostics.
    """
    window = filters.execution_clauses()
    if among_executions and not filters.include_retracted:
        among_executions = [*among_executions, Execution.retracted.is_(False)]
    stmt = stmt.outerjoin(Execution, Execution.id == latest_execution_id([*window, *among_executions]))
    stmt = _where_group_in_scope(stmt, filters)
    if filters.execution_group_ids is not None:
        stmt = stmt.where(ExecutionGroup.id.in_(filters.execution_group_ids))
    if filters.narrows_executions or among_executions:
        stmt = stmt.where(Execution.id.is_not(None))
    if not filters.include_retracted:
        stmt = stmt.where(or_(Execution.id.is_(None), Execution.retracted.is_(False)))
    if filters.input_datasets:
        stmt = stmt.where(filters.input_datasets.execution_clause(filters.source_types))
    return stmt


def _select_groups(stmt: Select[Any], filters: ExecutionGroupFilter) -> Select[Any]:
    """Apply an ``ExecutionGroupFilter`` to a select over ``ExecutionGroup``, rejecting ``selectors``."""
    if filters.selectors:
        raise ValueError(
            "`selectors` cannot be applied in SQL. Use `filter_by_selectors` on the rows instead."
        )
    among = () if filters.latest_successful is None else (_successful_clause(filters.latest_successful),)
    stmt = _join_latest_execution(stmt, filters, among)
    if filters.dirty is True:
        stmt = stmt.where(ExecutionGroup.dirty.is_(True))
    elif filters.dirty is False:
        stmt = stmt.where(or_(ExecutionGroup.dirty.is_(False), ExecutionGroup.dirty.is_(None)))
    if filters.successful is not None:
        stmt = stmt.where(_successful_clause(filters.successful))
    if filters.failure_reasons is not None:
        stmt = stmt.where(Execution.failure_reason.in_([reason.value for reason in filters.failure_reasons]))
    return stmt


def select_execution_groups(
    filters: ExecutionGroupFilter | None = None,
) -> Select[tuple[ExecutionGroup, Execution | None]]:
    """
    Build the ``Select`` for execution groups paired with their newest ``Execution``, ordered by group id.

    The execution is ``None`` for a group that has never run.
    Each group's diagnostic and provider are loaded with it.
    ``latest_successful`` changes which execution counts as newest,
    and the other filters on the execution apply to that one.

    Raises
    ------
    ValueError
        If ``filters.selectors`` is set.
        Selectors are matched in Python, so run this without them
        and pass the rows to [filter_by_selectors][climate_ref.query.executions.filter_by_selectors].
    """
    filters = filters or ExecutionGroupFilter()
    stmt: Select[Any] = select(ExecutionGroup, Execution).options(
        joinedload(ExecutionGroup.diagnostic).joinedload(Diagnostic.provider)
    )
    return _select_groups(stmt, filters).order_by(ExecutionGroup.id)


def select_execution_group_ids(filters: ExecutionGroupFilter | None = None) -> Select[tuple[int]]:
    """
    Build the ``Select`` of matching group ids, for counting.

    It matches the groups of [select_execution_groups][climate_ref.query.executions.select_execution_groups].

    Raises
    ------
    ValueError
        If ``filters.selectors`` is set.
    """
    return _select_groups(select(ExecutionGroup.id), filters or ExecutionGroupFilter())


def filter_by_selectors(
    rows: Sequence[tuple[ExecutionGroup, Execution | None]], selectors: Mapping[str, Sequence[str]]
) -> list[tuple[ExecutionGroup, Execution | None]]:
    """
    Keep the ``(group, execution)`` rows whose group ``selectors`` match.

    This is how ``ExecutionGroupFilter.selectors`` is applied.

    A key is a bare facet such as ``source_id``, matched in any dataset type,
    or ``<dataset_type>.<facet>`` for one dataset type.
    Every key must match, and the values for one key are alternatives.
    """

    def matches(group_selectors: Mapping[str, Any], key: str, values: Sequence[str]) -> bool:
        dataset_type, facet = split_selector_key(key)
        if dataset_type is not None:
            pairs = group_selectors.get(dataset_type, [])
            return any([facet, value] in pairs for value in values)
        return any([facet, value] in pairs for pairs in group_selectors.values() for value in values)

    return [
        (group, execution)
        for group, execution in rows
        if all(matches(group.selectors, key, values) for key, values in selectors.items())
    ]
