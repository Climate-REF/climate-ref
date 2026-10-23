"""
Builders over scalar and series metric values.
"""

from typing import Any

from sqlalchemy import ColumnElement, Select, cast, literal, select, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.functions import FunctionElement

from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.metric_value import (
    MetricValue,
    MetricValueType,
    ScalarMetricValue,
    SeriesMetricValue,
)
from climate_ref.query.executions import _scope_executions
from climate_ref.query.filters import MetricValueFilter


def _apply_common(
    stmt: Select[Any],
    entity: type[MetricValue],
    filters: MetricValueFilter,
    *,
    subquery_type: MetricValueType | None = None,
) -> Select[Any]:
    """
    Apply every filter that scalar and series values share to a ``Select``.

    This covers the execution, version, retraction, dimension and id filters.
    Keeping them in one place stops scalar and series queries from drifting apart.

    With ``subquery_type`` the execution, group, diagnostic and provider filters and the value type
    become one ``(execution_id, type) IN (...)`` subquery instead of joins, which matches the same rows.
    """
    scope: Select[Any] = (
        select(Execution.id, cast(literal(subquery_type, entity.type.type), entity.type.type))
        if subquery_type is not None
        else stmt.join(Execution, entity.execution_id == Execution.id)
    )
    scope = scope.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
    scope = _scope_executions(scope, filters, latest_only=filters.latest_only)
    if subquery_type is not None:
        stmt = stmt.where(tuple_(entity.execution_id, entity.type).in_(scope))
    else:
        stmt = scope

    for clause in filters.dimension_clauses(entity):
        stmt = stmt.where(clause)

    if filters.isolate_ids is not None:
        stmt = stmt.where(entity.id.in_(list(filters.isolate_ids)))
    elif filters.exclude_ids:
        stmt = stmt.where(~entity.id.in_(list(filters.exclude_ids)))

    return stmt


def select_scalar_values(filters: MetricValueFilter | None = None) -> Select[tuple[ScalarMetricValue]]:
    """
    Build a ``Select`` over ``ScalarMetricValue`` for the filter, or for every value if there is none.

    Rows are ordered by id, so ``offset`` and ``limit`` give stable pages.

    Examples
    --------
    ```python
    stmt = select_scalar_values(MetricValueFilter(dimensions={"statistic": "mean"}))
    rows = db.session.execute(stmt.limit(10)).scalars().all()
    ```
    """
    filters = filters or MetricValueFilter()
    stmt = _apply_common(select(ScalarMetricValue), ScalarMetricValue, filters)
    return stmt.order_by(ScalarMetricValue.id)


def select_series_values(filters: MetricValueFilter | None = None) -> Select[tuple[SeriesMetricValue]]:
    """
    Build a ``Select`` over ``SeriesMetricValue`` for the filter, or for every series if there is none.

    ``reference_only`` on the filter keeps only the reference series, or only the model ones.
    The shared index is loaded with each row, so ``index`` and ``index_name`` are safe to read.
    Rows are ordered by id, so ``offset`` and ``limit`` give stable pages.
    """
    filters = filters or MetricValueFilter()
    stmt = _apply_common(select(SeriesMetricValue), SeriesMetricValue, filters)
    stmt = stmt.where(*_reference_clauses(SeriesMetricValue.reference_id, filters.reference_only))
    return stmt.order_by(SeriesMetricValue.id)


def _reference_clauses(reference_id: Any, reference_only: bool | None) -> list[ColumnElement[bool]]:
    """Keep the reference series for ``True`` and the model series for ``False``."""
    if reference_only is None:
        return []
    return [reference_id.is_not(None) if reference_only else reference_id.is_(None)]


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


def select_value_ids(
    value_type: MetricValueType, filters: MetricValueFilter | None = None
) -> Select[tuple[int]]:
    """
    Build an unordered ``Select`` of the ids of the ``value_type`` values that match the filter.

    It matches the same rows as [select_scalar_values][climate_ref.query.values.select_scalar_values]
    or [select_series_values][climate_ref.query.values.select_series_values],
    and is meant for counts, facets and outlier detection.
    A filter on diagnostics, providers or execution groups is written as a subquery
    that uses ``ix_metric_value_execution_id_type``, so SQLite reads only that scope's values.
    With ``isolate_ids`` it uses joins instead, because looking those ids up directly is cheaper.
    """
    filters = filters or MetricValueFilter()
    scoped = not filters.isolate_ids and bool(filters.matches_any or filters.execution_group_ids)
    if scoped:
        stmt = _apply_common(select(MetricValue.id), MetricValue, filters, subquery_type=value_type)
    else:
        stmt = _apply_common(select(MetricValue.id), MetricValue, filters)
        stmt = stmt.where(MetricValue.type == value_type)
    if value_type is MetricValueType.SERIES:
        reference_id: Any = MetricValue.__table__.c.reference_id
        if scoped:
            reference_id = _Unindexed(reference_id)
        stmt = stmt.where(*_reference_clauses(reference_id, filters.reference_only))
    return stmt
