"""
``Select`` builders for metric-value queries, and the plumbing the readers share.

This module is the single source of truth for *how* metric values are filtered and joined.
The filter objects themselves live in [climate_ref.results.filters][].
``apply_execution_scope`` applies an ``ExecutionScope`` for every reader over executions.
Both the typed facade ([climate_ref.results.values.Reader][]) and any power-user / API
caller build on the ``select_*`` functions here, so the filter/join semantics live in exactly
one place. This mirrors the existing house style of module-level query functions that take a
``Session`` (see [climate_ref.models.execution.get_execution_group_and_latest_filtered][]).

The functions return a SQLAlchemy ``Select`` and never touch a ``Session`` -- callers add
their own pagination, wrap in ``exists()``, or hand them to the facade to materialise.
"""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, Select, cast, func, literal, select, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.functions import FunctionElement

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import select_latest_executions as select_latest_executions  # noqa: PLC0414
from climate_ref.models.metric_value import (
    MetricValue,
    MetricValueType,
    ScalarMetricValue,
    SeriesMetricValue,
)
from climate_ref.models.provider import Provider
from climate_ref.results.filters import ExecutionScope, MetricValueFilter


def apply_execution_scope(
    stmt: Select[Any], scope: ExecutionScope, *, execution_group_ids: Sequence[int] | None = None
) -> Select[Any]:
    """
    Apply an ``ExecutionScope`` to a select already joined to ``Execution``.

    Pass ``execution_group_ids`` when the select is already limited to those groups,
    so ``latest_only`` ranks only their executions.
    """
    window = scope.created_clauses()
    stmt = stmt.where(*window)
    if not scope.include_retracted:
        stmt = stmt.where(Execution.retracted.is_(False))
    if scope.latest_only:
        among: list[ColumnElement[bool]] = [*window]
        if execution_group_ids:
            among.append(Execution.execution_group_id.in_(list(execution_group_ids)))
        latest = select_latest_executions(among).subquery()
        stmt = stmt.where(Execution.id.in_(select(latest.c.execution_id)))
    return stmt


def _apply_common(
    stmt: Select[Any],
    entity: type[MetricValue],
    f: MetricValueFilter,
    *,
    subquery_type: MetricValueType | None = None,
) -> Select[Any]:
    """
    Apply every filter that is identical for scalar and series to a ``Select``.

    This is the ONLY place provenance, version, retraction, dimension and id filtering is
    expressed, so scalar and series can never diverge.

    With ``subquery_type`` the execution, group, diagnostic and provider filters and the value type
    become one ``(execution_id, type) IN (...)`` subquery instead of joins, which matches the same rows.
    """
    needs_diag = f.matches_any or not f.include_superseded
    needs_exec = subquery_type is not None or bool(
        f.execution_ids is not None
        or f.execution_group_ids is not None
        or f.input_datasets
        or needs_diag
        or not f.include_retracted
        or f.latest_only
        or f.created_after is not None
        or f.created_before is not None
    )

    if needs_exec:
        scope: Select[Any] = (
            select(Execution.id, cast(literal(subquery_type, entity.type.type), entity.type.type))
            if subquery_type is not None
            else stmt.join(Execution, entity.execution_id == Execution.id)
        )
        scope = apply_execution_scope(scope, f, execution_group_ids=f.execution_group_ids)
        if f.execution_group_ids is not None:
            scope = scope.where(Execution.execution_group_id.in_(list(f.execution_group_ids)))
        if f.input_datasets:
            scope = scope.where(f.input_datasets.execution_clause())

        if needs_diag:
            scope = scope.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
            scope = scope.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
        scope = scope.where(*f.version_clauses())
        if f.matches_provider:
            scope = scope.join(Provider, Diagnostic.provider_id == Provider.id)
        scope = scope.where(*f.diagnostic_clauses())

        if subquery_type is not None:
            stmt = stmt.where(tuple_(entity.execution_id, entity.type).in_(scope))
        else:
            stmt = scope
    if f.execution_ids is not None:
        stmt = stmt.where(entity.execution_id.in_(list(f.execution_ids)))

    for clause in f.dimension_clauses(entity):
        stmt = stmt.where(clause)

    if f.isolate_ids is not None:
        stmt = stmt.where(entity.id.in_(list(f.isolate_ids)))
    elif f.exclude_ids:
        stmt = stmt.where(~entity.id.in_(list(f.exclude_ids)))

    return stmt


def select_scalar_values(f: MetricValueFilter | None = None) -> Select[tuple[ScalarMetricValue]]:
    """
    Build a ``Select`` over ``ScalarMetricValue`` for the given filter.

    Ordered by the value id ascending so SQL pagination is deterministic across pages.
    """
    f = f or MetricValueFilter()
    stmt = _apply_common(select(ScalarMetricValue), ScalarMetricValue, f)
    return stmt.order_by(ScalarMetricValue.id)


def select_series_values(f: MetricValueFilter | None = None) -> Select[tuple[SeriesMetricValue]]:
    """
    Build a ``Select`` over ``SeriesMetricValue``.

    The shared index axis is eager-loaded via the model relationship (``lazy="joined"``), so
    ``.index`` / ``.index_name`` are safe to read for the returned rows.
    Ordered by the value id ascending so SQL pagination is deterministic across pages.
    """
    f = f or MetricValueFilter()
    stmt = _apply_common(select(SeriesMetricValue), SeriesMetricValue, f)
    if f.reference_only is True:
        stmt = stmt.where(SeriesMetricValue.reference_id.is_not(None))
    elif f.reference_only is False:
        stmt = stmt.where(SeriesMetricValue.reference_id.is_(None))
    return stmt.order_by(SeriesMetricValue.id)


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


def select_value_ids(value_type: MetricValueType, f: MetricValueFilter | None = None) -> Select[tuple[int]]:
    """
    Build an unordered ``Select`` of the ids of the values of ``value_type`` matching the filter.

    It matches the same rows as [select_scalar_values][climate_ref.results._query.select_scalar_values]
    or [select_series_values][climate_ref.results._query.select_series_values],
    and is meant for counts, facets and outlier detection.
    A filter scoped to diagnostics, providers or groups becomes a subquery on
    ``ix_metric_value_execution_id_type``, so SQLite reads only that scope's values
    rather than every value of the type.
    ``isolate_ids`` keep the joins, because looking those ids up directly is cheaper.
    """
    f = f or MetricValueFilter()
    scoped = not f.isolate_ids and bool(f.matches_any or f.execution_group_ids)
    if scoped:
        stmt = _apply_common(select(MetricValue.id), MetricValue, f, subquery_type=value_type)
    else:
        stmt = _apply_common(select(MetricValue.id), MetricValue, f).where(MetricValue.type == value_type)
    if value_type is MetricValueType.SERIES:
        reference_id: Any = MetricValue.__table__.c.reference_id
        if scoped:
            reference_id = _Unindexed(reference_id)
        if f.reference_only is True:
            stmt = stmt.where(reference_id.is_not(None))
        elif f.reference_only is False:
            stmt = stmt.where(reference_id.is_(None))
    return stmt


def count_values(session: Session, stmt: Select[Any]) -> int:
    """Total row count for a builder's ``Select``, ignoring any offset/limit. For pagination."""
    return session.execute(select(func.count()).select_from(stmt.order_by(None).subquery())).scalar_one()


def paginate(stmt: Select[Any], offset: int, limit: int | None) -> Select[Any]:
    """Apply ``offset``/``limit`` to a ``Select``, leaving it untouched when neither is set."""
    if limit is not None:
        return stmt.offset(offset).limit(limit)
    if offset:
        return stmt.offset(offset)
    return stmt


def latest_execution_for_group(session: Session, execution_group_id: int) -> Execution | None:
    """
    Return the most recent [Execution][climate_ref.models.execution.Execution] for a group.

    This is the read-side primitive behind the API's ``/executions/{group_id}/values`` behaviour
    of defaulting to the latest execution when no ``execution_id`` is supplied.
    """
    return session.execute(
        select(Execution)
        .where(Execution.execution_group_id == execution_group_id)
        .order_by(Execution.created_at.desc(), Execution.id.desc())
        .limit(1)
    ).scalar_one_or_none()
