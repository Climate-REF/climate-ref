"""
Query builders behind every reader, for callers that want database rows rather than reader objects.

Each reader method runs one builder here, pages it and turns the rows into plain objects.
Calling the builder yourself gives the same rows,
so you can add your own conditions, columns or loading options before running it.
The builders take the same filter as the matching reader method and never touch a ``Session``.

Builders over one kind of record select the ORM entity:

* [select_scalar_values][climate_ref.results.queries.select_scalar_values] and
  [select_series_values][climate_ref.results.queries.select_series_values]: metric values.
* [select_datasets][climate_ref.results.queries.select_datasets]: datasets.
* [select_execution_groups][climate_ref.results.queries.select_execution_groups]:
  execution groups, each paired with its newest execution.
* [select_execution_outputs][climate_ref.results.queries.select_execution_outputs]: one execution's outputs.
* [select_latest_execution][climate_ref.results.queries.select_latest_execution]: a group's newest execution.

Builders that count or summarise select labelled columns, listed on each builder:

* [select_diagnostics][climate_ref.results.queries.select_diagnostics] and
  [select_diagnostics_with_values][climate_ref.results.queries.select_diagnostics_with_values].
* [select_execution_statistics][climate_ref.results.queries.select_execution_statistics].
* [select_execution_resources][climate_ref.results.queries.select_execution_resources] and
  [select_resource_summary][climate_ref.results.queries.select_resource_summary].

[count][climate_ref.results.queries.count] and [paginate][climate_ref.results.queries.paginate]
give the ``total_count`` and paging that the readers use.

Examples
--------
Scalar values as ORM rows, narrowed further than ``MetricValueFilter`` allows:

```python
from climate_ref.models.metric_value import ScalarMetricValue
from climate_ref.results import MetricValueFilter
from climate_ref.results.queries import count, paginate, select_scalar_values

stmt = select_scalar_values(MetricValueFilter(provider_slug="pmp", latest_only=True))
stmt = stmt.where(ScalarMetricValue.value > 0)
total = count(db.session, stmt)
rows = db.session.execute(paginate(stmt, offset=0, limit=50)).scalars().all()
```
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from sqlalchemy import ColumnElement, Select, case, cast, func, literal, null, or_, select, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.functions import FunctionElement

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.dataset_query import select_datasets as select_datasets  # noqa: PLC0414
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionOutput
from climate_ref.models.execution import select_latest_executions as select_latest_executions  # noqa: PLC0414
from climate_ref.models.metric_value import (
    MetricValue,
    MetricValueType,
    ScalarMetricValue,
    SeriesMetricValue,
)
from climate_ref.models.provider import Provider
from climate_ref.results.filters import (
    DiagnosticFilter,
    ExecutionGroupFilter,
    ExecutionScope,
    MetricValueFilter,
    ResourceFilter,
    promoted_version_clause,
)

GroupBy = Literal["diagnostic", "provider"]
"""How resource summaries are grouped: one row per diagnostic, or per provider."""


def apply_execution_scope(
    stmt: Select[Any], scope: ExecutionScope, *, execution_group_ids: Sequence[int] | None = None
) -> Select[Any]:
    """
    Apply an ``ExecutionScope`` to a select already joined to ``Execution``.

    This adds the creation window, drops retracted executions unless ``include_retracted`` is set,
    and with ``latest_only`` keeps each group's newest execution.
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
    Apply every filter that scalar and series values share to a ``Select``.

    This covers the execution, version, retraction, dimension and id filters.
    Keeping them in one place stops scalar and series queries from drifting apart.

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
    Build a ``Select`` over ``ScalarMetricValue`` for the filter, or for every value if ``f`` is ``None``.

    Rows are ordered by id, so ``offset`` and ``limit`` give stable pages.

    Examples
    --------
    ```python
    stmt = select_scalar_values(MetricValueFilter(dimensions={"statistic": "mean"}))
    rows = db.session.execute(stmt.limit(10)).scalars().all()
    ```
    """
    f = f or MetricValueFilter()
    stmt = _apply_common(select(ScalarMetricValue), ScalarMetricValue, f)
    return stmt.order_by(ScalarMetricValue.id)


def select_series_values(f: MetricValueFilter | None = None) -> Select[tuple[SeriesMetricValue]]:
    """
    Build a ``Select`` over ``SeriesMetricValue`` for the filter, or for every series if ``f`` is ``None``.

    ``reference_only`` on the filter keeps only the reference series, or only the model ones.
    The shared index is loaded with each row, so ``index`` and ``index_name`` are safe to read.
    Rows are ordered by id, so ``offset`` and ``limit`` give stable pages.
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
    Build an unordered ``Select`` of the ids of the ``value_type`` values that match the filter.

    It matches the same rows as [select_scalar_values][climate_ref.results.queries.select_scalar_values]
    or [select_series_values][climate_ref.results.queries.select_series_values],
    and is meant for counts, facets and outlier detection.
    A filter on diagnostics, providers or execution groups is written as a subquery
    that uses ``ix_metric_value_execution_id_type``, so SQLite reads only that scope's values.
    With ``isolate_ids`` it uses joins instead, because looking those ids up directly is cheaper.
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


def count(session: Session, stmt: Select[Any]) -> int:
    """Count the rows ``stmt`` returns, ignoring any ordering, offset and limit. Used for ``total_count``."""
    return session.execute(select(func.count()).select_from(stmt.order_by(None).subquery())).scalar_one()


def paginate(stmt: Select[Any], offset: int, limit: int | None) -> Select[Any]:
    """Apply ``offset``/``limit`` to a ``Select``, leaving it untouched when neither is set."""
    if limit is not None:
        return stmt.offset(offset).limit(limit)
    if offset:
        return stmt.offset(offset)
    return stmt


_NEWEST_FIRST = (Execution.created_at.desc(), Execution.id.desc())


def select_latest_execution(execution_group_id: int) -> Select[tuple[Execution]]:
    """
    Build the ``Select`` for one group's newest ``Execution``, which returns no row if it has none.

    The newest is chosen by ``created_at``, then id, and retracted executions count.
    """
    return (
        select(Execution)
        .where(Execution.execution_group_id == execution_group_id)
        .order_by(*_NEWEST_FIRST)
        .limit(1)
    )


def select_diagnostics(filters: DiagnosticFilter | None = None) -> Select[Any]:
    """
    Build the ``Select`` for diagnostics joined to their provider, with an execution-group count.

    ``execution_group_count`` counts every ``ExecutionGroup`` of the diagnostic, across all versions.
    With ``filters.input_datasets`` it counts only matching groups, and diagnostics with none are dropped.
    Ordered by ``(Provider.slug, Diagnostic.slug, Diagnostic.id)``, so paging is stable.
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
    By default retracted executions do not count, as in
    [Reader.values][climate_ref.results.reader.Reader.values].
    ``execution_group_where`` further restricts which execution groups count.

    Selects ``id``, ``provider_slug`` and ``slug``, ordered like
    [select_diagnostics][climate_ref.results.queries.select_diagnostics].
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


def select_execution_statistics(filters: DiagnosticFilter | None = None) -> Select[Any]:
    """
    Build the ``Select`` that counts execution groups by status for each (provider, diagnostic).

    It takes a [DiagnosticFilter][climate_ref.results.filters.DiagnosticFilter]
    rather than an ``ExecutionGroupFilter``, because the counts cannot honour ``dirty``, ``successful``
    or ``selectors``.
    With ``filters.input_datasets`` only groups whose latest execution matches are counted.

    The status columns are:

    * ``running``: the latest execution has not finished.
    * ``failed``: the latest execution failed.
    * ``successful``: the latest execution succeeded.
    * ``not_started``: the group has no execution.
    * ``dirty``: the group is marked to be rerun.

    Retraction is ignored, so a retracted latest execution still sets its group's status.
    Only groups at the diagnostic's ``promoted_version`` are counted.
    """
    filters = filters or DiagnosticFilter()
    latest = select_latest_executions().subquery()

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
        .where(promoted_version_clause())
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .outerjoin(latest, latest.c.execution_group_id == ExecutionGroup.id)
        .outerjoin(Execution, Execution.id == latest.c.execution_id)
        .group_by(Provider.slug, Diagnostic.slug)
        .order_by(Provider.slug, Diagnostic.slug)
    )

    stmt = stmt.where(*filters.diagnostic_clauses())
    if filters.input_datasets:
        # The outer-joined Execution is each group's latest.
        stmt = stmt.where(filters.input_datasets.execution_clause())

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
    clauses = filters.diagnostic_clauses()
    if clauses:
        # Resolved to group ids first, so SQLite reaches executions by index rather than scanning them.
        scoped_groups = (
            select(ExecutionGroup.id)
            .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
            .join(Provider, Diagnostic.provider_id == Provider.id)
            .where(*clauses)
        )
        stmt = stmt.where(Execution.execution_group_id.in_(scoped_groups))
    stmt = stmt.where(*filters.version_clauses())
    if filters.input_datasets:
        stmt = stmt.where(filters.input_datasets.execution_clause())
    return apply_execution_scope(stmt, filters)


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


def _latest_execution_id(among_executions: Sequence[ColumnElement[bool]] = ()) -> Any:
    """Correlated subquery of the outer group's newest execution id, or ``NULL`` if it has none."""
    return (
        select(Execution.id)
        .where(Execution.execution_group_id == ExecutionGroup.id, *among_executions)
        .order_by(*_NEWEST_FIRST)
        .limit(1)
        .correlate(ExecutionGroup)
        .scalar_subquery()
    )


def _successful_clause(successful: bool) -> ColumnElement[bool]:
    """Match succeeded executions for ``True``, and failed, running or missing ones for ``False``."""
    if successful:
        return Execution.successful.is_(True)
    return or_(Execution.successful.is_(False), Execution.successful.is_(None))


def _filter_execution_groups(stmt: Select[Any], filters: ExecutionGroupFilter) -> Select[Any]:
    """Apply the filters on ``ExecutionGroup``, which never depend on which execution is newest."""
    if not filters.include_superseded:
        stmt = stmt.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
    stmt = stmt.where(*filters.version_clauses())
    # Matched against diagnostics first, so SQLite reaches the groups by index rather than scanning them.
    if filters.matches_any:
        stmt = stmt.where(ExecutionGroup.diagnostic_id.in_(filters.diagnostic_ids()))
    if filters.dirty is True:
        stmt = stmt.where(ExecutionGroup.dirty.is_(True))
    elif filters.dirty is False:
        stmt = stmt.where(or_(ExecutionGroup.dirty.is_(False), ExecutionGroup.dirty.is_(None)))
    return stmt


def _latest_execution_clauses(filters: ExecutionGroupFilter) -> list[ColumnElement[bool]]:
    """Clauses on each group's outer-joined newest ``Execution``."""
    clauses: list[ColumnElement[bool]] = []
    if filters.successful is not None:
        clauses.append(_successful_clause(filters.successful))
    if filters.failure_reasons:
        clauses.append(Execution.failure_reason.in_([reason.value for reason in filters.failure_reasons]))
    if filters.input_datasets:
        clauses.append(filters.input_datasets.execution_clause())
    return clauses


def _join_latest_execution(stmt: Select[Any], filters: ExecutionGroupFilter) -> Select[Any]:
    """Outer-join each group's newest ``Execution`` and apply every filter, rejecting ``selectors``."""
    if filters.selectors:
        raise ValueError(
            "`selectors` cannot be applied in SQL. Use `filter_by_selectors` on the rows instead."
        )
    among = () if filters.latest_successful is None else (_successful_clause(filters.latest_successful),)
    stmt = stmt.outerjoin(Execution, Execution.id == _latest_execution_id(among))
    return _filter_execution_groups(stmt, filters).where(*_latest_execution_clauses(filters))


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
        and pass the rows to [filter_by_selectors][climate_ref.results.queries.filter_by_selectors].
    """
    filters = filters or ExecutionGroupFilter()
    stmt: Select[Any] = select(ExecutionGroup, Execution).options(
        joinedload(ExecutionGroup.diagnostic).joinedload(Diagnostic.provider)
    )
    return _join_latest_execution(stmt, filters).order_by(ExecutionGroup.id)


def select_execution_group_ids(filters: ExecutionGroupFilter | None = None) -> Select[tuple[int]]:
    """
    Build the ``Select`` of matching group ids, for counting.

    It matches the groups of [select_execution_groups][climate_ref.results.queries.select_execution_groups].
    The newest execution is only looked up when a filter on it can drop a group.

    Raises
    ------
    ValueError
        If ``filters.selectors`` is set.
    """
    filters = filters or ExecutionGroupFilter()
    stmt: Select[Any] = select(ExecutionGroup.id)
    if _latest_execution_clauses(filters) or filters.selectors:
        return _join_latest_execution(stmt, filters)
    return _filter_execution_groups(stmt, filters)


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
        if "." in key:
            dataset_type, facet = key.split(".", 1)
            pairs = group_selectors.get(dataset_type, [])
            return any([facet, value] in pairs for value in values)
        return any([key, value] in pairs for pairs in group_selectors.values() for value in values)

    return [
        (group, execution)
        for group, execution in rows
        if all(matches(group.selectors, key, values) for key, values in selectors.items())
    ]
