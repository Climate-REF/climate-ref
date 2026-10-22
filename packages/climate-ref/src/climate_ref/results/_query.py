"""
Filter and ``Select`` builders for metric-value queries, and the execution scope they share.

This module is the single source of truth for *how* metric values are filtered and joined.
``ExecutionScope`` and ``apply_execution_scope`` also decide which executions resource queries consider.
Both the typed facade ([climate_ref.results.values.Reader][]) and any power-user / API
caller build on the ``select_*`` functions here, so the filter/join semantics live in exactly
one place. This mirrors the existing house style of module-level query functions that take a
``Session`` (see [climate_ref.models.execution.get_execution_group_and_latest_filtered][]).

The functions return a SQLAlchemy ``Select`` and never touch a ``Session`` -- callers add
their own pagination, wrap in ``exists()``, or hand them to the facade to materialise.
"""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import attrs
from sqlalchemy import ColumnElement, Select, and_, cast, false, func, literal, or_, select, true, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.compiler import SQLCompiler
from sqlalchemy.sql.functions import FunctionElement

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.dataset_query import _as_facets, _entity_for
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import execution_datasets
from climate_ref.models.execution import select_latest_executions as select_latest_executions  # noqa: PLC0414
from climate_ref.models.metric_value import (
    MetricValue,
    MetricValueType,
    ScalarMetricValue,
    SeriesMetricValue,
)
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_id_tuple, _as_source_types, _as_str_tuple
from climate_ref_core.source_types import SourceDatasetType


@attrs.frozen(kw_only=True)
class InputDatasetFilter:
    """
    Match executions by the datasets they ran against.

    An execution matches when one of its input datasets has one of ``source_types``
    and satisfies every facet in ``facets`` (values for one facet are OR-combined).
    A source type whose dataset table lacks a requested facet cannot match,
    so a CMIP7-only facet does not pull in all of CMIP6.
    With ``dataset_ids`` the matching dataset must also be one of those ids,
    and ``source_types`` may then be left empty to match a dataset of any type.
    With ``or_no_model_input`` an execution with no model-simulation input dataset also matches,
    which keeps observation-only executions when scoping to one era.

    Raises
    ------
    ValueError
        If neither ``source_types`` nor ``dataset_ids`` is set,
        or a facet is a column on none of the ``source_types``.
    """

    source_types: tuple[SourceDatasetType, ...] = attrs.field(default=(), converter=_as_source_types)
    """Dataset types to match against, e.g. ``(SourceDatasetType.CMIP6, SourceDatasetType.CMIP7)``."""

    dataset_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
    """Dataset primary keys to match. An empty tuple matches nothing."""

    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """Dataset columns to match, e.g. ``{"source_id": ("ACCESS-CM2",)}``. ``None`` matches any dataset."""

    or_no_model_input: bool = False
    """Also match executions with no model-simulation input dataset, such as observation-only ones."""

    def __attrs_post_init__(self) -> None:
        if not self.source_types and self.dataset_ids is None:
            raise ValueError("`source_types` must name at least one dataset type, or `dataset_ids` be set")
        if self.facets and not self.source_types:
            raise ValueError("`facets` need `source_types` to say which dataset table they are on")
        for facet in self.facets or {}:
            if not any(facet in _entity_for(t).__mapper__.columns for t in self.source_types):
                names = ", ".join(t.value for t in self.source_types)
                raise ValueError(f"Unknown facet {facet!r} for {names}")

    def execution_clause(self) -> ColumnElement[bool]:
        """Clause over ``Execution`` that is true when the execution ran against a matching dataset."""
        facets = self.facets or {}
        branches: list[ColumnElement[bool]] = []
        if not self.source_types and self.dataset_ids is not None:
            used = select(execution_datasets.c.execution_id).where(
                execution_datasets.c.dataset_id.in_(self.dataset_ids)
            )
            branches.append(Execution.id.in_(used))
        for source_type in self.source_types:
            entity = _entity_for(source_type)
            if any(facet not in entity.__mapper__.columns for facet in facets):
                continue
            conditions = [getattr(entity, facet).in_(values) for facet, values in facets.items()]
            if self.dataset_ids is not None:
                conditions.append(entity.id.in_(self.dataset_ids))
            branches.append(Execution.datasets.of_type(entity).any(and_(true(), *conditions)))
        if self.or_no_model_input:
            model_types = (t for t in SourceDatasetType if t.is_model_output)
            absent = (~Execution.datasets.of_type(_entity_for(t)).any() for t in model_types)
            branches.append(and_(*absent))
        # An empty or_() emits no predicate, which would match every execution.
        return or_(*branches) if branches else false()

    def execution_group_clause(self) -> ColumnElement[bool]:
        """Clause over ``ExecutionGroup`` that is true when any of its executions matches."""
        return ExecutionGroup.executions.any(self.execution_clause())


# Unslotted so it can be mixed in beside ExecutionScope.
@attrs.frozen(kw_only=True, slots=False)
class SlugFilter:
    """
    Diagnostic and provider slug matching, shared by every reader filter.

    The exact and substring fields can be combined, and every set field must match.
    Apply it with [slug_clauses][climate_ref.results._query.SlugFilter.slug_clauses].
    """

    diagnostic_slug: str | None = None
    """Exact-match diagnostic slug, for API path-scoped queries."""

    provider_slug: str | None = None
    """Exact-match provider slug, for API path-scoped queries."""

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on diagnostic slug (OR-combined), for CLI-style search."""

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on provider slug (OR-combined), for CLI-style search."""

    @property
    def matches_any(self) -> bool:
        """Whether any diagnostic or provider axis is set."""
        return self.diagnostic_slug is not None or bool(self.diagnostic_contains) or self.matches_provider

    @property
    def matches_provider(self) -> bool:
        """Whether any provider axis is set."""
        return self.provider_slug is not None or bool(self.provider_contains)

    def slug_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses for a select joined to ``Diagnostic``, and to ``Provider`` if a provider axis is set."""
        clauses: list[ColumnElement[bool]] = []
        if self.diagnostic_slug is not None:
            clauses.append(Diagnostic.slug == self.diagnostic_slug)
        if self.diagnostic_contains:
            clauses.append(or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in self.diagnostic_contains)))
        if self.provider_slug is not None:
            clauses.append(Provider.slug == self.provider_slug)
        if self.provider_contains:
            clauses.append(or_(*(Provider.slug.ilike(f"%{s.lower()}%") for s in self.provider_contains)))
        return clauses

    def diagnostic_ids(self) -> Select[tuple[int]]:
        """Select the ids of the diagnostics that match."""
        stmt = select(Diagnostic.id)
        if self.matches_provider:
            stmt = stmt.join(Provider, Diagnostic.provider_id == Provider.id)
        return stmt.where(*self.slug_clauses())


@attrs.frozen(kw_only=True)
class ExecutionScope:
    """
    Which executions a filter considers, shared by every filter over execution outputs.

    Apply it with [apply_execution_scope][climate_ref.results._query.apply_execution_scope].
    """

    include_retracted: bool = False
    """Include retracted executions."""

    latest_only: bool = False
    """
    Keep only each group's latest execution, as picked by ``select_latest_executions``.

    The latest is picked before retracted executions are dropped.
    So if a group's newest execution is retracted, that group contributes nothing
    unless ``include_retracted=True``.
    It never falls back to an older execution, because the newer one superseded it.
    ``DatasetFilter`` drops retracted rows first instead, because an older dataset version is still valid.
    """


def apply_execution_scope(
    stmt: Select[Any], scope: ExecutionScope, *, execution_group_ids: Sequence[int] | None = None
) -> Select[Any]:
    """
    Apply an ``ExecutionScope`` to a select already joined to ``Execution``.

    Pass ``execution_group_ids`` when the select is already limited to those groups,
    so ``latest_only`` ranks only their executions.
    """
    if not scope.include_retracted:
        stmt = stmt.where(Execution.retracted.is_(False))
    if scope.latest_only:
        among = [Execution.execution_group_id.in_(list(execution_group_ids))] if execution_group_ids else []
        latest = select_latest_executions(among).subquery()
        stmt = stmt.where(Execution.id.in_(select(latest.c.execution_id)))
    return stmt


@attrs.frozen(kw_only=True)
class MetricValueFilter(ExecutionScope, SlugFilter):
    """
    Declarative filter over metric values.

    ``None`` means "do not constrain on this axis",
    except ``include_retracted`` and ``include_superseded``, which constrain by default.
    An empty id sequence matches nothing, and an empty ``exclude_ids`` excludes nothing.
    The same object drives scalar and series queries, frame conversion and facet collection,
    so a filter is written once and reused across output shapes.

    Diagnostic and provider matching comes from [SlugFilter][climate_ref.results._query.SlugFilter].
    """

    execution_ids: Sequence[int] | None = None
    """Restrict to values produced by these executions."""

    execution_group_ids: Sequence[int] | None = None
    """Restrict to values produced by executions belonging to these groups."""

    dimensions: Mapping[str, str | Sequence[str]] | None = None
    """CV dimension filters keyed by registered dimension name; a string is equality, a sequence is IN."""

    isolate_ids: Sequence[int] | None = None
    """Restrict to exactly these value ids; takes precedence over ``exclude_ids``, matching the API."""

    exclude_ids: Sequence[int] | None = None
    """Exclude these value ids; ignored when ``isolate_ids`` is set."""

    reference_only: bool | None = None
    """Series-only: ``True`` for observation/reference series, ``False`` for model series."""

    include_superseded: bool | None = attrs.field(default=None, eq=bool)
    """
    Include execution groups at superseded diagnostic versions, matching ``ExecutionGroupFilter``.

    ``None`` (unset) or ``False`` keeps only groups at the diagnostic's currently promoted version.
    """

    promoted_only: bool | None = attrs.field(default=None, repr=False, eq=False)
    """Deprecated alias for ``not include_superseded``, kept for one release."""

    input_datasets: InputDatasetFilter | None = None
    """Restrict to values produced by executions that ran against matching datasets."""

    def __attrs_post_init__(self) -> None:
        if self.promoted_only is None:
            return
        # Unset stays None so attrs.evolve can still apply the alias to an existing filter.
        if self.include_superseded is not None and self.include_superseded == self.promoted_only:
            raise ValueError("include_superseded and the deprecated promoted_only disagree")
        warnings.warn(
            "MetricValueFilter(promoted_only=...) is deprecated. Use include_superseded instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        object.__setattr__(self, "include_superseded", not self.promoted_only)
        object.__setattr__(self, "promoted_only", None)

    def dimension_clauses(self, entity: type[MetricValue]) -> list[Any]:
        """
        Build validated SQLAlchemy clauses for the dynamic CV dimension columns.

        Raises
        ------
        KeyError
            If a key is not a registered CV dimension on ``entity``.
        """
        clauses: list[Any] = []
        for key, value in (self.dimensions or {}).items():
            if key not in entity._cv_dimensions:
                raise KeyError(f"Unknown dimension column {key!r}")
            col = getattr(entity, key)
            if isinstance(value, str) or not isinstance(value, Sequence):
                clauses.append(col == value)
            else:
                vals = list(value)
                clauses.append(col == vals[0] if len(vals) == 1 else col.in_(vals))
        return clauses


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
        if not f.include_superseded:
            scope = scope.where(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)
        if f.matches_provider:
            scope = scope.join(Provider, Diagnostic.provider_id == Provider.id)
        scope = scope.where(*f.slug_clauses())

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
