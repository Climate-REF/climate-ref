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

from collections.abc import Mapping, Sequence
from typing import Any

import attrs
from sqlalchemy import ColumnElement, Select, and_, func, or_, select
from sqlalchemy.orm import Session

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.dataset_query import _as_facets, _entity_for
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.metric_value import (
    MetricValue,
    ScalarMetricValue,
    SeriesMetricValue,
)
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_str_tuple
from climate_ref_core.source_types import SourceDatasetType


@attrs.frozen(kw_only=True)
class InputDatasetFilter:
    """
    Match executions by the datasets they ran against.

    An execution matches when one of its input datasets has one of ``source_types``
    and satisfies every facet in ``facets`` (values for one facet are OR-combined).
    A source type whose dataset table lacks a requested facet cannot match,
    so a CMIP7-only facet does not pull in all of CMIP6.

    Raises
    ------
    ValueError
        If ``source_types`` is empty, or a facet is a column on none of the ``source_types``.
    """

    source_types: tuple[SourceDatasetType, ...] = attrs.field(converter=tuple)
    """Dataset types to match against, e.g. ``(SourceDatasetType.CMIP6, SourceDatasetType.CMIP7)``."""

    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """Dataset columns to match, e.g. ``{"source_id": ("ACCESS-CM2",)}``. ``None`` matches any dataset."""

    def __attrs_post_init__(self) -> None:
        if not self.source_types:
            raise ValueError("`source_types` must name at least one dataset type")
        for facet in self.facets or {}:
            if not any(facet in _entity_for(t).__mapper__.columns for t in self.source_types):
                names = ", ".join(t.value for t in self.source_types)
                raise ValueError(f"Unknown facet {facet!r} for {names}")

    def execution_clause(self) -> ColumnElement[bool]:
        """Clause over ``Execution`` that is true when the execution ran against a matching dataset."""
        facets = self.facets or {}
        branches = []
        for source_type in self.source_types:
            entity = _entity_for(source_type)
            if any(facet not in entity.__mapper__.columns for facet in facets):
                continue
            conditions = [getattr(entity, facet).in_(values) for facet, values in facets.items()]
            branches.append(Execution.datasets.of_type(entity).any(and_(*conditions)))
        return or_(*branches)

    def execution_group_clause(self) -> ColumnElement[bool]:
        """Clause over ``ExecutionGroup`` that is true when any of its executions matches."""
        return ExecutionGroup.executions.any(self.execution_clause())


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


def apply_execution_scope(stmt: Select[Any], scope: ExecutionScope) -> Select[Any]:
    """Apply an ``ExecutionScope`` to a select already joined to ``Execution``."""
    if not scope.include_retracted:
        stmt = stmt.where(Execution.retracted.is_(False))
    if scope.latest_only:
        latest = select_latest_executions().subquery()
        stmt = stmt.where(Execution.id.in_(select(latest.c.execution_id)))
    return stmt


@attrs.frozen(kw_only=True)
class MetricValueFilter(ExecutionScope):
    """
    Declarative filter over metric values.

    ``None``/empty means "do not constrain on this axis",
    except ``include_retracted`` and ``promoted_only``, which constrain by default.
    The same object drives scalar and series queries, frame conversion and facet collection,
    so a filter is written once and reused across output shapes.

    Two styles of diagnostic/provider filtering are offered deliberately:

    * ``diagnostic_slug`` / ``provider_slug`` are **exact** matches, for API path-scoped queries
      (``/diagnostics/{provider_slug}/{diagnostic_slug}/values``).
    * ``diagnostic_contains`` / ``provider_contains`` are case-insensitive **substring** matches
      (OR-combined), for CLI-style search.

    Keeping them separate avoids the ``Diagnostic.slug`` vs ``Diagnostic.name`` divergence that
    exists today between the CLI and the ref-app API.
    """

    execution_ids: Sequence[int] | None = None
    """Restrict to values produced by these executions."""

    execution_group_ids: Sequence[int] | None = None
    """Restrict to values produced by executions belonging to these groups."""

    diagnostic_slug: str | None = None
    """Exact-match diagnostic slug, for API path-scoped queries."""

    provider_slug: str | None = None
    """Exact-match provider slug, for API path-scoped queries."""

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on diagnostic slug (OR-combined), for CLI-style search."""

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on provider slug (OR-combined), for CLI-style search."""

    dimensions: Mapping[str, str | Sequence[str]] | None = None
    """CV dimension filters keyed by registered dimension name; a string is equality, a sequence is IN."""

    isolate_ids: Sequence[int] | None = None
    """Restrict to exactly these value ids; takes precedence over ``exclude_ids``, matching the API."""

    exclude_ids: Sequence[int] | None = None
    """Exclude these value ids; ignored when ``isolate_ids`` is set."""

    reference_only: bool | None = None
    """Series-only: ``True`` for observation/reference series, ``False`` for model series."""

    promoted_only: bool = True
    """Restrict to execution groups at the diagnostic's currently promoted version."""

    input_datasets: InputDatasetFilter | None = None
    """Restrict to values produced by executions that ran against matching datasets."""

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


def _apply_common(  # noqa: PLR0912
    stmt: Select[Any], entity: type[MetricValue], f: MetricValueFilter
) -> Select[Any]:
    """
    Apply every filter that is identical for scalar and series to a ``Select``.

    This is the ONLY place provenance, version, retraction, dimension and id filtering is
    expressed, so scalar and series can never diverge.
    """
    needs_diag = (
        f.diagnostic_slug is not None
        or f.provider_slug is not None
        or bool(f.diagnostic_contains or f.provider_contains)
        or f.promoted_only
    )
    needs_exec = bool(
        f.execution_ids
        or f.execution_group_ids
        or f.input_datasets
        or needs_diag
        or not f.include_retracted
        or f.latest_only
    )

    if needs_exec:
        stmt = stmt.join(Execution, entity.execution_id == Execution.id)
        stmt = apply_execution_scope(stmt, f)
    if f.execution_ids:
        stmt = stmt.where(entity.execution_id.in_(list(f.execution_ids)))
    if f.execution_group_ids:
        stmt = stmt.where(Execution.execution_group_id.in_(list(f.execution_group_ids)))
    if f.input_datasets:
        stmt = stmt.where(f.input_datasets.execution_clause())

    if needs_diag:
        stmt = stmt.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
        stmt = stmt.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
    if f.promoted_only:
        stmt = stmt.where(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)
    if f.diagnostic_slug is not None:
        stmt = stmt.where(Diagnostic.slug == f.diagnostic_slug)
    if f.diagnostic_contains:
        stmt = stmt.where(or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in f.diagnostic_contains)))
    if f.provider_slug is not None or f.provider_contains:
        stmt = stmt.join(Provider, Diagnostic.provider_id == Provider.id)
        if f.provider_slug is not None:
            stmt = stmt.where(Provider.slug == f.provider_slug)
        if f.provider_contains:
            stmt = stmt.where(or_(*(Provider.slug.ilike(f"%{s.lower()}%") for s in f.provider_contains)))

    for clause in f.dimension_clauses(entity):
        stmt = stmt.where(clause)

    if f.isolate_ids:
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


def count_values(session: Session, stmt: Select[Any]) -> int:
    """Total row count for a builder's ``Select``, ignoring any offset/limit. For pagination."""
    return session.execute(select(func.count()).select_from(stmt.order_by(None).subquery())).scalar_one()


def select_latest_executions() -> Select[tuple[int, int]]:
    """
    Build a ``Select`` of the latest execution in each execution group.

    "Latest" is the greatest ``created_at``, with the greatest ``id`` breaking a tie,
    the same ordering as [latest_execution_for_group][climate_ref.results._query.latest_execution_for_group].
    Retracted executions are not skipped, so a retracted latest execution still wins its group.

    Returns
    -------
        A ``Select`` with ``execution_group_id`` and ``execution_id`` columns, one row per group
        that has at least one execution. Call ``.subquery()`` to join or filter against it.
    """
    newest = (
        select(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("created_at"),
        )
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
        .group_by(Execution.execution_group_id)
    )


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
