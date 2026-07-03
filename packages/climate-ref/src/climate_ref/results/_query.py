"""
Filter and ``Select`` builders for metric-value queries.

This module is the single source of truth for *how* metric values are filtered and joined.
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
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.metric_value import (
    MetricValue,
    ScalarMetricValue,
    SeriesMetricValue,
)
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_str_tuple


@attrs.frozen(kw_only=True)
class MetricValueFilter:
    """
    Declarative filter over metric values.

    Every field is optional; ``None``/empty means "do not constrain on this axis". The same
    object drives scalar and series queries, frame conversion and facet collection, so a filter
    is written once and reused across output shapes.

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

    diagnostic_contains: Sequence[str] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on diagnostic slug (OR-combined), for CLI-style search."""

    provider_contains: Sequence[str] | None = attrs.field(default=None, converter=_as_str_tuple)
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

    include_retracted: bool = False
    """Include values produced by retracted executions."""

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
    needs_diag = bool(
        f.diagnostic_slug or f.provider_slug or f.diagnostic_contains or f.provider_contains
    ) or (f.promoted_only)
    needs_exec = bool(f.execution_ids or f.execution_group_ids or needs_diag or not f.include_retracted)

    if needs_exec:
        stmt = stmt.join(Execution, entity.execution_id == Execution.id)
    if not f.include_retracted:
        stmt = stmt.where(Execution.retracted.is_(False))
    if f.execution_ids:
        stmt = stmt.where(entity.execution_id.in_(list(f.execution_ids)))
    if f.execution_group_ids:
        stmt = stmt.where(Execution.execution_group_id.in_(list(f.execution_group_ids)))

    if needs_diag:
        stmt = stmt.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
        stmt = stmt.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
    if f.promoted_only:
        stmt = stmt.where(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)
    if f.diagnostic_slug:
        stmt = stmt.where(Diagnostic.slug == f.diagnostic_slug)
    if f.diagnostic_contains:
        stmt = stmt.where(or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in f.diagnostic_contains)))
    if f.provider_slug or f.provider_contains:
        stmt = stmt.join(Provider, Diagnostic.provider_id == Provider.id)
        if f.provider_slug:
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
    """Build a ``Select`` over ``ScalarMetricValue`` for the given filter. No session required."""
    f = f or MetricValueFilter()
    return _apply_common(select(ScalarMetricValue), ScalarMetricValue, f)


def select_series_values(f: MetricValueFilter | None = None) -> Select[tuple[SeriesMetricValue]]:
    """
    Build a ``Select`` over ``SeriesMetricValue``.

    The shared index axis is eager-loaded via the model relationship (``lazy="joined"``), so
    ``.index`` / ``.index_name`` are safe to read for the returned rows.
    """
    f = f or MetricValueFilter()
    stmt = _apply_common(select(SeriesMetricValue), SeriesMetricValue, f)
    if f.reference_only is True:
        stmt = stmt.where(SeriesMetricValue.reference_id.is_not(None))
    elif f.reference_only is False:
        stmt = stmt.where(SeriesMetricValue.reference_id.is_(None))
    return stmt


def count_values(session: Session, stmt: Select[Any]) -> int:
    """Total row count for a builder's ``Select``, ignoring any offset/limit. For pagination."""
    return session.execute(select(func.count()).select_from(stmt.order_by(None).subquery())).scalar_one()


def latest_execution_for_group(session: Session, execution_group_id: int) -> Execution | None:
    """
    Return the most recent [Execution][climate_ref.models.execution.Execution] for a group.

    This is the read-side primitive behind the API's ``/executions/{group_id}/values`` behaviour
    of defaulting to the latest execution when no ``execution_id`` is supplied. Centralising it
    here means consumers stop re-deriving the "latest execution" lookup.
    """
    return session.execute(
        select(Execution)
        .where(Execution.execution_group_id == execution_group_id)
        .order_by(Execution.created_at.desc(), Execution.id.desc())
        .limit(1)
    ).scalar_one_or_none()
