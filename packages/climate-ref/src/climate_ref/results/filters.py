"""
Filter objects for the results readers.

Every reader filter is a [BaseFilter][climate_ref.results.filters.BaseFilter]
composed from one mixin per axis:

* [SlugFilter][climate_ref.results.filters.SlugFilter]: which diagnostics and providers.
* [VersionScope][climate_ref.results.filters.VersionScope]: promoted or every diagnostic version.
* [ExecutionScope][climate_ref.results.filters.ExecutionScope]: which executions count
  (retraction, latest only, creation time).
* [InputDatasetScope][climate_ref.results.filters.InputDatasetScope]: which input datasets
  the counted executions ran against.

A mixin means the same thing, with the same default, on every filter that carries it.
The concrete filters add only the fields specific to their reader.
[DatasetFilter][climate_ref.models.dataset_query.DatasetFilter] filters datasets rather than
executions, so it stands outside this family.
"""

import datetime
from collections.abc import Mapping, Sequence

import attrs
from sqlalchemy import ColumnElement, Select, and_, false, or_, select, true

from climate_ref.models.dataset_query import _as_facets, _entity_for
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import (
    Execution,
    ExecutionFailureReason,
    ExecutionGroup,
    execution_datasets,
    select_latest_executions,
)
from climate_ref.models.metric_value import MetricValue
from climate_ref.models.provider import Provider
from climate_ref.results._converters import (
    _as_failure_reasons,
    _as_id_tuple,
    _as_source_types,
    _as_str_tuple,
)
from climate_ref_core.source_types import SourceDatasetType


def promoted_version_clause() -> ColumnElement[bool]:
    """Clause that keeps execution groups at their diagnostic's promoted version, joined to ``Diagnostic``."""
    return ExecutionGroup.diagnostic_version == Diagnostic.promoted_version


# Unslotted, so several mixins can be combined into one filter.
@attrs.frozen(kw_only=True, slots=False)
class BaseFilter:
    """
    Base of every reader filter.

    The conventions every field follows:

    * ``None`` means "do not constrain on this axis".
    * An empty id tuple matches nothing.
    * Multi-value fields are stored as tuples, and a bare ``str`` is rejected.
    """


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
        """
        Clause over ``ExecutionGroup`` that is true when its latest execution matches.

        A group whose older executions used the datasets but whose latest did not is left out,
        so group-level readers agree with the values that group's latest execution produced.
        """
        latest = select_latest_executions().subquery()
        matching = (
            select(latest.c.execution_group_id)
            .join(Execution, Execution.id == latest.c.execution_id)
            .where(self.execution_clause())
        )
        return ExecutionGroup.id.in_(matching)


@attrs.frozen(kw_only=True, slots=False)
class SlugFilter(BaseFilter):
    """
    Diagnostic and provider slug matching.

    The exact and substring fields can be combined, and every set field must match.
    Apply it with [slug_clauses][climate_ref.results.filters.SlugFilter.slug_clauses].
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


@attrs.frozen(kw_only=True, slots=False)
class VersionScope(BaseFilter):
    """Which diagnostic versions count."""

    include_superseded: bool = False
    """Include execution groups at superseded diagnostic versions, not only the promoted version."""

    def version_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses for a select joined to ``ExecutionGroup`` and ``Diagnostic``."""
        return [] if self.include_superseded else [promoted_version_clause()]


@attrs.frozen(kw_only=True, slots=False)
class ExecutionScope(BaseFilter):
    """
    Which executions count.

    Apply it with [apply_execution_scope][climate_ref.results._query.apply_execution_scope].
    The creation window applies before the latest execution is picked,
    so ``latest_only`` with ``created_before`` keeps each group's latest execution created before then.
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

    created_after: datetime.datetime | None = None
    """Keep only executions created at or after this naive UTC timestamp."""

    created_before: datetime.datetime | None = None
    """Keep only executions created before this naive UTC timestamp."""

    def created_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses over ``Execution`` for the creation window."""
        clauses: list[ColumnElement[bool]] = []
        if self.created_after is not None:
            clauses.append(Execution.created_at >= self.created_after)
        if self.created_before is not None:
            clauses.append(Execution.created_at < self.created_before)
        return clauses


@attrs.frozen(kw_only=True, slots=False)
class InputDatasetScope(BaseFilter):
    """Which input datasets the counted executions ran against."""

    input_datasets: InputDatasetFilter | None = None
    """
    Keep only executions that ran against matching datasets.

    Readers over execution groups check each group's latest execution.
    """


@attrs.frozen(kw_only=True)
class MetricValueFilter(SlugFilter, VersionScope, ExecutionScope, InputDatasetScope):
    """
    Declarative filter over metric values.

    An empty ``exclude_ids`` excludes nothing.
    The same object drives scalar and series queries, frame conversion and facet collection,
    so a filter is written once and reused across output shapes.
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

    def dimension_clauses(self, entity: type[MetricValue]) -> list[ColumnElement[bool]]:
        """
        Build validated SQLAlchemy clauses for the dynamic CV dimension columns.

        Raises
        ------
        KeyError
            If a key is not a registered CV dimension on ``entity``.
        """
        clauses: list[ColumnElement[bool]] = []
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


@attrs.frozen(kw_only=True)
class ResourceFilter(SlugFilter, VersionScope, ExecutionScope, InputDatasetScope):
    """Declarative filter over the executions considered for resource measurements and profiles."""


@attrs.frozen(kw_only=True)
class ExecutionGroupFilter(SlugFilter, VersionScope, InputDatasetScope):
    """
    Declarative filter over execution groups and their latest execution.

    ``selectors`` is matched against each group's ``selectors`` in Python,
    and ``successful`` vs ``latest_successful`` expose the
    [get_execution_group_and_latest_filtered][climate_ref.models.execution.get_execution_group_and_latest_filtered]
    post-rank / pre-rank success filters.
    ``input_datasets`` is checked against the latest execution, as picked under ``latest_successful``.
    """

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

    selectors: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """
    Facet map matched against each group's ``ExecutionGroup.selectors``, in Python.

    A key is a bare facet such as ``source_id``, or ``<dataset_type>.<facet>`` for one dataset type.
    Unlike ``input_datasets``, this checks how the group was selected, not the datasets its executions used.
    """

    failure_reasons: tuple[ExecutionFailureReason, ...] | None = attrs.field(
        default=None, converter=_as_failure_reasons
    )
    """Keep a group only if its latest execution failed for one of these reasons."""


@attrs.frozen(kw_only=True)
class DiagnosticFilter(SlugFilter, InputDatasetScope):
    """
    Declarative filter over diagnostics, and over the execution statistics grouped by diagnostic.

    With ``input_datasets`` only execution groups whose latest execution matches are counted,
    and diagnostics left with no groups are dropped.
    """
