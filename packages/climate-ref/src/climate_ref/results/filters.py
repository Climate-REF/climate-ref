"""
Filters that narrow what the results readers return.

Build a filter and pass it to a reader method.
Every field is optional, and a field left as ``None`` does not narrow the result.
So ``MetricValueFilter()`` matches every value at the promoted diagnostic version.

* [MetricValueFilter][climate_ref.results.filters.MetricValueFilter]:
  ``reader.values.scalar_values`` and ``series_values``.
* [ExecutionGroupFilter][climate_ref.results.filters.ExecutionGroupFilter]:
  ``reader.executions.groups`` and ``group_rows``.
* [DiagnosticFilter][climate_ref.results.filters.DiagnosticFilter]:
  ``reader.diagnostics.list`` and ``reader.executions.statistics``.
* [ResourceFilter][climate_ref.results.filters.ResourceFilter]:
  ``reader.resources.measurements``, ``summary``, ``summaries`` and ``profiles``.

Some fields are shared by several filters, and each means the same thing, with the same default, on each.
They come in four groups, one class each.

* [DiagnosticScope][climate_ref.results.filters.DiagnosticScope]:
  ``diagnostic_slug``, ``provider_slug``, ``diagnostic_contains``, ``provider_contains``.
  On every filter.
* [VersionScope][climate_ref.results.filters.VersionScope]: ``include_superseded``, ``diagnostic_version``.
  On every filter.
* [ExecutionScope][climate_ref.results.filters.ExecutionScope]:
  ``include_retracted``, ``created_after``, ``created_before``.
  On every filter.
* [InputDatasetScope][climate_ref.results.filters.InputDatasetScope]: ``input_datasets``.
  On every filter.

Each filter adds its own fields on top, listed on its class.
``MetricValueFilter`` and ``ResourceFilter`` add ``latest_only``.
The group and diagnostic filters always judge a group by its newest execution, so they have no such field.
Dataset-level fields are in [InputDatasetFilter][climate_ref.results.filters.InputDatasetFilter].
[DatasetFilter][climate_ref.models.dataset_query.DatasetFilter] filters datasets rather than executions,
so it is not part of this family.

Examples
--------
The latest PMP values from runs created this year:

>>> import datetime
>>> from climate_ref.results import MetricValueFilter
>>> filters = MetricValueFilter(
...     provider_slug="pmp",
...     latest_only=True,
...     created_after=datetime.datetime(2026, 1, 1),
... )

Then read through a [Reader][climate_ref.results.reader.Reader]:

```python
with Database.from_config(config, read_only=True) as db:
    df = Reader(db).values.scalar_values(filters).to_pandas()
```

Multi-value fields take a list or tuple.
A bare string is rejected, because it would otherwise be matched one character at a time:

>>> MetricValueFilter(diagnostic_contains="enso")
Traceback (most recent call last):
...
TypeError: Expected a sequence of strings, not a bare str. Wrap it in a list/tuple.
>>> MetricValueFilter(diagnostic_contains=["enso"]).diagnostic_contains
('enso',)
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
    """Clause over ``ExecutionGroup`` that keeps groups at their diagnostic's promoted version."""
    promoted = (
        select(Diagnostic.promoted_version)
        .where(Diagnostic.id == ExecutionGroup.diagnostic_id)
        .correlate_except(Diagnostic)
        .scalar_subquery()
    )
    return ExecutionGroup.diagnostic_version == promoted


# Unslotted, so several mixins can be combined into one filter.
@attrs.frozen(kw_only=True, slots=False)
class BaseFilter:
    """
    Base of every reader filter.

    Every field follows the same rules:

    * ``None`` does not narrow the result.
    * An empty id list matches nothing.
    * Multi-value fields are stored as tuples.
    """


@attrs.frozen(kw_only=True)
class InputDatasetFilter:
    """
    Match executions by the datasets they ran against.

    Pass it as ``input_datasets`` on any reader filter.
    An execution matches when one of its input datasets passes every condition:

    * Its type is one of ``source_types``.
    * It has one of the listed values for every facet in ``facets``.
    * Its id is one of ``dataset_ids``, if set.

    A dataset type without a requested facet never matches,
    so a CMIP7-only facet does not pull in all of CMIP6.

    Raises
    ------
    ValueError
        If neither ``source_types`` nor ``dataset_ids`` is set,
        or a facet is a column on none of the ``source_types``.

    Examples
    --------
    Runs that used ACCESS-CM2 data, from either CMIP6 or CMIP7:

    >>> from climate_ref.results import InputDatasetFilter, MetricValueFilter
    >>> from climate_ref_core.source_types import SourceDatasetType
    >>> access = InputDatasetFilter(
    ...     source_types=[SourceDatasetType.CMIP6, SourceDatasetType.CMIP7],
    ...     facets={"source_id": "ACCESS-CM2"},
    ... )
    >>> filters = MetricValueFilter(input_datasets=access)

    A single facet value can be a bare string:

    >>> access.facets
    {'source_id': ('ACCESS-CM2',)}

    Runs that used particular datasets, of any type:

    >>> InputDatasetFilter(dataset_ids=[12, 13]).dataset_ids
    (12, 13)

    CMIP6 runs plus the observation-only runs, which use no model data:

    >>> cmip6 = InputDatasetFilter(source_types=[SourceDatasetType.CMIP6], or_no_model_input=True)

    A facet that is not a column on any of the types is rejected:

    >>> InputDatasetFilter(source_types=[SourceDatasetType.CMIP6], facets={"model": ["x"]})
    Traceback (most recent call last):
    ...
    ValueError: Unknown facet 'model' for cmip6
    """

    source_types: tuple[SourceDatasetType, ...] = attrs.field(default=(), converter=_as_source_types)
    """
    Dataset types to match against, e.g. ``[SourceDatasetType.CMIP6, SourceDatasetType.CMIP7]``.

    Leave it empty with ``dataset_ids`` to match a dataset of any type.
    """

    dataset_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
    """Dataset ids to match. An empty list matches nothing."""

    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """
    Dataset attributes to match, e.g. ``{"source_id": ["ACCESS-CM2", "CanESM5"]}``.

    Values for one facet are alternatives. Every facet must match.
    """

    or_no_model_input: bool = False
    """Also match executions with no model input dataset, such as observation-only ones."""

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
        if self.dataset_ids == ():
            return false()
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
        if not branches:
            return false()
        # An outer-joined missing execution would pass the no-model-input branch.
        return and_(Execution.id.is_not(None), or_(*branches))

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
class DiagnosticScope(BaseFilter):
    """
    A group of fields shared by every filter: which diagnostics or providers to keep.

    Use the ``*_slug`` fields for one exact diagnostic or provider,
    and the ``*_contains`` fields to search by part of a name.
    Every field you set must match.

    Examples
    --------
    One exact diagnostic, then every diagnostic whose name mentions ENSO or the annual cycle:

    >>> from climate_ref.results import DiagnosticFilter
    >>> one = DiagnosticFilter(provider_slug="pmp", diagnostic_slug="enso_tel")
    >>> search = DiagnosticFilter(diagnostic_contains=["enso", "annual-cycle"])
    """

    diagnostic_slug: str | None = None
    """Keep only the diagnostic with exactly this slug, e.g. ``"enso_tel"``."""

    provider_slug: str | None = None
    """Keep only the diagnostics of the provider with exactly this slug, e.g. ``"pmp"``."""

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Keep diagnostics whose slug contains any of these, ignoring case, e.g. ``["enso"]``."""

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Keep the diagnostics of providers whose slug contains any of these, ignoring case."""

    @property
    def matches_any(self) -> bool:
        """Whether any diagnostic or provider axis is set."""
        return self.diagnostic_slug is not None or bool(self.diagnostic_contains) or self.matches_provider

    @property
    def matches_provider(self) -> bool:
        """Whether any provider axis is set."""
        return self.provider_slug is not None or bool(self.provider_contains)

    def diagnostic_clauses(self) -> list[ColumnElement[bool]]:
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
        return stmt.where(*self.diagnostic_clauses())


@attrs.frozen(kw_only=True, slots=False)
class VersionScope(BaseFilter):
    """
    A group of fields shared by most filters: which diagnostic versions to keep.

    By default only results from each diagnostic's promoted (current) version are kept.

    Examples
    --------
    Resource use across every diagnostic version, for sizing jobs:

    >>> from climate_ref.results import ResourceFilter
    >>> filters = ResourceFilter(include_superseded=True)

    Results from version 1 only, after a newer version was promoted:

    >>> filters = ResourceFilter(diagnostic_version=1)
    """

    include_superseded: bool = False
    """Also keep results from older, superseded diagnostic versions, not only the promoted one."""

    diagnostic_version: int | None = None
    """
    Keep only results from this diagnostic version, promoted or not.

    This wins over ``include_superseded``.
    """

    def version_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses for a select over ``ExecutionGroup``."""
        if self.diagnostic_version is not None:
            return [ExecutionGroup.diagnostic_version == self.diagnostic_version]
        return [] if self.include_superseded else [promoted_version_clause()]


@attrs.frozen(kw_only=True, slots=False)
class ExecutionScope(BaseFilter):
    """
    A group of fields shared by every filter: which executions to keep.

    By default every execution that is not retracted counts.
    The creation window applies before the newest execution is picked,
    so ``latest_only`` with ``created_before`` keeps each group's newest execution from before then.

    Filters over execution groups or diagnostics judge each group by its newest execution in the window.
    A group whose newest execution is retracted is dropped, unless ``include_retracted=True``.
    It never falls back to an older execution, because the newer one replaced it.
    With a window set, a group with no execution in it is dropped.

    Examples
    --------
    Values from each group's newest run in March 2026, retracted runs included:

    >>> import datetime
    >>> from climate_ref.results import MetricValueFilter
    >>> filters = MetricValueFilter(
    ...     latest_only=True,
    ...     include_retracted=True,
    ...     created_after=datetime.datetime(2026, 3, 1),
    ...     created_before=datetime.datetime(2026, 4, 1),
    ... )
    """

    include_retracted: bool = False
    """Also keep retracted executions."""

    created_after: datetime.datetime | None = None
    """Keep only executions created at or after this time, as naive UTC."""

    created_before: datetime.datetime | None = None
    """Keep only executions created before this time, as naive UTC."""

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
    """
    A group of fields shared by every filter: which input datasets the runs used.

    See [InputDatasetFilter][climate_ref.results.filters.InputDatasetFilter] for examples.
    """

    input_datasets: InputDatasetFilter | None = None
    """
    Keep only executions that ran against matching datasets.

    Filters over execution groups or diagnostics check each group's newest execution.
    """


@attrs.frozen(kw_only=True)
class MetricValueFilter(DiagnosticScope, VersionScope, ExecutionScope, InputDatasetScope):
    """
    Filter for metric values, passed to ``reader.values.scalar_values`` and ``series_values``.

    It also has every field of
    [DiagnosticScope][climate_ref.results.filters.DiagnosticScope],
    [VersionScope][climate_ref.results.filters.VersionScope],
    [ExecutionScope][climate_ref.results.filters.ExecutionScope] and
    [InputDatasetScope][climate_ref.results.filters.InputDatasetScope].

    Examples
    --------
    Mean values for two models from one diagnostic:

    >>> from climate_ref.results import MetricValueFilter
    >>> filters = MetricValueFilter(
    ...     provider_slug="example",
    ...     diagnostic_slug="global-mean-timeseries",
    ...     dimensions={"statistic": "mean", "source_id": ["ACCESS-ESM1-5", "CanESM5"]},
    ... )

    Every value from one execution, whatever its version or retraction:

    >>> filters = MetricValueFilter(execution_ids=[42], include_superseded=True, include_retracted=True)

    Only the reference series, as opposed to the model ones:

    >>> filters = MetricValueFilter(reference_only=True)
    """

    latest_only: bool = False
    """
    Keep only values from each execution group's newest execution.

    A group whose newest execution is retracted gives nothing, unless ``include_retracted=True``.
    """

    execution_ids: Sequence[int] | None = None
    """Keep only values from these executions."""

    execution_group_ids: Sequence[int] | None = None
    """Keep only values from executions in these groups."""

    dimensions: Mapping[str, str | Sequence[str]] | None = None
    """
    Keep values by CV dimension, e.g. ``{"statistic": "mean", "source_id": ["CanESM5", "MIROC6"]}``.

    A string must match exactly, and a list matches any of its values.
    An unknown dimension raises ``KeyError`` when the query runs.
    """

    isolate_ids: Sequence[int] | None = None
    """Keep only these value ids. This wins over ``exclude_ids``."""

    exclude_ids: Sequence[int] | None = None
    """Drop these value ids. Ignored when ``isolate_ids`` is set."""

    reference_only: bool | None = None
    """Series only: ``True`` keeps the reference series, ``False`` keeps the model series."""

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
class ResourceFilter(DiagnosticScope, VersionScope, ExecutionScope, InputDatasetScope):
    """
    Filter for the executions whose resource use is read by ``reader.resources``.

    It also has every field of
    [DiagnosticScope][climate_ref.results.filters.DiagnosticScope],
    [VersionScope][climate_ref.results.filters.VersionScope],
    [ExecutionScope][climate_ref.results.filters.ExecutionScope] and
    [InputDatasetScope][climate_ref.results.filters.InputDatasetScope].

    Examples
    --------
    ESMValTool runs from the last 30 days, across every diagnostic version:

    >>> import datetime
    >>> from climate_ref.results import ResourceFilter
    >>> filters = ResourceFilter(
    ...     provider_slug="esmvaltool",
    ...     created_after=datetime.datetime(2026, 10, 1),
    ...     include_superseded=True,
    ... )
    """

    latest_only: bool = False
    """Keep only each execution group's newest execution."""


@attrs.frozen(kw_only=True)
class ExecutionGroupFilter(DiagnosticScope, VersionScope, ExecutionScope, InputDatasetScope):
    """
    Filter for execution groups, passed to ``reader.executions.groups`` and ``group_rows``.

    Each group is judged by its newest execution.
    It also has every field of
    [DiagnosticScope][climate_ref.results.filters.DiagnosticScope],
    [VersionScope][climate_ref.results.filters.VersionScope],
    [ExecutionScope][climate_ref.results.filters.ExecutionScope] and
    [InputDatasetScope][climate_ref.results.filters.InputDatasetScope].

    Examples
    --------
    PMP groups whose newest run failed inside the diagnostic:

    >>> from climate_ref.models.execution import ExecutionFailureReason
    >>> from climate_ref.results import ExecutionGroupFilter
    >>> filters = ExecutionGroupFilter(
    ...     provider_slug="pmp",
    ...     successful=False,
    ...     failure_reasons=[ExecutionFailureReason.DiagnosticError],
    ... )

    Groups selected for one model, with their newest successful run:

    >>> filters = ExecutionGroupFilter(selectors={"source_id": "ACCESS-CM2"}, latest_successful=True)

    Groups waiting to be rerun:

    >>> filters = ExecutionGroupFilter(dirty=True)
    """

    dirty: bool | None = None
    """``True`` keeps groups marked to be rerun, ``False`` keeps the rest."""

    successful: bool | None = None
    """
    Keep groups by how their newest execution went.

    ``True`` keeps groups whose newest execution succeeded.
    ``False`` keeps groups whose newest execution failed, is still running, or does not exist yet.
    This does not change which execution counts as newest. ``latest_successful`` does that.
    """

    latest_successful: bool | None = None
    """
    Change which execution counts as a group's newest.

    ``True`` picks the newest successful execution, even when a later one failed.
    ``False`` picks the newest unsuccessful or running execution.
    It can be combined with ``successful``.
    """

    selectors: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """
    Keep groups by the facets they were selected with, e.g. ``{"source_id": ["ACCESS-CM2"]}``.

    A key is a bare facet such as ``source_id``, or ``<dataset_type>.<facet>`` for one dataset type.
    Unlike ``input_datasets``, this checks how the group was selected, not the datasets its runs used.
    """

    failure_reasons: tuple[ExecutionFailureReason, ...] | None = attrs.field(
        default=None, converter=_as_failure_reasons
    )
    """Keep groups whose newest execution failed for one of these reasons."""


@attrs.frozen(kw_only=True)
class DiagnosticFilter(DiagnosticScope, VersionScope, ExecutionScope, InputDatasetScope):
    """
    Filter for diagnostics and their run counts.

    Pass it to ``reader.diagnostics.list``, ``reader.diagnostics.with_values``
    or ``reader.executions.statistics``.
    It has no fields of its own, only those of
    [DiagnosticScope][climate_ref.results.filters.DiagnosticScope],
    [VersionScope][climate_ref.results.filters.VersionScope],
    [ExecutionScope][climate_ref.results.filters.ExecutionScope] and
    [InputDatasetScope][climate_ref.results.filters.InputDatasetScope].
    With ``input_datasets`` only groups whose newest run matches are counted,
    and a diagnostic left with no groups is dropped.

    Examples
    --------
    Every ENSO diagnostic, from any provider:

    >>> from climate_ref.results import DiagnosticFilter
    >>> filters = DiagnosticFilter(diagnostic_contains=["enso"])
    """
