"""
Filters that choose which results a read returns.

Build a filter and pass it to a reader method, or to the matching builder in
[queries][climate_ref.results.queries].
Filters are frozen [attrs](https://www.attrs.org) classes, so one filter can be reused across reads.

## Which filter to use

Each reader method takes one kind of filter:

* [MetricValueFilter][climate_ref.results.filters.MetricValueFilter]:
  ``reader.values.scalar_values`` and ``series_values``.
* [ExecutionGroupFilter][climate_ref.results.filters.ExecutionGroupFilter]:
  ``reader.executions.groups`` and ``group_rows``.
* [DiagnosticFilter][climate_ref.results.filters.DiagnosticFilter]:
  ``reader.diagnostics.list`` and ``with_values``.
* [ResourceFilter][climate_ref.results.filters.ResourceFilter]:
  ``reader.resources.measurements``, ``summary``, ``summaries`` and ``profiles``.
* [DatasetFilter][climate_ref.models.dataset_query.DatasetFilter]:
  ``reader.datasets``. It filters the datasets themselves rather than executions.

## Defaults

Every field is optional, and a field left as ``None`` does not narrow the result.
Two defaults still apply to an empty filter, on every filter except ``DatasetFilter``:

* Only results from each diagnostic's promoted version are kept.
  Pass ``include_superseded=True`` for every version.
* Retracted executions are left out.
  Pass ``include_retracted=True`` to keep them.

So ``MetricValueFilter()`` matches every value from a promoted, non-retracted execution.

## Shared fields

Most fields mean the same thing, with the same default, on every filter that has them.
They are grouped into scopes, and each filter is built from the scopes it needs:

* [DiagnosticScope][climate_ref.results.filters.DiagnosticScope]: which diagnostics and providers,
  by exact slug (``diagnostic_slug``, ``provider_slug``)
  or by substring (``diagnostic_contains``, ``provider_contains``).
* [VersionScope][climate_ref.results.filters.VersionScope]: which diagnostic versions
  (``include_superseded``, ``diagnostic_version``).
* [ExecutionScope][climate_ref.results.filters.ExecutionScope]: which executions
  (``include_retracted``, ``created_after``, ``created_before``, ``execution_ids``, ``execution_group_ids``,
  ``source_types``, ``or_no_model_input``).
* [InputDatasetScope][climate_ref.results.filters.InputDatasetScope]: which input datasets the executions used
  (``input_datasets``).

Fields that take a list keep rows matching any of its values, so an empty list keeps nothing.
``exclude_ids`` is the exception: an empty list drops nothing.

## Latest execution only

An execution group can have several executions, one each time it is rerun.
``MetricValueFilter`` and ``ResourceFilter`` read single executions,
so they have ``latest_only`` to keep only each group's newest one.
``ExecutionGroupFilter`` and ``DiagnosticFilter`` always judge a group by its newest execution,
so they do not need it.

## Filtering by input dataset

[InputDatasetFilter][climate_ref.results.filters.InputDatasetFilter] keeps executions by the datasets they ran
against, such as a ``source_id`` or a list of dataset ids.
Pass it as ``input_datasets`` on any filter.

Examples
--------
The latest PMP values from executions created this year:

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

Fields that pick slugs, ids, source types or failure reasons take one value or a list,
and a list matches any of its values:

>>> MetricValueFilter(diagnostic_slug="enso_tel").diagnostic_slug
('enso_tel',)
>>> MetricValueFilter(diagnostic_slug=["enso_tel", "enso_perf"], source_types="cmip6").source_types
(<SourceDatasetType.CMIP6: 'cmip6'>,)
"""

import datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import attrs
from sqlalchemy import ColumnElement, Select, and_, exists, false, or_, select, true

from climate_ref.models.dataset import Dataset
from climate_ref.models.dataset_query import _as_facets, _entity_for
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import (
    Execution,
    ExecutionFailureReason,
    ExecutionGroup,
    ExecutionGroupSourceType,
    execution_datasets,
)
from climate_ref.models.metric_value import MetricValue
from climate_ref.models.provider import Provider
from climate_ref.results._converters import (
    _as_failure_reasons,
    _as_id_tuple,
    _as_naive_utc,
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


_MODEL_SOURCE_TYPES = tuple(t for t in SourceDatasetType if t.is_model_output)


def _dataset_types() -> tuple[SourceDatasetType, ...]:
    return tuple(t for t in SourceDatasetType if t in Dataset.__mapper__.polymorphic_map)


def _has_facets(source_type: SourceDatasetType, facets: Iterable[str]) -> bool:
    columns = _entity_for(source_type).__mapper__.columns
    return all(facet in columns for facet in facets)


def _check_known_facet(facet: str, types: Iterable[SourceDatasetType], name: str) -> None:
    if not any(_has_facets(t, [facet]) for t in types):
        raise ValueError(f"Unknown {name} {facet!r}. It is not a column on any dataset type")


def split_selector_key(key: str) -> tuple[str | None, str]:
    """Split a ``selectors`` key into its dataset type, if it names one, and its facet."""
    dataset_type, dot, facet = key.partition(".")
    return (dataset_type, facet) if dot else (None, key)


def _check_selector_keys(_instance: object, _attribute: object, selectors: Mapping[str, Any] | None) -> None:
    """Reject a ``selectors`` key that names no dataset type or facet, which would match nothing."""
    types = {t.value: t for t in _dataset_types()}
    for key in selectors or {}:
        dataset_type, facet = split_selector_key(key)
        if dataset_type is not None and dataset_type not in types:
            raise ValueError(f"Unknown dataset type {dataset_type!r} in selector {key!r}")
        _check_known_facet(facet, [types[dataset_type]] if dataset_type else types.values(), "selector")


def _contains_any(column: Any, searches: Iterable[str]) -> ColumnElement[bool]:
    return or_(false(), *(column.icontains(s, autoescape=True) for s in searches))


@attrs.frozen(kw_only=True)
class InputDatasetFilter:
    """
    Match executions by the datasets they ran against.

    Pass it as ``input_datasets`` on any reader filter.
    An execution matches when one of its input datasets passes every condition:

    * It has one of the listed values for every facet in ``facets``.
    * Its id is one of ``dataset_ids``, if set.
    * Its type is one of the reader filter's ``source_types``, if set.

    A dataset type without a requested facet never matches,
    so a CMIP7-only facet does not pull in all of CMIP6.

    Raises
    ------
    ValueError
        If neither ``facets`` nor ``dataset_ids`` is set, or no dataset type has every facet.

    Examples
    --------
    Executions that used ACCESS-CM2 data, of any dataset type:

    >>> from climate_ref.results import InputDatasetFilter, MetricValueFilter
    >>> access = InputDatasetFilter(facets={"source_id": "ACCESS-CM2"})
    >>> filters = MetricValueFilter(input_datasets=access)

    A single facet value can be a bare string:

    >>> access.facets
    {'source_id': ('ACCESS-CM2',)}

    Only CMIP6 executions that used ACCESS-CM2 data.
    A group that also draws on CMIP7 does not match through a CMIP7 ACCESS-CM2 dataset:

    >>> filters = MetricValueFilter(source_types="cmip6", input_datasets=access)

    Executions that used particular datasets:

    >>> InputDatasetFilter(dataset_ids=[12, 13]).dataset_ids
    (12, 13)

    A facet that is not a column on any dataset type is rejected:

    >>> InputDatasetFilter(facets={"model": ["x"]})
    Traceback (most recent call last):
    ...
    ValueError: Unknown facet 'model'. It is not a column on any dataset type
    """

    dataset_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
    """Dataset ids to match. An empty list matches nothing."""

    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    """
    Dataset attributes to match, e.g. ``{"source_id": ["ACCESS-CM2", "CanESM5"]}``.

    Values for one facet are alternatives. Every facet must match.
    """

    def __attrs_post_init__(self) -> None:
        if not self.facets and self.dataset_ids is None:
            raise ValueError("Set `facets` or `dataset_ids`")
        types = _dataset_types()
        for facet in self.facets or {}:
            _check_known_facet(facet, types, "facet")
        if self.facets and not any(_has_facets(t, self.facets) for t in types):
            raise ValueError(f"No dataset type has every facet in {sorted(self.facets)}")

    def execution_clause(
        self, source_types: Sequence[SourceDatasetType] | None = None
    ) -> ColumnElement[bool]:
        """
        Clause over ``Execution`` that is true when the execution ran against a matching dataset.

        ``source_types``, the reader filter's own, keeps only datasets of those types.

        Raises
        ------
        ValueError
            If no type in ``source_types`` has every facet.
        """
        if self.dataset_ids == ():
            return false()
        facets = self.facets or {}
        if not facets:
            used = select(execution_datasets.c.execution_id).where(
                execution_datasets.c.dataset_id.in_(self.dataset_ids or ())
            )
            if source_types is not None:
                used = used.join(Dataset, Dataset.id == execution_datasets.c.dataset_id).where(
                    Dataset.dataset_type.in_(source_types)
                )
            return Execution.id.in_(used)
        candidates = _dataset_types() if source_types is None else source_types
        types = [t for t in candidates if _has_facets(t, facets)]
        if candidates and not types:
            names = ", ".join(t.value for t in candidates)
            raise ValueError(f"No dataset type in {names} has every facet in {sorted(facets)}")
        branches: list[ColumnElement[bool]] = []
        for source_type in types:
            entity = _entity_for(source_type)
            conditions = [getattr(entity, facet).in_(values) for facet, values in facets.items()]
            if self.dataset_ids is not None:
                conditions.append(entity.id.in_(self.dataset_ids))
            branches.append(Execution.datasets.of_type(entity).any(and_(true(), *conditions)))
        # An empty or_() emits no predicate, which would match every execution.
        if not branches:
            return false()
        return or_(*branches)


@attrs.frozen(kw_only=True, slots=False)
class DiagnosticScope(BaseFilter):
    """
    A group of fields to filter for a given set of diagnostics or providers.

    Use the ``*_slug`` fields for exact diagnostics or providers,
    and the ``*_contains`` fields to search by part of a name.
    Each field takes one string or a list, and a list matches any of its values.
    Every field you set must match.

    Examples
    --------
    One exact diagnostic, then every diagnostic whose name mentions ENSO or the annual cycle:

    >>> from climate_ref.results import DiagnosticFilter
    >>> one = DiagnosticFilter(provider_slug="pmp", diagnostic_slug="enso_tel")
    >>> search = DiagnosticFilter(diagnostic_contains=["enso", "annual-cycle"])
    """

    diagnostic_slug: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Keep only the diagnostics with exactly these slugs, e.g. ``"enso_tel"``."""

    provider_slug: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Keep only the diagnostics of the providers with exactly these slugs, e.g. ``"pmp"``."""

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Keep diagnostics whose slug contains any of these, ignoring case, e.g. ``"enso"`` or ``["enso"]``."""

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Keep the diagnostics of providers whose slug contains any of these, ignoring case."""

    @property
    def matches_any(self) -> bool:
        """Whether any diagnostic or provider axis is set."""
        return (
            self.diagnostic_slug is not None or self.diagnostic_contains is not None or self.matches_provider
        )

    @property
    def matches_provider(self) -> bool:
        """Whether any provider axis is set."""
        return self.provider_slug is not None or self.provider_contains is not None

    def diagnostic_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses for a select joined to ``Diagnostic``, and to ``Provider`` if a provider axis is set."""
        clauses: list[ColumnElement[bool]] = []
        if self.diagnostic_slug is not None:
            clauses.append(Diagnostic.slug.in_(self.diagnostic_slug))
        if self.diagnostic_contains is not None:
            clauses.append(_contains_any(Diagnostic.slug, self.diagnostic_contains))
        if self.provider_slug is not None:
            clauses.append(Provider.slug.in_(self.provider_slug))
        if self.provider_contains is not None:
            clauses.append(_contains_any(Provider.slug, self.provider_contains))
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
    The creation window and the id lists apply before the newest execution is picked,
    so ``latest_only`` with ``created_before`` keeps each group's newest execution from before then.

    Filters over execution groups or diagnostics judge each group by its newest execution that is kept.
    A group whose newest execution is retracted is dropped, unless ``include_retracted=True``.
    It never falls back to an older execution, because the newer one replaced it.
    With a window or id list set, a group with no kept execution is dropped.

    Examples
    --------
    Values from each group's newest execution in March 2026, retracted executions included:

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

    created_after: datetime.datetime | None = attrs.field(default=None, converter=_as_naive_utc)
    """Keep only executions created at or after this time. A naive time is read as UTC."""

    created_before: datetime.datetime | None = attrs.field(default=None, converter=_as_naive_utc)
    """Keep only executions created before this time. A naive time is read as UTC."""

    execution_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
    """Keep only these executions."""

    execution_group_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
    """Keep only executions in these groups."""

    source_types: tuple[SourceDatasetType, ...] | None = attrs.field(default=None, converter=_as_source_types)
    """
    Keep groups that draw on any of these source types, with all their executions.

    Pass one type or a list. An empty list matches nothing, unless ``or_no_model_input`` is set.
    A group that has never been executed is kept if it draws on one of the types.
    """

    or_no_model_input: bool = False
    """
    Also keep groups that use no model source type, such as observation-only ones.

    This needs ``source_types``.
    """

    def __attrs_post_init__(self) -> None:
        if self.or_no_model_input and self.source_types is None:
            raise ValueError("`or_no_model_input` needs `source_types`")

    @property
    def narrows_executions(self) -> bool:
        """Whether a time or execution id narrows which executions count, so a group needs one of them."""
        return bool(self._narrowing_clauses())

    def _narrowing_clauses(self) -> list[ColumnElement[bool]]:
        clauses: list[ColumnElement[bool]] = []
        if self.created_after is not None:
            clauses.append(Execution.created_at >= self.created_after)
        if self.created_before is not None:
            clauses.append(Execution.created_at < self.created_before)
        if self.execution_ids is not None:
            clauses.append(Execution.id.in_(self.execution_ids))
        return clauses

    def execution_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses over ``Execution`` that narrow which executions count, before the newest is picked."""
        clauses = self._narrowing_clauses()
        if self.execution_group_ids is not None:
            clauses.append(Execution.execution_group_id.in_(self.execution_group_ids))
        return clauses

    def group_clauses(self) -> list[ColumnElement[bool]]:
        """Clauses over ``ExecutionGroup`` for the source types a group draws on."""
        if self.source_types is None:
            return []
        branches = [_group_uses(self.source_types) if self.source_types else false()]
        if self.or_no_model_input:
            branches.append(~_group_uses(_MODEL_SOURCE_TYPES))
        return [or_(*branches)]


def _group_uses(source_types: Sequence[SourceDatasetType]) -> ColumnElement[bool]:
    return exists().where(
        ExecutionGroupSourceType.execution_group_id == ExecutionGroup.id,
        ExecutionGroupSourceType.source_type.in_(source_types),
    )


@attrs.frozen(kw_only=True, slots=False)
class InputDatasetScope(BaseFilter):
    """
    A group of fields shared by every filter: which input datasets the executions used.

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

    dimensions: Mapping[str, str | Sequence[str]] | None = None
    """
    Keep values by CV dimension, e.g. ``{"statistic": "mean", "source_id": ["CanESM5", "MIROC6"]}``.

    A string must match exactly, and a list matches any of its values.
    An unknown dimension raises ``KeyError`` when the query runs.
    """

    isolate_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
    """Keep only these value ids. This wins over ``exclude_ids``."""

    exclude_ids: tuple[int, ...] | None = attrs.field(default=None, converter=_as_id_tuple)
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
            wanted: list[Any] = (
                [value] if isinstance(value, str) or not isinstance(value, Sequence) else list(value)
            )
            if key == "kind" and "model" in wanted:
                # A value with no ``kind`` reads as "model".
                wanted.append(None)
            clause = col.in_([v for v in wanted if v is not None])
            clauses.append(or_(clause, col.is_(None)) if None in wanted else clause)
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
    ESMValTool executions from the last 30 days, across every diagnostic version:

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
    PMP groups whose newest execution failed inside the diagnostic:

    >>> from climate_ref.models.execution import ExecutionFailureReason
    >>> from climate_ref.results import ExecutionGroupFilter
    >>> filters = ExecutionGroupFilter(
    ...     provider_slug="pmp",
    ...     successful=False,
    ...     failure_reasons=[ExecutionFailureReason.DiagnosticError],
    ... )

    Groups selected for one model, with their newest successful execution:

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
    Retracted executions are skipped unless ``include_retracted`` is set,
    and a group with no such execution is left out.
    It can be combined with ``successful``.
    """

    selectors: Mapping[str, tuple[str, ...]] | None = attrs.field(
        default=None, converter=_as_facets, validator=_check_selector_keys
    )
    """
    Keep groups by the facets they were selected with, e.g. ``{"source_id": ["ACCESS-CM2"]}``.

    A key is a bare facet such as ``source_id``, or ``<dataset_type>.<facet>`` for one dataset type.
    Unlike ``input_datasets``, this checks how the group was selected, not the datasets its executions used.
    """

    failure_reasons: tuple[ExecutionFailureReason, ...] | None = attrs.field(
        default=None, converter=_as_failure_reasons
    )
    """Keep groups whose newest execution failed for one of these reasons."""


@attrs.frozen(kw_only=True)
class DiagnosticFilter(DiagnosticScope, VersionScope, ExecutionScope, InputDatasetScope):
    """
    Filter for diagnostics and their execution counts.

    Pass it to ``reader.diagnostics.list`` or ``reader.diagnostics.with_values``.
    It has no fields of its own, only those of
    [DiagnosticScope][climate_ref.results.filters.DiagnosticScope],
    [VersionScope][climate_ref.results.filters.VersionScope],
    [ExecutionScope][climate_ref.results.filters.ExecutionScope] and
    [InputDatasetScope][climate_ref.results.filters.InputDatasetScope].
    With ``input_datasets`` only groups whose newest execution matches are counted,
    and a diagnostic left with no groups is dropped.

    Examples
    --------
    Every ENSO diagnostic, from any provider:

    >>> from climate_ref.results import DiagnosticFilter
    >>> filters = DiagnosticFilter(diagnostic_contains=["enso"])
    """
