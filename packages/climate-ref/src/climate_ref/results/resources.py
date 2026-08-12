"""
Read surface for per-execution resource measurements.

[ResourcesReader][climate_ref.results.resources.ResourcesReader] is reached via
[Reader.resources][climate_ref.results.values.Reader.resources].
It reads the resource columns on
[Execution][climate_ref.models.execution.Execution]
and aggregates them into a
[ResourceProfile][climate_ref.results.resources.ResourceProfile]
per diagnostic or per provider, which is the number a maintainer needs when sizing a worker.

Not all executions have resource data.
"""

import datetime
import math
import statistics
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import attrs
import pandas as pd
from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.provider import Provider
from climate_ref.results._converters import _as_str_tuple
from climate_ref.results._query import count_values
from climate_ref_core.resources import MemorySource

Confidence = Literal["good", "thin", "none"]
"""How much weight a profile's recommendation can carry."""

GroupBy = Literal["diagnostic", "provider"]
"""Axis a profile is aggregated over."""

_GOOD_SAMPLE_COUNT = 10
"""
Samples needed before a profile is reported as ``good``.

Below ten samples the 95th percentile sits between the top two observations,
so it carries no information the maximum does not already carry.
"""

_BYTES_PER_GIB = 1024**3

_SOURCE_PRECEDENCE = ("cgroup", "proc_tree", "rusage")
"""
Tie-break order when two memory sources contributed the same number of samples.

Ordered by how much of the process tree each one actually observes.
"""


def _percentile(values: Sequence[float], quantile: float) -> float:
    """
    Linear-interpolated percentile over a non-empty sequence.

    ``quantile`` is resolved to the nearest whole percent, which is all any caller here asks for.

    Parameters
    ----------
    values
        Observations, in any order.
    quantile
        Fraction between 0 and 1.

    Returns
    -------
    :
        The interpolated percentile.
    """
    if len(values) == 1:
        return float(values[0])

    cut_points = statistics.quantiles(values, n=100, method="inclusive")
    return float(cut_points[round(quantile * 100) - 1])


@attrs.frozen(kw_only=True)
class ResourceFilter:
    """
    Declarative filter over the executions considered for a resource profile.

    Every field is optional, and ``None`` means "do not constrain on this axis".
    ``diagnostic_contains``/``provider_contains`` are case-insensitive substring matches
    (OR-combined within each field),
    matching the semantics used by
    [ExecutionGroupFilter][climate_ref.results.executions.ExecutionGroupFilter].
    """

    diagnostic_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on diagnostic slug (OR-combined)."""

    provider_contains: tuple[str, ...] | None = attrs.field(default=None, converter=_as_str_tuple)
    """Case-insensitive substring matches on provider slug (OR-combined)."""

    since: datetime.datetime | None = None
    """Keep only executions created at or after this naive UTC timestamp."""


@attrs.frozen(kw_only=True)
class ResourceMeasurementView:
    """One execution's resource measurement, detached from the ORM."""

    execution_id: int
    """Primary key of the underlying ``Execution`` row."""

    provider_slug: str
    """Owning provider's slug."""

    diagnostic_slug: str
    """Owning diagnostic's slug."""

    successful: bool | None
    """``True``/``False`` once the execution has finished, ``None`` while still running."""

    wall_seconds: float | None
    """Wall clock time taken by the execution, in seconds."""

    cpu_seconds: float | None
    """CPU time consumed by the execution and its children, in seconds."""

    peak_memory_bytes: int | None
    """Peak resident memory observed during the execution, in bytes."""

    memory_source: MemorySource | None
    """Provenance of ``peak_memory_bytes``."""

    memory_limit_bytes: int | None
    """Memory limit in force while the execution ran, in bytes."""

    cpu_limit: float | None
    """CPU cores available to the execution."""

    resources_exclusive: bool | None
    """Whether this execution was the only one running on the worker."""

    queue_seconds: float | None
    """Time between submission and the start of the execution, in seconds."""

    created_at: Any
    """Timestamp the execution was created."""

    @property
    def parallelism(self) -> float | None:
        """Mean core occupancy, or ``None`` when either input is missing."""
        if self.cpu_seconds is None or not self.wall_seconds:
            return None
        return self.cpu_seconds / self.wall_seconds


@attrs.frozen(kw_only=True)
class ResourceMeasurementCollection:
    """An immutable page of resource measurements plus collection-level metadata."""

    items: tuple[ResourceMeasurementView, ...]
    """The measurements on this page."""

    total_count: int
    """Total measurements matching the filter before ``offset``/``limit``."""

    offset: int
    """Rows skipped before this page."""

    limit: int | None
    """Page size requested, or ``None`` when the whole result was returned."""

    def __iter__(self) -> Iterator[ResourceMeasurementView]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def to_pandas(self) -> pd.DataFrame:
        """
        DataFrame with one row per measurement.

        Columns are emitted explicitly even when the collection is empty,
        so callers can select columns without special-casing.
        """
        columns = [
            "execution_id",
            "provider",
            "diagnostic",
            "successful",
            "wall_seconds",
            "cpu_seconds",
            "peak_memory_bytes",
            "memory_source",
            "memory_limit_bytes",
            "cpu_limit",
            "resources_exclusive",
            "queue_seconds",
            "created_at",
        ]
        records = [
            {
                "execution_id": item.execution_id,
                "provider": item.provider_slug,
                "diagnostic": item.diagnostic_slug,
                "successful": item.successful,
                "wall_seconds": item.wall_seconds,
                "cpu_seconds": item.cpu_seconds,
                "peak_memory_bytes": item.peak_memory_bytes,
                "memory_source": item.memory_source,
                "memory_limit_bytes": item.memory_limit_bytes,
                "cpu_limit": item.cpu_limit,
                "resources_exclusive": item.resources_exclusive,
                "queue_seconds": item.queue_seconds,
                "created_at": item.created_at,
            }
            for item in self.items
        ]
        return pd.DataFrame.from_records(records, columns=columns)


@attrs.frozen(kw_only=True)
class ResourceProfile:
    """
    Aggregated resource usage for one diagnostic, or for one provider as a roll-up.

    Every percentile is computed over the *usable samples* only.
    A usable sample is an execution that recorded a wall time, a CPU time and a peak memory figure,
    whose memory source matches ``memory_source``,
    and which was measured exclusively when ``exclusive_only`` was requested.
    Executions that recorded nothing at all are absent from both counts,
    because an unmeasured run is not a run that used no memory.

    A run that failed is aggregated like any other, provided it recorded a measurement.
    A diagnostic that reached 40 GiB before dying is the row that most needs to reach a recommendation.
    """

    provider_slug: str
    """Owning provider's slug."""

    diagnostic_slug: str | None
    """Owning diagnostic's slug, or ``None`` for a provider-level roll-up."""

    n_samples: int
    """Executions with a usable measurement."""

    n_excluded: int
    """
    Executions that were measured but not usable.

    A row lands here when it was non-exclusive, incomplete,
    or carried a memory source other than ``memory_source``.
    """

    n_failed: int
    """
    Failed executions in the same window, measured or not.

    An execution killed for exceeding its memory limit records nothing,
    so it never appears in ``n_samples``.
    That biases every recommendation here low on exactly the diagnostics that need more memory,
    and this count is the only warning of it.
    """

    memory_source: MemorySource | None
    """
    Provenance of every peak memory figure aggregated here, or ``None`` when there are no samples.

    A cgroup reading and a rusage reading for the same run can differ by a factor of two,
    so they are never mixed.
    When a group's executions carry more than one source,
    the source contributing the most samples wins and the rest land in ``n_excluded``.
    """

    peak_memory_p50: int
    """Median peak resident memory, in bytes."""

    peak_memory_p95: int
    """95th percentile peak resident memory, in bytes."""

    peak_memory_max: int
    """Largest peak resident memory observed, in bytes."""

    wall_p95: float
    """95th percentile wall clock time, in seconds."""

    cpu_seconds_p95: float
    """95th percentile CPU time, in seconds."""

    parallelism_p95: float
    """
    95th percentile of ``cpu_seconds / wall_seconds``.

    This is the core-count signal.
    A diagnostic sitting near 1.0 is serial and gains nothing from more cores.
    """

    memory_limit_seen: int | None
    """
    Memory limit in force for the most recent sample that recorded one.

    The most recent limit rather than the largest, because the question being answered is
    whether the container as it is configured now is big enough.
    """

    headroom_ratio: float | None
    """
    ``memory_limit_seen / peak_memory_p95``, or ``None`` when no limit was recorded.

    Below 1 means the diagnostic is being killed by the current container size.
    """

    safety_factor: float = 1.3
    """Multiplier applied to the p95 peak when recommending a memory size."""

    @property
    def recommended_memory_bytes(self) -> int:
        """Peak memory p95 scaled by ``safety_factor``, rounded up to a whole GiB."""
        scaled = self.peak_memory_p95 * self.safety_factor
        return max(1, math.ceil(scaled / _BYTES_PER_GIB)) * _BYTES_PER_GIB

    @property
    def recommended_cpus(self) -> int:
        """Mean parallelism at the 95th percentile, rounded up, never below one core."""
        return max(1, math.ceil(self.parallelism_p95))

    @property
    def confidence(self) -> Confidence:
        """
        How much weight the recommendation can carry.

        ``none`` when there are no usable samples at all,
        ``thin`` below ten samples, where the p95 is indistinguishable from the maximum,
        and ``good`` at or above ten.
        """
        if self.n_samples == 0:
            return "none"
        if self.n_samples < _GOOD_SAMPLE_COUNT:
            return "thin"
        return "good"

    @property
    def over_limit(self) -> bool:
        """Whether the p95 peak has outgrown the recorded memory limit."""
        return self.headroom_ratio is not None and self.headroom_ratio < 1.0


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
        .join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id)
        .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .where(
            or_(
                Execution.wall_seconds.is_not(None),
                Execution.peak_memory_bytes.is_not(None),
                Execution.successful.is_(False),
            )
        )
        .order_by(Execution.created_at, Execution.id)
    )

    if filters.diagnostic_contains:
        stmt = stmt.where(
            or_(*(Diagnostic.slug.ilike(f"%{s.lower()}%") for s in filters.diagnostic_contains))
        )
    if filters.provider_contains:
        stmt = stmt.where(or_(*(Provider.slug.ilike(f"%{s.lower()}%") for s in filters.provider_contains)))
    if filters.since is not None:
        stmt = stmt.where(Execution.created_at >= filters.since)

    return stmt


def _source_precedence(source: str) -> int:
    return _SOURCE_PRECEDENCE.index(source) if source in _SOURCE_PRECEDENCE else len(_SOURCE_PRECEDENCE)


def _dominant_source(weights: dict[MemorySource, int]) -> MemorySource | None:
    """
    Heaviest-weighted memory source, tie-broken by ``_SOURCE_PRECEDENCE``.

    Weights are row counts per diagnostic and sample counts per provider,
    so the same rule decides both levels.
    """
    if not weights:
        return None
    return sorted(weights, key=lambda source: (-weights[source], _source_precedence(source)))[0]


def _source_weights(rows: Sequence[ResourceMeasurementView]) -> dict[MemorySource, int]:
    """Count usable rows per memory source, ignoring rows that named no source."""
    weights: dict[MemorySource, int] = {}
    for row in rows:
        source = row.memory_source
        if source is None or source == "unavailable":
            continue
        weights[source] = weights.get(source, 0) + 1
    return weights


def _is_measured(row: ResourceMeasurementView) -> bool:
    return row.wall_seconds is not None or row.cpu_seconds is not None or row.peak_memory_bytes is not None


def _is_complete(row: ResourceMeasurementView) -> bool:
    return (
        row.wall_seconds is not None
        and row.wall_seconds > 0
        and row.cpu_seconds is not None
        and row.peak_memory_bytes is not None
    )


def _empty_profile(
    *,
    provider_slug: str,
    diagnostic_slug: str | None,
    n_excluded: int,
    n_failed: int,
    safety_factor: float,
) -> ResourceProfile:
    return ResourceProfile(
        provider_slug=provider_slug,
        diagnostic_slug=diagnostic_slug,
        n_samples=0,
        n_excluded=n_excluded,
        n_failed=n_failed,
        memory_source=None,
        peak_memory_p50=0,
        peak_memory_p95=0,
        peak_memory_max=0,
        wall_p95=0.0,
        cpu_seconds_p95=0.0,
        parallelism_p95=0.0,
        memory_limit_seen=None,
        headroom_ratio=None,
        safety_factor=safety_factor,
    )


def _build_profile(
    *,
    provider_slug: str,
    diagnostic_slug: str | None,
    rows: Sequence[ResourceMeasurementView],
    exclusive_only: bool,
    safety_factor: float,
) -> ResourceProfile:
    """
    Aggregate one diagnostic's rows into a profile.

    Success is not a filter.
    A failed run that recorded a peak is the most informative row available,
    because a diagnostic outgrowing its container fails at the size it needs reported.
    """
    n_failed = sum(1 for row in rows if row.successful is False)

    measured = [row for row in rows if _is_measured(row)]
    candidates = [row for row in measured if _is_complete(row)]
    if exclusive_only:
        candidates = [row for row in candidates if row.resources_exclusive is True]

    source = _dominant_source(_source_weights(candidates))
    samples = [row for row in candidates if row.memory_source == source] if source else []
    n_excluded = len(measured) - len(samples)

    if not samples:
        return _empty_profile(
            provider_slug=provider_slug,
            diagnostic_slug=diagnostic_slug,
            n_excluded=n_excluded,
            n_failed=n_failed,
            safety_factor=safety_factor,
        )

    peaks = [row.peak_memory_bytes for row in samples if row.peak_memory_bytes is not None]
    walls = [row.wall_seconds for row in samples if row.wall_seconds is not None]
    cpus = [row.cpu_seconds for row in samples if row.cpu_seconds is not None]
    parallelisms = [row.parallelism for row in samples if row.parallelism is not None]

    limits = [row.memory_limit_bytes for row in samples if row.memory_limit_bytes is not None]
    limit_seen = limits[-1] if limits else None

    peak_p95 = round(_percentile(peaks, 0.95))
    headroom = limit_seen / peak_p95 if limit_seen is not None and peak_p95 > 0 else None

    return ResourceProfile(
        provider_slug=provider_slug,
        diagnostic_slug=diagnostic_slug,
        n_samples=len(samples),
        n_excluded=n_excluded,
        n_failed=n_failed,
        memory_source=source,
        peak_memory_p50=round(_percentile(peaks, 0.5)),
        peak_memory_p95=peak_p95,
        peak_memory_max=max(peaks),
        wall_p95=_percentile(walls, 0.95),
        cpu_seconds_p95=_percentile(cpus, 0.95),
        parallelism_p95=_percentile(parallelisms, 0.95),
        memory_limit_seen=limit_seen,
        headroom_ratio=headroom,
        safety_factor=safety_factor,
    )


def _roll_up(provider_slug: str, profiles: Sequence[ResourceProfile]) -> ResourceProfile:
    """
    Combine a provider's per-diagnostic profiles into one worker-sizing answer.

    Memory takes the maximum over diagnostics, because a worker has to fit its largest task.
    Cores take the 95th percentile of the per-diagnostic parallelism,
    so one unusually threaded diagnostic does not size the whole pool.
    This is not a re-aggregation of the raw rows,
    which would let a large population of small executions hide the one that needs the memory.

    The limit reported is the one the peak-driving diagnostic ran under,
    so ``headroom_ratio`` compares a peak against the container that peak actually faced.
    A minimum across diagnostics would pair the largest peak with an unrelated smaller container
    and flag ``OVER`` where no diagnostic was ever over.

    Sources are not mixed here either.
    A diagnostic measured by a different source than the provider's dominant one is left out,
    and its samples are counted in ``n_excluded``.
    """
    measured = [profile for profile in profiles if profile.n_samples]
    safety_factor = profiles[0].safety_factor
    n_excluded = sum(profile.n_excluded for profile in profiles)
    n_failed = sum(profile.n_failed for profile in profiles)

    source_weights: dict[MemorySource, int] = {}
    for profile in measured:
        if profile.memory_source is not None:
            source_weights[profile.memory_source] = (
                source_weights.get(profile.memory_source, 0) + profile.n_samples
            )
    source = _dominant_source(source_weights)

    with_samples = [profile for profile in measured if profile.memory_source == source]
    n_excluded += sum(profile.n_samples for profile in measured if profile.memory_source != source)

    if not with_samples:
        return _empty_profile(
            provider_slug=provider_slug,
            diagnostic_slug=None,
            n_excluded=n_excluded,
            n_failed=n_failed,
            safety_factor=safety_factor,
        )

    driver = max(with_samples, key=lambda profile: profile.peak_memory_p95)
    limit_seen = driver.memory_limit_seen

    peak_p95 = driver.peak_memory_p95
    headroom = limit_seen / peak_p95 if limit_seen is not None and peak_p95 > 0 else None

    return ResourceProfile(
        provider_slug=provider_slug,
        diagnostic_slug=None,
        n_samples=sum(profile.n_samples for profile in with_samples),
        n_excluded=n_excluded,
        n_failed=n_failed,
        memory_source=source,
        peak_memory_p50=max(profile.peak_memory_p50 for profile in with_samples),
        peak_memory_p95=peak_p95,
        peak_memory_max=max(profile.peak_memory_max for profile in with_samples),
        wall_p95=max(profile.wall_p95 for profile in with_samples),
        cpu_seconds_p95=max(profile.cpu_seconds_p95 for profile in with_samples),
        parallelism_p95=_percentile([profile.parallelism_p95 for profile in with_samples], 0.95),
        memory_limit_seen=limit_seen,
        headroom_ratio=headroom,
        safety_factor=safety_factor,
    )


class ResourcesReader:
    """
    Per-execution resource measurement read domain.

    Constructed from a [Database][climate_ref.database.Database],
    which owns the session and the read-only story.
    All read methods return detached DTOs that outlive the session.
    """

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._db.session

    def _to_view(self, row: Any) -> ResourceMeasurementView:
        return ResourceMeasurementView(
            execution_id=row.execution_id,
            provider_slug=row.provider_slug,
            diagnostic_slug=row.diagnostic_slug,
            successful=row.successful,
            wall_seconds=row.wall_seconds,
            cpu_seconds=row.cpu_seconds,
            peak_memory_bytes=row.peak_memory_bytes,
            memory_source=row.memory_source,
            memory_limit_bytes=row.memory_limit_bytes,
            cpu_limit=row.cpu_limit,
            resources_exclusive=row.resources_exclusive,
            queue_seconds=row.queue_seconds,
            created_at=row.created_at,
        )

    def measurements(
        self,
        filters: ResourceFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> ResourceMeasurementCollection:
        """
        Query the raw per-execution measurements behind the profiles.

        Pagination is applied in SQL over the deterministic ``created_at, id`` ordering.

        Parameters
        ----------
        filters
            Restricts the executions considered. ``None`` means no restriction.
        offset
            Rows to skip before the returned page.
        limit
            Page size, or ``None`` for every matching row.

        Returns
        -------
        :
            A page of measurements plus the pre-pagination total.
        """
        stmt = select_execution_resources(filters)
        total_count = count_values(self.session, stmt)

        page = stmt.offset(offset)
        if limit is not None:
            page = page.limit(limit)

        items = tuple(self._to_view(row) for row in self.session.execute(page).all())
        return ResourceMeasurementCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def measurement(self, execution_id: int) -> ResourceMeasurementView | None:
        """
        Fetch one execution's measurement by execution id.

        Returns ``None`` when no execution has that id,
        or when the execution neither recorded a measurement nor failed.
        """
        stmt = select_execution_resources().where(Execution.id == execution_id)
        row = self.session.execute(stmt).one_or_none()
        return self._to_view(row) if row is not None else None

    def profiles(  # noqa: PLR0913
        self,
        *,
        diagnostic_contains: Sequence[str] | None = None,
        provider_contains: Sequence[str] | None = None,
        since: datetime.datetime | None = None,
        group_by: GroupBy = "diagnostic",
        exclusive_only: bool = True,
        safety_factor: float = 1.3,
    ) -> tuple[ResourceProfile, ...]:
        """
        Aggregate resource measurements into a sizing answer per diagnostic or per provider.

        Rows are filtered in SQL and the percentiles are computed in Python,
        because SQLite has no percentile function.

        A cgroup peak recorded while a worker ran four executions at once measures the worker,
        not the execution, so ``exclusive_only`` defaults to ``True``.
        Those rows are counted in ``n_excluded`` rather than dropped silently.

        Failed executions that recorded a measurement are aggregated alongside successful ones.
        A run that died at 40 GiB is the strongest evidence there is about what the diagnostic needs.

        Peaks from different memory sources are never mixed.
        Within each group the source contributing the most samples wins,
        the rest are counted in ``n_excluded``,
        and the winner is reported as ``memory_source``.

        Parameters
        ----------
        diagnostic_contains
            Case-insensitive substring matches on diagnostic slug (OR-combined).
        provider_contains
            Case-insensitive substring matches on provider slug (OR-combined).
        since
            Keep only executions created at or after this naive UTC timestamp.
        group_by
            ``diagnostic`` for one profile per diagnostic,
            or ``provider`` for the worker-sizing roll-up across a provider's diagnostics.
        exclusive_only
            Restrict samples to executions measured while nothing else ran on the worker.
        safety_factor
            Multiplier applied to the p95 peak by ``recommended_memory_bytes``.

        Returns
        -------
        :
            One profile per group, ordered by provider then diagnostic slug.
        """
        filters = ResourceFilter(
            diagnostic_contains=diagnostic_contains,
            provider_contains=provider_contains,
            since=since,
        )
        rows = [self._to_view(row) for row in self.session.execute(select_execution_resources(filters)).all()]

        grouped: dict[tuple[str, str], list[ResourceMeasurementView]] = {}
        for row in rows:
            grouped.setdefault((row.provider_slug, row.diagnostic_slug), []).append(row)

        per_diagnostic = [
            _build_profile(
                provider_slug=provider_slug,
                diagnostic_slug=diagnostic_slug,
                rows=group_rows,
                exclusive_only=exclusive_only,
                safety_factor=safety_factor,
            )
            for (provider_slug, diagnostic_slug), group_rows in sorted(grouped.items())
        ]

        if group_by == "diagnostic":
            return tuple(per_diagnostic)

        by_provider: dict[str, list[ResourceProfile]] = {}
        for profile in per_diagnostic:
            by_provider.setdefault(profile.provider_slug, []).append(profile)
        return tuple(_roll_up(slug, profiles) for slug, profiles in sorted(by_provider.items()))
