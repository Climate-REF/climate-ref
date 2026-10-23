"""
How much time, CPU and memory did diagnostic executions use?

Each execution can record how long it took, the CPU time it used and its peak memory.

[ResourcesReader][climate_ref.results.resources.ResourcesReader] has five read methods.
They differ in how much they combine:

* [measurements][climate_ref.results.resources.ResourcesReader.measurements]:
  one row per execution, as a paged collection.
* [measurement][climate_ref.results.resources.ResourcesReader.measurement]:
  one execution, looked up by id.
* [summary][climate_ref.results.resources.ResourcesReader.summary]:
  one set of totals and extremes over every matching execution.
* [summaries][climate_ref.results.resources.ResourcesReader.summaries]:
  the same totals and extremes, one per diagnostic or provider.
* [profiles][climate_ref.results.resources.ResourcesReader.profiles]:
  percentiles and a recommended memory and core count per diagnostic or provider.

Examples
--------
How much memory and how many cores should each PMP diagnostic get?

```python
from climate_ref.results import Reader, ResourceFilter

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    for profile in reader.resources.profiles(ResourceFilter(provider_slug="pmp")):
        print(profile.diagnostic_slug, profile.recommended_memory_bytes, profile.recommended_cpus)
```
"""

import math
import statistics
from collections.abc import Sequence
from typing import Any, Literal

import attrs
import pandas as pd
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.execution import Execution
from climate_ref.query import (
    GroupBy,
    ResourceFilter,
    count,
    paginate,
    select_execution_resources,
    select_resource_summary,
)
from climate_ref.results._page import Page
from climate_ref_core.resources import MemorySource

Confidence = Literal["good", "thin", "none"]
"""How much weight a profile's recommendation can carry."""

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
class ResourceMeasurementView:
    """
    The time, CPU and memory one execution used.

    Every measured field is ``None`` when the execution did not record it.
    These are the executions behind a [ResourceProfile][climate_ref.results.resources.ResourceProfile].
    Get them from [ResourcesReader.measurements][climate_ref.results.resources.ResourcesReader.measurements]
    or [ResourcesReader.measurement][climate_ref.results.resources.ResourcesReader.measurement].

    Examples
    --------
    How long did execution 42 take, and how much memory did it peak at?

    ```python
    with Database.from_config(config, read_only=True) as db:
        view = Reader(db).resources.measurement(42)
        if view is not None:
            print(view.diagnostic_slug, view.wall_seconds, view.peak_memory_bytes, view.parallelism)
    ```
    """

    execution_id: int
    """Id of the execution."""

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
        """Mean core occupancy (``cpu_seconds / wall_seconds``), or ``None`` when either input is missing."""
        if self.cpu_seconds is None or not self.wall_seconds:
            return None
        return self.cpu_seconds / self.wall_seconds


@attrs.frozen(kw_only=True)
class ResourceSummary:
    """
    Totals and extremes of the resource usage recorded across a set of executions.

    Only executions that recorded a wall time are counted in ``timed_execution_count``.
    CPU time and peak memory are optional, so they may be missing even when wall time is present.

    A summary holds sums, means and extremes.
    Use a [ResourceProfile][climate_ref.results.resources.ResourceProfile] for percentiles and sizing advice.
    Get them from [ResourcesReader.summary][climate_ref.results.resources.ResourcesReader.summary]
    or [ResourcesReader.summaries][climate_ref.results.resources.ResourcesReader.summaries].

    Examples
    --------
    How long did each ILAMB diagnostic take, and what was its largest memory peak?

    ```python
    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        for s in reader.resources.summaries(ResourceFilter(provider_slug="ilamb"), group_by="diagnostic"):
            print(s.diagnostic_slug, s.timed_execution_count, s.wall_seconds_mean, s.peak_memory_bytes_max)
    ```
    """

    provider_slug: str | None
    """Provider the summary covers, or ``None`` when it is not grouped by provider."""

    diagnostic_slug: str | None
    """Diagnostic the summary covers, or ``None`` when it is not grouped by diagnostic."""

    timed_execution_count: int
    """Number of executions that recorded a wall time."""

    wall_seconds_total: float
    """
    Sum of wall clock time across the timed executions, in seconds.

    Executions often run in parallel, so this can be more than the elapsed time of the whole batch.
    """

    wall_seconds_mean: float
    """Mean wall clock time per timed execution, in seconds."""

    wall_seconds_max: float
    """Longest wall clock time of any timed execution, in seconds."""

    cpu_seconds_total: float | None
    """Sum of CPU time across the executions that recorded it, in seconds."""

    cpu_seconds_mean: float | None
    """Mean CPU time per execution that recorded it, in seconds."""

    peak_memory_bytes_min: int | None
    """Smallest peak resident memory of any execution that recorded it, in bytes."""

    peak_memory_bytes_max: int | None
    """Largest peak resident memory of any execution that recorded it, in bytes."""


@attrs.frozen(kw_only=True)
class ResourceMeasurementCollection(Page[ResourceMeasurementView]):
    """
    A page of per-execution measurements.

    Iterate it for [ResourceMeasurementView][climate_ref.results.resources.ResourceMeasurementView] items,
    or call ``to_pandas`` for a DataFrame.
    ``total_count`` is the number of matching executions, even when ``limit`` returned fewer.

    Examples
    --------
    Load up to 100 ESMValTool measurements into pandas:

    ```python
    with Database.from_config(config, read_only=True) as db:
        page = Reader(db).resources.measurements(ResourceFilter(provider_slug="esmvaltool"), limit=100)
        print(len(page), page.total_count)
        df = page.to_pandas()
    ```
    """

    def to_pandas(self) -> pd.DataFrame:
        """
        DataFrame with one row per measurement.

        The provider and diagnostic slugs are in the ``provider`` and ``diagnostic`` columns.
        The other columns keep their field names.
        Columns are emitted explicitly even when the collection is empty,
        so callers can select columns without special-casing.

        Returns
        -------
        :
            A DataFrame with one row per item on this page.
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
    How much memory and how many cores does this diagnostic need?

    A profile combines every measured execution of one diagnostic, or of one provider as a whole.
    Read the answer from these properties:

    * ``recommended_memory_bytes``: the memory to request.
    * ``recommended_cpus``: the cores to request.
    * ``confidence``: how far to trust them.
    * ``over_limit``: whether the memory limit it last ran under is already too small.

    Get profiles from [ResourcesReader.profiles][climate_ref.results.resources.ResourcesReader.profiles].

    The "p95" fields are 95th percentiles over executions.
    Every percentile is computed over the *usable samples* only.
    A usable sample is an execution that recorded a wall time, a CPU time and a peak memory figure,
    whose memory source matches ``memory_source``,
    and whose peak is attributable to it alone when ``exclusive_only`` was requested.
    This rules out a memory reading taken from a container shared with other jobs.
    Executions that recorded nothing at all are absent from both counts,
    because an unmeasured execution is not one that used no memory.

    An execution that failed is aggregated like any other, provided it recorded a measurement.
    A diagnostic that reached 40 GiB before dying is the row that most needs to reach a recommendation.

    Examples
    --------
    A diagnostic whose p95 peak is 3 GiB, last run under a 4 GiB limit:

    >>> from climate_ref.results.resources import ResourceProfile
    >>> gib = 1024**3
    >>> profile = ResourceProfile(
    ...     provider_slug="pmp",
    ...     diagnostic_slug="annual-cycle",
    ...     sample_count=12,
    ...     excluded_execution_count=0,
    ...     failed_execution_count=0,
    ...     memory_source="cgroup",
    ...     peak_memory_bytes_p50=2 * gib,
    ...     peak_memory_bytes_p95=3 * gib,
    ...     peak_memory_bytes_max=3 * gib,
    ...     wall_seconds_p95=600.0,
    ...     cpu_seconds_p95=1900.0,
    ...     parallelism_p95=3.2,
    ...     memory_limit_seen=4 * gib,
    ...     headroom_ratio=4 / 3,
    ... )

    The p95 peak is scaled by ``safety_factor`` (1.3) and rounded up to a whole GiB.
    Parallelism is rounded up to whole cores:

    >>> profile.recommended_memory_bytes // gib
    4
    >>> profile.recommended_cpus
    4

    Twelve samples is enough to trust the p95, and the 4 GiB limit is not exceeded:

    >>> profile.confidence
    'good'
    >>> profile.over_limit
    False

    Under a 2 GiB limit the headroom ratio is below 1, so the limit is too small for this diagnostic:

    >>> tight = attrs.evolve(profile, memory_limit_seen=2 * gib, headroom_ratio=2 / 3)
    >>> tight.over_limit
    True
    """

    provider_slug: str
    """Owning provider's slug."""

    diagnostic_slug: str | None
    """Owning diagnostic's slug, or ``None`` for a provider-level roll-up."""

    sample_count: int
    """Executions with a usable measurement."""

    excluded_execution_count: int
    """
    Executions that were measured but not usable.

    An execution lands here when its memory reading came from a container shared with other jobs,
    when it was missing a figure,
    or when it carried a memory source other than ``memory_source``.
    """

    failed_execution_count: int
    """
    Failed executions in the same window, measured or not.

    An execution killed for exceeding its memory limit records nothing,
    so it never appears in ``sample_count``.
    That biases every recommendation here low on exactly the diagnostics that need more memory,
    and this count is the only warning of it.
    """

    memory_source: MemorySource | None
    """
    Provenance of every peak memory figure aggregated here, or ``None`` when there are no samples.

    A cgroup reading and a rusage reading for the same execution can differ by a factor of two,
    so they are never mixed.
    When a group's executions carry more than one source,
    the source contributing the most samples wins and the rest land in ``excluded_execution_count``.
    """

    peak_memory_bytes_p50: int
    """Median peak resident memory, in bytes."""

    peak_memory_bytes_p95: int
    """95th percentile peak resident memory, in bytes."""

    peak_memory_bytes_max: int
    """Largest peak resident memory observed, in bytes."""

    wall_seconds_p95: float
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
    ``memory_limit_seen / peak_memory_bytes_p95``, or ``None`` when no limit was recorded.

    Below 1 means the diagnostic is being killed by the current container size.
    """

    safety_factor: float = 1.3
    """Multiplier applied to the p95 peak by ``recommended_memory_bytes``."""

    @property
    def recommended_memory_bytes(self) -> int:
        """
        Memory to request, in bytes.

        This is ``peak_memory_bytes_p95`` scaled by ``safety_factor``,
        rounded up to a whole GiB, and at least 1 GiB.
        """
        scaled = self.peak_memory_bytes_p95 * self.safety_factor
        return max(1, math.ceil(scaled / _BYTES_PER_GIB)) * _BYTES_PER_GIB

    @property
    def recommended_cpus(self) -> int:
        """
        Cores to request.

        This is ``parallelism_p95`` rounded up, never below one core.
        A serial diagnostic gets 1.
        """
        return max(1, math.ceil(self.parallelism_p95))

    @property
    def confidence(self) -> Confidence:
        """
        How much weight the recommendation can carry.

        The value is one of ``"none"``, ``"thin"`` and ``"good"``:

        ``none`` when there are no usable samples at all,
        ``thin`` below ten samples, where the p95 is indistinguishable from the maximum,
        and ``good`` at or above ten.
        """
        if self.sample_count == 0:
            return "none"
        if self.sample_count < _GOOD_SAMPLE_COUNT:
            return "thin"
        return "good"

    @property
    def over_limit(self) -> bool:
        """
        Whether the p95 peak has outgrown the recorded memory limit.

        ``True`` means the diagnostic needs more memory than the container it last ran in.
        ``False`` also covers the case where no limit was recorded.
        """
        return self.headroom_ratio is not None and self.headroom_ratio < 1.0


def _source_precedence(source: str) -> int:
    """Rank of a memory source in the tie-break order, with unknown sources last."""
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
    """Whether the row recorded any of wall time, CPU time or peak memory."""
    return row.wall_seconds is not None or row.cpu_seconds is not None or row.peak_memory_bytes is not None


def _is_attributable(row: ResourceMeasurementView) -> bool:
    """
    Whether the peak on a row describes the execution rather than everything sharing its container.

    Only a cgroup reading has that problem, and this is the axis ``exclusive_only`` is about.
    A ``proc_tree`` reading sweeps the execution's own processes,
    so it stays attributable however busy the worker was.

    A ``rusage`` reading is not contaminated by neighbours either, but it is a process-lifetime
    high-water mark, so on a worker that reuses processes it can carry an earlier execution's peak.
    That is a temporal weakness rather than a shared-container one,
    and it is handled by naming the source on the profile rather than by dropping the row:
    ``rusage`` is the last resort, reached only when neither other source answered,
    and dropping it would leave such a host with no profile at all rather than a high one.
    Erring high is the safe direction for a sizing recommendation.
    """
    return row.memory_source != "cgroup" or row.resources_exclusive is True


def _is_complete(row: ResourceMeasurementView) -> bool:
    """Whether the row has the positive wall time, CPU time and peak memory a sample needs."""
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
    excluded_execution_count: int,
    failed_execution_count: int,
    safety_factor: float,
) -> ResourceProfile:
    """Profile with no samples, which keeps the exclusion and failure counts."""
    return ResourceProfile(
        provider_slug=provider_slug,
        diagnostic_slug=diagnostic_slug,
        sample_count=0,
        excluded_execution_count=excluded_execution_count,
        failed_execution_count=failed_execution_count,
        memory_source=None,
        peak_memory_bytes_p50=0,
        peak_memory_bytes_p95=0,
        peak_memory_bytes_max=0,
        wall_seconds_p95=0.0,
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
    A failed execution that recorded a peak is the most informative row available,
    because a diagnostic outgrowing its container fails at the size it needs reported.
    """
    failed_execution_count = sum(1 for row in rows if row.successful is False)

    measured = [row for row in rows if _is_measured(row)]
    candidates = [row for row in measured if _is_complete(row)]
    if exclusive_only:
        candidates = [row for row in candidates if _is_attributable(row)]

    source = _dominant_source(_source_weights(candidates))
    samples = [row for row in candidates if row.memory_source == source] if source else []
    excluded_execution_count = len(measured) - len(samples)

    if not samples:
        return _empty_profile(
            provider_slug=provider_slug,
            diagnostic_slug=diagnostic_slug,
            excluded_execution_count=excluded_execution_count,
            failed_execution_count=failed_execution_count,
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
        sample_count=len(samples),
        excluded_execution_count=excluded_execution_count,
        failed_execution_count=failed_execution_count,
        memory_source=source,
        peak_memory_bytes_p50=round(_percentile(peaks, 0.5)),
        peak_memory_bytes_p95=peak_p95,
        peak_memory_bytes_max=max(peaks),
        wall_seconds_p95=_percentile(walls, 0.95),
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

    Each diagnostic's profile already comes from one memory source.
    Every measured diagnostic counts, even one measured by a different source,
    because leaving out the largest would size the worker too small.
    ``memory_source`` is the peak-driving diagnostic's source.
    """
    with_samples = [profile for profile in profiles if profile.sample_count]
    safety_factor = profiles[0].safety_factor
    excluded_execution_count = sum(profile.excluded_execution_count for profile in profiles)
    failed_execution_count = sum(profile.failed_execution_count for profile in profiles)

    if not with_samples:
        return _empty_profile(
            provider_slug=provider_slug,
            diagnostic_slug=None,
            excluded_execution_count=excluded_execution_count,
            failed_execution_count=failed_execution_count,
            safety_factor=safety_factor,
        )

    driver = max(with_samples, key=lambda profile: profile.peak_memory_bytes_p95)
    limit_seen = driver.memory_limit_seen

    peak_p95 = driver.peak_memory_bytes_p95
    headroom = limit_seen / peak_p95 if limit_seen is not None and peak_p95 > 0 else None

    return ResourceProfile(
        provider_slug=provider_slug,
        diagnostic_slug=None,
        sample_count=sum(profile.sample_count for profile in with_samples),
        excluded_execution_count=excluded_execution_count,
        failed_execution_count=failed_execution_count,
        memory_source=driver.memory_source,
        peak_memory_bytes_p50=max(profile.peak_memory_bytes_p50 for profile in with_samples),
        peak_memory_bytes_p95=peak_p95,
        peak_memory_bytes_max=max(profile.peak_memory_bytes_max for profile in with_samples),
        wall_seconds_p95=max(profile.wall_seconds_p95 for profile in with_samples),
        cpu_seconds_p95=max(profile.cpu_seconds_p95 for profile in with_samples),
        parallelism_p95=_percentile([profile.parallelism_p95 for profile in with_samples], 0.95),
        memory_limit_seen=limit_seen,
        headroom_ratio=headroom,
        safety_factor=safety_factor,
    )


class ResourcesReader:
    """
    Reads how much time, CPU and memory diagnostic executions used.

    Reach it as ``reader.resources`` rather than constructing it directly.

    Every method takes an optional
    [ResourceFilter][climate_ref.query.filters.ResourceFilter] to choose which executions count.
    Pick the method by how much to combine:

    * ``measurements`` and ``measurement`` return one record per execution.
    * ``summary`` and ``summaries`` return totals, means and extremes.
    * ``profiles`` returns percentiles and a recommended memory and core count.

    Examples
    --------
    Recommended resources for each ESMValTool provider as a whole:

    ```python
    from climate_ref.results import Reader, ResourceFilter

    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        profiles = reader.resources.profiles(ResourceFilter(provider_slug="esmvaltool"), group_by="provider")
    ```
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._session if self._session is not None else self._db.session

    def _to_view(self, row: Any) -> ResourceMeasurementView:
        """Build a view from a row of ``select_execution_resources``."""
        # The labels in ``select_execution_resources`` are the view's field names.
        return ResourceMeasurementView(**row._mapping)

    def measurements(
        self,
        filters: ResourceFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        with_count: bool = True,
    ) -> ResourceMeasurementCollection:
        """
        Query the raw per-execution measurements behind the profiles.

        Returns one record per execution that recorded a measurement or failed, oldest first.
        Use ``offset`` and ``limit`` to read a long list in pages.

        Parameters
        ----------
        filters
            Restricts the executions considered. ``None`` means no restriction.
        offset
            Rows to skip before the returned page.
        limit
            Page size, or ``None`` for every matching row.
        with_count
            Whether to run the count query. When ``False``, ``total_count`` is ``None``.

        Returns
        -------
        :
            A page of measurements, plus the total number of matching executions.

        Examples
        --------
        The ten slowest executions of one diagnostic:

        ```python
        with Database.from_config(config, read_only=True) as db:
            page = Reader(db).resources.measurements(
                ResourceFilter(provider_slug="pmp", diagnostic_slug="annual-cycle")
            )
            slowest = page.to_pandas().nlargest(10, "wall_seconds")
        ```
        """
        stmt = select_execution_resources(filters)
        total_count = count(self.session, stmt) if with_count else None

        items = tuple(self._to_view(row) for row in self.session.execute(paginate(stmt, offset, limit)).all())
        return ResourceMeasurementCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def summary(self, filters: ResourceFilter | None = None) -> ResourceSummary | None:
        """
        Total the resource usage of every execution matching ``filters``.

        This gives one set of totals and extremes for the whole selection.
        Use [summaries][climate_ref.results.resources.ResourcesReader.summaries] for one set per group.

        Parameters
        ----------
        filters
            Restricts the executions considered. ``None`` means no restriction.

        Returns
        -------
        :
            The totals, or ``None`` when no matching execution recorded a wall time.

        Examples
        --------
        Total compute spent on ILAMB:

        ```python
        with Database.from_config(config, read_only=True) as db:
            total = Reader(db).resources.summary(ResourceFilter(provider_slug="ilamb"))
            if total is not None:
                print(total.timed_execution_count, total.wall_seconds_total, total.cpu_seconds_total)
        ```
        """
        row = self.session.execute(select_resource_summary(filters)).one()
        return ResourceSummary(**row._asdict()) if row.timed_execution_count else None

    def summaries(
        self, filters: ResourceFilter | None = None, *, group_by: GroupBy = "diagnostic"
    ) -> tuple[ResourceSummary, ...]:
        """
        Total the resource usage per diagnostic or per provider.

        Groups with no execution that recorded a wall time are left out.
        Ordered by provider then diagnostic slug.

        Parameters
        ----------
        filters
            Restricts the executions considered. ``None`` means no restriction.
        group_by
            ``"diagnostic"`` for one summary per diagnostic, or ``"provider"`` for one per provider.

        Returns
        -------
        :
            One summary per group.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            for s in Reader(db).resources.summaries(group_by="provider"):
                print(s.provider_slug, s.timed_execution_count, s.wall_seconds_total)
        ```
        """
        rows = self.session.execute(select_resource_summary(filters, group_by=group_by)).all()
        return tuple(ResourceSummary(**row._asdict()) for row in rows if row.timed_execution_count)

    def measurement(self, execution_id: int) -> ResourceMeasurementView | None:
        """
        Fetch one execution's measurement by execution id.

        A retracted or superseded execution is found too,
        because the id names it directly.

        Parameters
        ----------
        execution_id
            Primary key of the execution.

        Returns
        -------
        :
            The measurement, or ``None`` when no execution has that id,
            or when the execution neither recorded a measurement nor failed.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            view = Reader(db).resources.measurement(42)
            if view is not None:
                print(view.peak_memory_bytes, view.memory_source)
        ```
        """
        # A lookup by id names the execution, so neither retraction nor promotion hides it.
        stmt = select_execution_resources(
            ResourceFilter(include_retracted=True, include_superseded=True)
        ).where(Execution.id == execution_id)
        row = self.session.execute(stmt).one_or_none()
        return self._to_view(row) if row is not None else None

    def profiles(
        self,
        filters: ResourceFilter | None = None,
        *,
        group_by: GroupBy = "diagnostic",
        exclusive_only: bool = True,
        safety_factor: float = 1.3,
    ) -> tuple[ResourceProfile, ...]:
        """
        Aggregate resource measurements into a sizing answer per diagnostic or per provider.

        Use this to decide how much memory and how many cores to request for a diagnostic.
        Read the answer from ``recommended_memory_bytes``, ``recommended_cpus``, ``confidence`` and
        ``over_limit`` on each [ResourceProfile][climate_ref.results.resources.ResourceProfile].

        A memory peak read from a container that ran four jobs at once measures all four, not one.
        So ``exclusive_only`` defaults to ``True`` and leaves those executions out.
        They are counted in ``excluded_execution_count`` rather than dropped silently.
        It costs nothing for other memory sources,
        which only look at the execution's own processes and stay accurate however busy the worker was.

        Failed executions that recorded a measurement are aggregated alongside successful ones.
        An execution that died at 40 GiB is the strongest evidence there is about what the diagnostic needs.

        Peaks from different memory sources are never mixed.
        Within each group the source contributing the most samples wins,
        the rest are counted in ``excluded_execution_count``,
        and the winner is reported as ``memory_source``.

        Parameters
        ----------
        filters
            Which executions to aggregate. ``None`` means no restriction.
        group_by
            ``diagnostic`` for one profile per diagnostic,
            or ``provider`` for the worker-sizing roll-up across a provider's diagnostics.
        exclusive_only
            Drop memory readings taken from a container the execution shared with other jobs.
        safety_factor
            Multiplier applied to the p95 peak by ``recommended_memory_bytes``.

        Returns
        -------
        :
            One profile per group, ordered by provider then diagnostic slug.

        Examples
        --------
        Memory needs of each ESMValTool diagnostic, largest first, including older diagnostic versions,
        with a 50% margin:

        ```python
        with Database.from_config(config, read_only=True) as db:
            profiles = Reader(db).resources.profiles(
                ResourceFilter(provider_slug="esmvaltool", include_superseded=True),
                safety_factor=1.5,
            )
            for p in sorted(profiles, key=lambda p: -p.recommended_memory_bytes):
                print(p.diagnostic_slug, p.recommended_memory_bytes, p.confidence, p.over_limit)
        ```

        One answer per provider, for sizing a worker that runs all of its diagnostics:

        ```python
        with Database.from_config(config, read_only=True) as db:
            pools = Reader(db).resources.profiles(group_by="provider")
        ```
        """
        filters = filters or ResourceFilter()
        # Sorting here is cheaper than SQLite's unindexed sort, and ``limit_seen`` needs this order.
        unordered = self.session.execute(select_execution_resources(filters).order_by(None)).all()
        rows = sorted(map(self._to_view, unordered), key=lambda row: (row.created_at, row.execution_id))

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
