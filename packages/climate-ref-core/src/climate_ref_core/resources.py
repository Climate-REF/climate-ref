"""
Measurement of the resources a block of work consumes.

A block can be measured via a context manager, :func:`measure_resources`,
and results are returned as a frozen dataclass, :class:`ResourceUsage`.

.. code-block:: python

    with measure_resources() as recorder:
        run_diagnostic()

    usage = recorder.usage
    print(usage.wall_seconds, usage.peak_memory_bytes, usage.memory_source)

Measurement never raises.
Any probe that fails degrades a single field to None,
or degrades :attr:`ResourceUsage.memory_source` to ``"unavailable"``.

Peak memory comes from a summed sweep of the process tree by default,
because that is the only source that is correct by construction:
it observes this block's processes and nothing else.
A cgroup reading covers every process in the container,
so it only describes this block when the caller declares,
via ``cgroup_exclusive``, that nothing else shares the cgroup.
The library cannot verify that on its own,
so it is preferred only when it has been declared.
The fallbacks, in order, are a cgroup reading and then ``getrusage``.
:attr:`ResourceUsage.memory_source` records which one won,
because a ``getrusage`` figure must never be silently compared against a cgroup figure.

Both sampled peaks are always recorded,
in :attr:`ResourceUsage.cgroup_peak_bytes` and :attr:`ResourceUsage.proc_tree_peak_bytes`,
whichever of them was reported.
A large divergence between the two is itself the evidence that the cgroup was shared.
"""

import os
import resource
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from attrs import frozen

# psutil is optional.
# Without it the process-tree source is unavailable and the next source in the precedence wins.
_psutil: Any
try:
    import psutil as _psutil_module

    _psutil = _psutil_module
except ImportError:  # pragma: no cover
    _psutil = None

MemorySource = Literal["cgroup", "proc_tree", "rusage", "unavailable"]
"""Provenance of a peak memory measurement."""

CGROUP_V2_MOUNT = Path("/sys/fs/cgroup")
"""Mount point of the cgroup v2 unified hierarchy."""

_SAMPLER_JOIN_TIMEOUT = 5.0
"""Seconds to wait for the sampler thread to finish before giving up on it."""

_registry_lock = threading.Lock()
_in_flight: list["ResourceRecorder"] = []


def _safe[T](probe: Callable[[], T], default: T) -> T:
    """
    Run a probe, degrading to a default rather than propagating a failure.

    Parameters
    ----------
    probe
        The measurement to attempt.
    default
        The value standing for "this could not be measured".

    Returns
    -------
    :
        The probe's value, or ``default`` if it raised.
    """
    try:
        return probe()
    except Exception:
        return default


def _read_cgroup_text(path: Path) -> str | None:
    """
    Read a cgroup control file.

    Parameters
    ----------
    path
        The control file to read.

    Returns
    -------
    :
        The stripped contents, or None if the file cannot be read.
    """
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_cgroup_int(path: Path) -> int | None:
    """
    Read a single integer from a cgroup control file.

    The literal ``max`` means unlimited and reads as None,
    as does an absent, unreadable or unparseable file.

    Parameters
    ----------
    path
        The control file to read.

    Returns
    -------
    :
        The value in bytes, or None when there is no usable number.
    """
    raw = _read_cgroup_text(path)
    if raw is None or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cgroup_cpu_limit(path: Path) -> float | None:
    """
    Read ``cpu.max`` as a number of cores.

    The file holds a quota and a period in microseconds, or ``max`` when the group is unlimited.

    Parameters
    ----------
    path
        The ``cpu.max`` control file.

    Returns
    -------
    :
        Cores available to the group, or None when it is unlimited or unreadable.
    """
    raw = _read_cgroup_text(path)
    if raw is None:
        return None

    parts = raw.split()
    if len(parts) != 2 or parts[0] == "max":  # noqa: PLR2004
        return None

    try:
        quota = float(parts[0])
        period = float(parts[1])
    except ValueError:
        return None

    if period <= 0 or quota <= 0:
        return None
    return quota / period


def _own_cgroup_path() -> str:
    """
    Path of this process within the unified hierarchy, relative to the mount point.

    Returns
    -------
    :
        The relative path, empty when ``/proc/self/cgroup`` does not name one.
    """
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return ""

    for line in lines:
        if line.startswith("0::"):
            return line[3:].strip().lstrip("/")
    return ""


def _cgroup_directory() -> Path | None:
    """
    Locate the cgroup v2 control files that apply to this process.

    Returns
    -------
    :
        The directory holding the control files, or None when this is not a cgroup v2 host.
    """
    try:
        if not (CGROUP_V2_MOUNT / "cgroup.controllers").exists():
            return None

        relative = _own_cgroup_path()
        candidate = CGROUP_V2_MOUNT / relative
        if candidate.is_dir():
            return candidate

        # A container usually sees only its own root of the hierarchy,
        # so the path named in /proc/self/cgroup does not resolve.
        return CGROUP_V2_MOUNT
    except OSError:
        return None


def _rss_bytes(process: Any) -> int:
    """
    Resident set size of one process, or zero if it has gone away.

    Parameters
    ----------
    process
        A :class:`psutil.Process`.

    Returns
    -------
    :
        The resident set size in bytes.
    """
    try:
        return int(process.memory_info().rss)
    except Exception:
        return 0


def _proc_tree_rss() -> int | None:
    """
    Sum the resident set size of this process and every descendant.

    Returns
    -------
    :
        The total in bytes, or None when psutil is missing or the sweep fails.
    """
    if _psutil is None:
        return None
    try:
        process = _psutil.Process()
        children = process.children(recursive=True)
    except Exception:
        return None
    return _rss_bytes(process) + sum(_rss_bytes(child) for child in children)


def _maxrss_bytes(who: int) -> int | None:
    """
    Peak resident set size high-water mark for ``who``, in bytes.

    ``ru_maxrss`` is reported in kilobytes on Linux and in bytes on macOS.
    The value covers the whole lifetime of the process, so it cannot be differenced.

    Parameters
    ----------
    who
        ``resource.RUSAGE_SELF`` or ``resource.RUSAGE_CHILDREN``.

    Returns
    -------
    :
        The high-water mark in bytes, or None if it cannot be read.
    """
    try:
        peak = resource.getrusage(who).ru_maxrss
    except (OSError, ValueError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def _rusage_peak_bytes() -> int | None:
    """
    Largest process-lifetime peak over this process and its reaped children.

    Returns
    -------
    :
        The high-water mark in bytes, or None if neither counter can be read.
    """
    peaks = [
        peak
        for peak in (_maxrss_bytes(resource.RUSAGE_SELF), _maxrss_bytes(resource.RUSAGE_CHILDREN))
        if peak is not None
    ]
    return max(peaks) if peaks else None


def _cpu_seconds() -> float | None:
    """
    Cumulative CPU time of this process and its reaped children, user plus system.

    Unlike ``ru_maxrss`` these counters accumulate, so two readings can be differenced.

    Returns
    -------
    :
        Seconds of CPU time, or None if the counters cannot be read.
    """
    try:
        times = os.times()
    except OSError:
        return None
    return times.user + times.system + times.children_user + times.children_system


class _PeakSampler(threading.Thread):
    """
    Daemon thread tracking the peak of the two sampled memory sources.

    It takes a sample on start, once per interval, and once more on stop,
    so a block shorter than the interval is still measured twice.
    """

    def __init__(self, cgroup: Path | None, interval: float) -> None:
        super().__init__(daemon=True, name="climate-ref-resource-sampler")
        self._cgroup = cgroup
        self._interval = interval
        self._stop_requested = threading.Event()
        self.cgroup_peak: int | None = None
        self.proc_tree_peak: int | None = None
        self.samples = 0

    def run(self) -> None:
        """Sample until stopped."""
        self._sample()
        while not self._stop_requested.wait(self._interval):
            self._sample()
        self._sample()

    def stop(self) -> None:
        """Ask the thread to take a final sample and finish."""
        self._stop_requested.set()

    def _sample(self) -> None:
        """Take one sample, keeping the running maxima."""
        try:
            self.samples += 1
            if self._cgroup is not None:
                current = _read_cgroup_int(self._cgroup / "memory.current")
                if current is not None:
                    self.cgroup_peak = max(current, self.cgroup_peak or 0)

            rss = _proc_tree_rss()
            if rss is not None:
                self.proc_tree_peak = max(rss, self.proc_tree_peak or 0)
        except Exception:
            return


@frozen
class ResourceUsage:
    """
    What one block of work cost.

    Every field except :attr:`wall_seconds` and :attr:`exclusive` is nullable,
    because each of them comes from a probe that a given host may not answer.
    """

    wall_seconds: float
    """Elapsed monotonic time."""

    cpu_seconds: float | None
    """CPU time used, self plus children, user plus system."""

    peak_memory_bytes: int | None
    """Peak memory, measured by whichever source :attr:`memory_source` names."""

    memory_source: MemorySource
    """Provenance of :attr:`peak_memory_bytes`.

    Two numbers are only comparable when they share a source.
    """

    cgroup_peak_bytes: int | None
    """Peak memory of the whole cgroup, whether or not it was the source that won.

    None when this is not a cgroup v2 host or the control files could not be read.
    """

    proc_tree_peak_bytes: int | None
    """Peak summed resident memory of this process and its descendants, whether or not it won.

    None when psutil is missing or every sweep failed.
    """

    memory_limit_bytes: int | None
    """cgroup ``memory.max`` at run time, or None when the group is unlimited."""

    cpu_limit: float | None
    """cgroup ``cpu.max`` quota expressed in cores, or None when the group is unlimited."""

    exclusive: bool
    """Whether the cgroup readings are attributable to this block alone.

    True only when the caller declared the cgroup exclusive
    *and* no other measured block overlapped this one in this process.
    The declaration is the load-bearing half:
    sibling worker processes saturating the same container are invisible from here,
    so without it a cgroup figure describes the container rather than this block.
    """

    context: dict[str, Any]
    """Host and sampler detail, JSON serialisable."""


class ResourceRecorder:
    """
    Handle yielded by :func:`measure_resources`.

    :attr:`usage` is None while the block runs and holds a :class:`ResourceUsage` once it exits.
    """

    def __init__(self, interval: float, cgroup_exclusive: bool = False) -> None:
        self.usage: ResourceUsage | None = None
        self._interval = interval
        self._cgroup_exclusive = cgroup_exclusive
        self._exclusive = cgroup_exclusive
        self._cgroup: Path | None = None
        self._sampler: _PeakSampler | None = None
        self._wall_start = 0.0
        self._cpu_start: float | None = None
        self._cgroup_peak_at_entry: int | None = None

    def _start(self) -> None:
        """Begin measuring."""
        with _registry_lock:
            for other in _in_flight:
                other._exclusive = False
            self._exclusive = self._cgroup_exclusive and not _in_flight
            _in_flight.append(self)

        self._wall_start = time.monotonic()
        try:
            self._start_probes()
        except Exception:
            return

    def _start_probes(self) -> None:
        """Take the entry readings and start the sampler."""
        self._cgroup = _cgroup_directory()
        if self._cgroup is not None:
            self._cgroup_peak_at_entry = _read_cgroup_int(self._cgroup / "memory.peak")

        self._cpu_start = _cpu_seconds()

        if self._cgroup is not None or _psutil is not None:
            self._sampler = _PeakSampler(self._cgroup, self._interval)
            self._sampler.start()

    def _finish(self) -> None:
        """Stop measuring and populate :attr:`usage`."""
        wall_seconds = max(0.0, time.monotonic() - self._wall_start)
        try:
            self._stop_sampler()
        finally:
            self._deregister()

        self.usage = self._build_usage(wall_seconds)

    def _deregister(self) -> None:
        """
        Drop this recorder from the in-flight registry.

        Safe to call more than once.
        The registry is process-global state behind ``exclusive``,
        so a leaked entry would mark every later measurement as non-exclusive
        and quietly remove it from aggregation.
        """
        with _registry_lock:
            if self in _in_flight:
                _in_flight.remove(self)

    def _stop_sampler(self) -> None:
        """
        Ask the sampler to finish, waiting only for a bounded time.

        Safe to call more than once.
        A sampler left running would keep sweeping the process for the life of the process,
        charging its CPU cost to whatever runs next.
        """
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=_SAMPLER_JOIN_TIMEOUT)

    def _build_usage(self, wall_seconds: float) -> ResourceUsage:
        """Assemble the record from the exit readings."""
        cpu_seconds = _safe(self._elapsed_cpu, None)
        sampler = self._sampler
        rusage_peak = _safe(_rusage_peak_bytes, None)
        cgroup_peak = _safe(lambda: self._cgroup_peak(sampler), None)
        proc_tree_peak = sampler.proc_tree_peak if sampler is not None else None
        unmeasured: tuple[int | None, MemorySource] = (None, "unavailable")
        peak, source = _safe(lambda: self._resolve_peak(cgroup_peak, proc_tree_peak, rusage_peak), unmeasured)

        return ResourceUsage(
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_memory_bytes=peak,
            memory_source=source,
            cgroup_peak_bytes=cgroup_peak,
            proc_tree_peak_bytes=proc_tree_peak,
            memory_limit_bytes=_safe(self._memory_limit, None),
            cpu_limit=_safe(self._cpu_limit, None),
            exclusive=self._exclusive,
            context={
                "host": _safe(_hostname, None),
                "cpu_count": _safe(os.cpu_count, None),
                "sample_interval": self._interval,
                "samples": sampler.samples if sampler is not None else 0,
                "cgroup": str(self._cgroup) if self._cgroup is not None else None,
                "cgroup_peak_at_entry": self._cgroup_peak_at_entry,
                "cgroup_exclusive_declared": self._cgroup_exclusive,
                "cgroup_peak_bytes": cgroup_peak,
                "proc_tree_peak_bytes": proc_tree_peak,
                "psutil_available": _psutil is not None,
                "rusage_peak_bytes": rusage_peak,
                "rusage_is_process_lifetime_peak": True,
            },
        )

    def _elapsed_cpu(self) -> float | None:
        """CPU seconds used inside the block, or None when the counters are unreadable."""
        end = _cpu_seconds()
        if self._cpu_start is None or end is None:
            return None
        return max(0.0, end - self._cpu_start)

    def _cgroup_peak(self, sampler: _PeakSampler | None) -> int | None:
        """
        Best cgroup figure for this block, or None when the group could not be read.

        The group high-water mark when this block raised it,
        otherwise the largest sampled ``memory.current``.
        """
        if self._cgroup is not None:
            peak = _read_cgroup_int(self._cgroup / "memory.peak")
            entry = self._cgroup_peak_at_entry
            # memory.peak is a high-water mark for the whole group,
            # so it only describes this block when the block pushed it higher.
            if peak is not None and (entry is None or peak > entry):
                return peak

        return sampler.cgroup_peak if sampler is not None else None

    def _resolve_peak(
        self, cgroup_peak: int | None, proc_tree_peak: int | None, rusage_peak: int | None
    ) -> tuple[int | None, MemorySource]:
        """
        Pick the peak to report, and name the source it came from.

        The cgroup wins only when this block had the cgroup to itself,
        because otherwise it measures the container rather than the block.
        The process tree is the default because it is correct by construction:
        it sweeps this process and its descendants and nothing else.
        A shared cgroup reading is still preferred over ``getrusage``,
        which cannot be attributed to a block at all.
        """
        if self._exclusive and cgroup_peak is not None:
            return cgroup_peak, "cgroup"

        if proc_tree_peak is not None:
            return proc_tree_peak, "proc_tree"

        if cgroup_peak is not None:
            # Names the container, not this block, which ``exclusive`` being False records.
            return cgroup_peak, "cgroup"

        if rusage_peak is not None:
            # A lifetime high-water mark rather than a measurement of this block,
            # which is what "rusage" in memory_source warns the reader about.
            return rusage_peak, "rusage"

        return None, "unavailable"

    def _memory_limit(self) -> int | None:
        """Read the cgroup ``memory.max`` limit in bytes, or None when unlimited or unavailable."""
        if self._cgroup is None:
            return None
        return _read_cgroup_int(self._cgroup / "memory.max")

    def _cpu_limit(self) -> float | None:
        """Read the cgroup ``cpu.max`` quota in cores, or None when unlimited or unavailable."""
        if self._cgroup is None:
            return None
        return _read_cgroup_cpu_limit(self._cgroup / "cpu.max")


def _hostname() -> str | None:
    """Name of the host the block ran on, or None if it cannot be determined."""
    try:
        return socket.gethostname()
    except OSError:
        return None


@contextmanager
def measure_resources(
    *, interval: float = 0.5, enabled: bool = True, cgroup_exclusive: bool = False
) -> Iterator[ResourceRecorder]:
    """
    Measure wall time, CPU time and peak memory of everything done in the block.

    The yielded recorder exposes a single attribute,
    ``usage``, which is None inside the block and a :class:`ResourceUsage` after it exits.

    A sampling failure degrades individual fields to None rather than failing the execution.
    An exception raised inside the block still propagates, with ``usage`` populated.

    Parameters
    ----------
    interval
        Seconds between memory samples.
    enabled
        Whether to measure at all.

        When False the block runs untouched and ``usage`` stays None,
        which every consumer already reads as unmeasured.
        No sampler thread is started and no cgroup file is read.
    cgroup_exclusive
        Whether the caller can promise that nothing else shares this process's cgroup
        while the block runs.

        Only a caller that owns the concurrency knows this,
        which is why it is declared rather than probed:
        sibling worker processes in the same container are invisible from inside one of them.
        It defaults to False, under which a cgroup figure is reported
        only when the process tree cannot be swept,
        and is marked as non-exclusive so aggregation excludes it.

    Yields
    ------
    :
        The recorder holding the result.
    """
    recorder = ResourceRecorder(interval, cgroup_exclusive)
    if not enabled:
        yield recorder
        return

    recorder._start()
    try:
        yield recorder
    finally:
        try:
            recorder._finish()
        except Exception:
            recorder.usage = None
        finally:
            _safe(recorder._stop_sampler, None)
            recorder._deregister()
