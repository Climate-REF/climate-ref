"""Unit tests for :mod:`climate_ref_core.resources`."""

import json
import threading
import time

import pytest

from climate_ref_core import resources
from climate_ref_core.resources import (
    ResourceRecorder,
    ResourceUsage,
    _cgroup_directory,
    _read_cgroup_cpu_limit,
    _read_cgroup_int,
    measure_resources,
)


@pytest.fixture
def fake_cgroup(tmp_path, monkeypatch):
    """A cgroup v2 hierarchy on disk, so the tests run on macOS as well as Linux."""
    (tmp_path / "cgroup.controllers").write_text("cpu memory\n")
    (tmp_path / "memory.peak").write_text("1000\n")
    (tmp_path / "memory.current").write_text("500\n")
    (tmp_path / "memory.max").write_text("8000\n")
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    monkeypatch.setattr("climate_ref_core.resources.CGROUP_V2_MOUNT", tmp_path)
    return tmp_path


def test_usage_is_none_inside_and_populated_after():
    with measure_resources(interval=0.01, cgroup_exclusive=True) as recorder:
        assert recorder.usage is None
        time.sleep(0.02)

    usage = recorder.usage
    assert isinstance(usage, ResourceUsage)
    assert usage.wall_seconds >= 0.02
    assert usage.memory_source in {"cgroup", "proc_tree", "rusage", "unavailable"}
    assert usage.exclusive is True


def test_exclusive_is_false_unless_the_caller_declares_it():
    """
    The default is not exclusive, because a worker cannot see its siblings.

    Nothing in this process overlaps the block,
    so the old in-process-only rule would have called it exclusive
    even with three sibling workers saturating the same cgroup.
    """
    with measure_resources(interval=0.01) as recorder:
        pass

    assert recorder.usage is not None
    assert recorder.usage.exclusive is False
    assert recorder.usage.context["cgroup_exclusive_declared"] is False


def test_both_peaks_are_recorded_whichever_one_is_reported(fake_cgroup):
    """Both sampled sources survive into the record, so precedence stays a read-time decision."""
    with measure_resources(interval=0.01, cgroup_exclusive=True) as recorder:
        time.sleep(0.03)

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_source == "cgroup"
    assert usage.cgroup_peak_bytes == 500
    assert usage.proc_tree_peak_bytes is not None
    assert usage.proc_tree_peak_bytes > 0
    # The reported figure is one of the two, and the other is still available for comparison.
    assert usage.peak_memory_bytes == usage.cgroup_peak_bytes
    assert usage.context["cgroup_peak_bytes"] == usage.cgroup_peak_bytes
    assert usage.context["proc_tree_peak_bytes"] == usage.proc_tree_peak_bytes


def test_disabled_measures_nothing():
    """A disabled block runs untouched and reports as unmeasured, with no sampler started"""
    with measure_resources(interval=0.01, enabled=False) as recorder:
        assert recorder.usage is None
        time.sleep(0.02)

    assert recorder.usage is None
    assert recorder._sampler is None


def test_context_is_json_serialisable():
    with measure_resources(interval=0.01) as recorder:
        pass

    assert recorder.usage is not None
    round_tripped = json.loads(json.dumps(recorder.usage.context))
    assert round_tripped["sample_interval"] == 0.01
    assert "host" in round_tripped
    assert "cpu_count" in round_tripped
    assert round_tripped["rusage_is_process_lifetime_peak"] is True
    assert round_tripped["rusage_peak_bytes"] > 0


def test_exception_propagates_and_usage_is_recorded():
    with pytest.raises(ValueError, match="boom"):
        with measure_resources(interval=0.01) as recorder:
            raise ValueError("boom")

    assert recorder.usage is not None
    assert recorder.usage.wall_seconds >= 0.0


def test_probe_failure_degrades_rather_than_raises(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr("climate_ref_core.resources._cgroup_directory", explode)
    monkeypatch.setattr("climate_ref_core.resources._proc_tree_rss", explode)
    monkeypatch.setattr("climate_ref_core.resources._rusage_peak_bytes", explode)
    monkeypatch.setattr("climate_ref_core.resources._cpu_seconds", explode)

    with measure_resources(interval=0.01) as recorder:
        pass

    assert recorder.usage is not None
    assert recorder.usage.memory_source == "unavailable"
    assert recorder.usage.cpu_seconds is None
    assert recorder.usage.memory_limit_bytes is None


def test_memory_source_unavailable_when_every_source_fails(monkeypatch):
    monkeypatch.setattr("climate_ref_core.resources._psutil", None)
    monkeypatch.setattr("climate_ref_core.resources._cgroup_directory", lambda: None)
    monkeypatch.setattr("climate_ref_core.resources._rusage_peak_bytes", lambda: None)

    with measure_resources(interval=0.01) as recorder:
        pass

    assert recorder.usage is not None
    assert recorder.usage.memory_source == "unavailable"
    assert recorder.usage.peak_memory_bytes is None
    assert recorder.usage.context["psutil_available"] is False


def test_falls_back_to_rusage_without_psutil_or_cgroup(monkeypatch):
    monkeypatch.setattr("climate_ref_core.resources._psutil", None)
    monkeypatch.setattr("climate_ref_core.resources._cgroup_directory", lambda: None)

    with measure_resources(interval=0.01) as recorder:
        pass

    assert recorder.usage is not None
    assert recorder.usage.memory_source == "rusage"
    assert recorder.usage.peak_memory_bytes > 0


def test_cpu_seconds_tracks_a_busy_loop():
    with measure_resources(interval=0.01) as recorder:
        deadline = time.monotonic() + 0.2
        total = 0
        while time.monotonic() < deadline:
            total += 1

    usage = recorder.usage
    assert usage is not None
    assert usage.cpu_seconds is not None
    assert usage.cpu_seconds >= 0.05


def test_cpu_seconds_is_none_when_the_counters_cannot_be_read(monkeypatch):
    monkeypatch.setattr("climate_ref_core.resources._cpu_seconds", lambda: None)

    with measure_resources(interval=0.01) as recorder:
        pass

    assert recorder.usage is not None
    assert recorder.usage.cpu_seconds is None


def test_exclusive_is_false_for_overlapping_blocks():
    entered = threading.Barrier(2)
    recorders = {}

    def measure(name):
        with measure_resources(interval=0.01, cgroup_exclusive=True) as recorder:
            entered.wait(timeout=5)
        recorders[name] = recorder

    threads = [threading.Thread(target=measure, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert set(recorders) == {"a", "b"}
    assert not any(recorder.usage.exclusive for recorder in recorders.values())


def test_exclusive_is_false_for_a_block_that_was_overlapped():
    with measure_resources(interval=0.01, cgroup_exclusive=True) as outer:
        with measure_resources(interval=0.01, cgroup_exclusive=True) as inner:
            pass

    assert inner.usage is not None
    assert outer.usage is not None
    assert inner.usage.exclusive is False
    assert outer.usage.exclusive is False


def test_a_failure_while_finishing_does_not_leak_the_registry_entry(monkeypatch):
    """
    A recorder that blows up on the way out must still release its registry slot.

    The registry is process-global,
    so a leaked entry would mark every later measurement in the process as non-exclusive.
    """

    def explode(self):
        raise RuntimeError("measurement fell over")

    monkeypatch.setattr(ResourceRecorder, "_finish", explode)
    with measure_resources(interval=0.01, cgroup_exclusive=True) as broken:
        pass
    monkeypatch.undo()

    assert broken.usage is None

    # The sampler must be stopped too.
    # A leaked sampler keeps sweeping the process for the life of the process,
    # charging its CPU cost to whatever runs next.
    if broken._sampler is not None:
        broken._sampler.join(timeout=5)
        assert not broken._sampler.is_alive()

    with measure_resources(interval=0.01, cgroup_exclusive=True) as later:
        pass

    assert later.usage is not None
    assert later.usage.exclusive is True


def test_cgroup_sources_are_used(fake_cgroup):
    (fake_cgroup / "memory.peak").write_text("1000\n")

    with measure_resources(interval=0.01, cgroup_exclusive=True) as recorder:
        # The block pushes the group high-water mark, so memory.peak describes it.
        (fake_cgroup / "memory.peak").write_text("4096\n")

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_source == "cgroup"
    assert usage.peak_memory_bytes == 4096
    assert usage.memory_limit_bytes == 8000
    assert usage.cpu_limit == 4.0
    assert usage.context["cgroup_peak_at_entry"] == 1000


def test_sampled_current_is_used_when_the_peak_was_inherited(fake_cgroup):
    with measure_resources(interval=0.01, cgroup_exclusive=True) as recorder:
        time.sleep(0.05)

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_source == "cgroup"
    # memory.peak did not move, so the sampled memory.current is the honest number.
    assert usage.peak_memory_bytes == 500
    assert usage.context["samples"] >= 2


def test_an_unbaselined_peak_falls_back_to_the_sampled_series(fake_cgroup, monkeypatch):
    """
    Without an entry reading there is nothing to tell this block's memory from the group's history.

    ``memory.peak`` covers the whole life of the cgroup,
    so reporting it unbaselined would charge this block for whatever ran in the container before it.
    The sampled ``memory.current`` is a measurement of this block, so it wins instead.
    """
    (fake_cgroup / "memory.peak").write_text("100000\n")
    entry_reads = []
    real_read = resources._read_cgroup_int

    def fail_the_entry_read(path):
        # Only the entry reading of memory.peak fails; everything else answers normally.
        if path.name == "memory.peak" and not entry_reads:
            entry_reads.append(path)
            return None
        return real_read(path)

    monkeypatch.setattr(resources, "_read_cgroup_int", fail_the_entry_read)

    with measure_resources(interval=0.01, cgroup_exclusive=True) as recorder:
        time.sleep(0.05)

    usage = recorder.usage
    assert usage is not None
    assert usage.context["cgroup_peak_at_entry"] is None
    assert usage.memory_source == "cgroup"
    assert usage.peak_memory_bytes == 500
    assert usage.cgroup_peak_bytes == 500


def test_a_shared_cgroup_is_measured_over_the_process_tree(fake_cgroup):
    """
    Without a declaration the cgroup describes the container, so the process tree is reported.

    The cgroup figure is still recorded, it just does not get to be the answer.
    """
    (fake_cgroup / "memory.peak").write_text("1000\n")

    with measure_resources(interval=0.01) as recorder:
        (fake_cgroup / "memory.peak").write_text("4096\n")

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_source == "proc_tree"
    assert usage.peak_memory_bytes == usage.proc_tree_peak_bytes
    assert usage.cgroup_peak_bytes == 4096
    assert usage.exclusive is False


def test_a_shared_cgroup_still_beats_rusage_without_psutil(fake_cgroup, monkeypatch):
    """
    With no process tree to sweep, a container-wide reading is better than a lifetime mark.

    ``exclusive`` being False is what tells a reader the figure covers the container.
    """
    monkeypatch.setattr("climate_ref_core.resources._psutil", None)

    with measure_resources(interval=0.01) as recorder:
        time.sleep(0.03)

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_source == "cgroup"
    assert usage.peak_memory_bytes == 500
    assert usage.proc_tree_peak_bytes is None
    assert usage.exclusive is False


def test_an_overlapped_block_loses_the_cgroup_even_when_declared(fake_cgroup):
    """A declaration only covers other processes; an overlapping block in this one still voids it."""
    with measure_resources(interval=0.01, cgroup_exclusive=True) as outer:
        with measure_resources(interval=0.01, cgroup_exclusive=True) as inner:
            time.sleep(0.03)

    for recorder in (outer, inner):
        usage = recorder.usage
        assert usage is not None
        assert usage.exclusive is False
        assert usage.memory_source == "proc_tree"


def test_unlimited_cgroup_reads_as_none(fake_cgroup):
    (fake_cgroup / "memory.max").write_text("max\n")
    (fake_cgroup / "cpu.max").write_text("max 100000\n")

    with measure_resources(interval=0.01) as recorder:
        pass

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_limit_bytes is None
    assert usage.cpu_limit is None


def test_read_cgroup_int_handles_the_max_sentinel(tmp_path):
    unlimited = tmp_path / "memory.max"
    unlimited.write_text("max\n")
    assert _read_cgroup_int(unlimited) is None

    limited = tmp_path / "memory.current"
    limited.write_text(" 1234\n")
    assert _read_cgroup_int(limited) == 1234

    rubbish = tmp_path / "rubbish"
    rubbish.write_text("not a number\n")
    assert _read_cgroup_int(rubbish) is None

    assert _read_cgroup_int(tmp_path / "missing") is None


@pytest.mark.parametrize(
    "content, expected",
    [
        ("max 100000", None),
        ("200000 100000", 2.0),
        ("50000 100000", 0.5),
        ("100000", None),
        ("nonsense here", None),
        ("100000 0", None),
    ],
)
def test_read_cgroup_cpu_limit(tmp_path, content, expected):
    path = tmp_path / "cpu.max"
    path.write_text(content + "\n")
    assert _read_cgroup_cpu_limit(path) == expected


def test_read_cgroup_cpu_limit_missing_file(tmp_path):
    assert _read_cgroup_cpu_limit(tmp_path / "missing") is None


def test_cgroup_directory_is_none_without_a_hierarchy(tmp_path, monkeypatch):
    monkeypatch.setattr("climate_ref_core.resources.CGROUP_V2_MOUNT", tmp_path)
    assert _cgroup_directory() is None


def test_cgroup_directory_falls_back_to_the_mount_point(fake_cgroup):
    assert _cgroup_directory() == fake_cgroup
