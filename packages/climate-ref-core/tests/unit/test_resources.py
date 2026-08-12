"""Unit tests for :mod:`climate_ref_core.resources`."""

import json
import threading
import time

import pytest

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
    with measure_resources(interval=0.01) as recorder:
        assert recorder.usage is None
        time.sleep(0.02)

    usage = recorder.usage
    assert isinstance(usage, ResourceUsage)
    assert usage.wall_seconds >= 0.02
    assert usage.memory_source in {"cgroup", "proc_tree", "rusage", "unavailable"}
    assert usage.exclusive is True


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
        with measure_resources(interval=0.01) as recorder:
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
    with measure_resources(interval=0.01) as outer:
        with measure_resources(interval=0.01) as inner:
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
    with measure_resources(interval=0.01) as broken:
        pass
    monkeypatch.undo()

    assert broken.usage is None

    # The sampler must be stopped too.
    # A leaked sampler keeps sweeping the process for the life of the process,
    # charging its CPU cost to whatever runs next.
    if broken._sampler is not None:
        broken._sampler.join(timeout=5)
        assert not broken._sampler.is_alive()

    with measure_resources(interval=0.01) as later:
        pass

    assert later.usage is not None
    assert later.usage.exclusive is True


def test_cgroup_sources_are_used(fake_cgroup):
    (fake_cgroup / "memory.peak").write_text("1000\n")

    with measure_resources(interval=0.01) as recorder:
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
    with measure_resources(interval=0.01) as recorder:
        time.sleep(0.05)

    usage = recorder.usage
    assert usage is not None
    assert usage.memory_source == "cgroup"
    # memory.peak did not move, so the sampled memory.current is the honest number.
    assert usage.peak_memory_bytes == 500
    assert usage.context["samples"] >= 2


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
