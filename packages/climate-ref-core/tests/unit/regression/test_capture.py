"""Unit tests for :mod:`climate_ref_core.regression.capture`."""

import json
from pathlib import Path

import pytest

from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.output_files import PLACEHOLDER_OUTPUT_DIR, PLACEHOLDER_TEST_DATA_DIR
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.regression.capture import (
    build_native_snapshot,
    capture_execution,
    materialise_native,
    write_committed_bundle,
)
from climate_ref_core.regression.manifest import sha256_file
from climate_ref_core.regression.store import LocalFilesystemStore


class _FakeResult:
    def __init__(self, metric_bundle_filename, output_bundle_filename=None, series_filename=None):
        self.metric_bundle_filename = metric_bundle_filename
        self.output_bundle_filename = output_bundle_filename
        self.series_filename = series_filename


def _seed_execution(scratch, fragment, *, output_dir, test_data_dir):
    """Write a minimal committed bundle + log into scratch/fragment."""
    base = scratch / fragment
    base.mkdir(parents=True, exist_ok=True)
    (base / EXECUTION_LOG_FILENAME).write_text("ran ok\n")
    # diagnostic.json (metric bundle) embeds an absolute path -> must be sanitised.
    (base / "diagnostic.json").write_text(
        json.dumps({"provenance": {"path": str(output_dir), "data": str(test_data_dir)}})
    )
    # output.json is a valid (empty) CMEC output bundle so copy can introspect it.
    (base / "output.json").write_text(json.dumps(CMECOutput.create_template()))
    (base / "series.json").write_text(json.dumps([{"v": str(output_dir)}]))
    return base


def test_write_committed_bundle_sanitises_and_digests(tmp_path):
    output_dir = (tmp_path / "scratch" / "frag").resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    source = _seed_execution(tmp_path / "scratch", "frag", output_dir=output_dir, test_data_dir=test_data_dir)
    regression_dir = tmp_path / "regression"

    digests = write_committed_bundle(
        source, regression_dir, output_dir=output_dir, test_data_dir=test_data_dir
    )

    assert set(digests) == {"series.json", "diagnostic.json", "output.json"}
    written = (regression_dir / "diagnostic.json").read_text()
    assert str(output_dir) not in written
    assert PLACEHOLDER_OUTPUT_DIR in written
    assert PLACEHOLDER_TEST_DATA_DIR in written
    # Digest is over the sanitised bytes as they sit on disk.
    assert digests["diagnostic.json"] == sha256_file(regression_dir / "diagnostic.json")


def test_build_native_snapshot_digests_relpaths(tmp_path):
    base = tmp_path / "results" / "frag"
    base.mkdir(parents=True)
    (base / "a.nc").write_bytes(b"data-a")
    (base / "sub").mkdir()
    (base / "sub" / "b.png").write_bytes(b"data-b")

    snapshot = build_native_snapshot(base, [Path("a.nc"), Path("sub/b.png")])

    assert set(snapshot) == {"a.nc", "sub/b.png"}
    assert snapshot["a.nc"].size == len(b"data-a")
    assert snapshot["a.nc"].sha256 == sha256_file(base / "a.nc")


def test_capture_execution_end_to_end(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    output_dir = (scratch / fragment).resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    _seed_execution(scratch, fragment, output_dir=output_dir, test_data_dir=test_data_dir)
    regression_dir = tmp_path / "regression"

    result = _FakeResult(
        metric_bundle_filename="diagnostic.json",
        output_bundle_filename="output.json",
        series_filename="series.json",
    )

    committed, native = capture_execution(
        scratch,
        results,
        fragment,
        result,
        regression_dir=regression_dir,
        output_dir=output_dir,
        test_data_dir=test_data_dir,
    )

    # Committed bundle is the three CMEC artefacts.
    assert set(committed) == {"series.json", "diagnostic.json", "output.json"}
    # Native snapshot is exactly what copy_execution_outputs persists in production:
    # the 3 bundles, without the execution log (the production default).
    assert set(native) == {"diagnostic.json", "output.json", "series.json"}
    # Persisted files actually landed in the results directory.
    for relpath in native:
        assert (results / fragment / relpath).exists()


def test_capture_execution_include_log(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    output_dir = (scratch / fragment).resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    _seed_execution(scratch, fragment, output_dir=output_dir, test_data_dir=test_data_dir)

    result = _FakeResult(
        metric_bundle_filename="diagnostic.json",
        output_bundle_filename="output.json",
        series_filename="series.json",
    )

    _, native = capture_execution(
        scratch,
        results,
        fragment,
        result,
        regression_dir=tmp_path / "regression",
        output_dir=output_dir,
        test_data_dir=test_data_dir,
        include_log=True,
    )

    assert set(native) == {EXECUTION_LOG_FILENAME, "diagnostic.json", "output.json", "series.json"}


def test_capture_execution_requires_metric_bundle(tmp_path):
    result = _FakeResult(metric_bundle_filename=None)
    with pytest.raises(ValueError, match="without a metric bundle"):
        capture_execution(
            tmp_path / "scratch",
            tmp_path / "results",
            "frag",
            result,
            regression_dir=tmp_path / "regression",
            output_dir=tmp_path / "out",
            test_data_dir=tmp_path / "td",
        )


def test_materialise_native_round_trip(tmp_path):
    # Put two blobs into a local store, snapshot them, then materialise.
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.nc").write_bytes(b"alpha")
    (src / "b.png").write_bytes(b"beta")
    store = LocalFilesystemStore(root=tmp_path / "store")

    snapshot = build_native_snapshot(src, [Path("a.nc"), Path("b.png")])
    for relpath, entry in snapshot.items():
        digest = store.put(src / relpath)
        assert digest == entry.sha256

    dest = tmp_path / "dest"
    materialise_native(snapshot, store, dest)

    assert (dest / "a.nc").read_bytes() == b"alpha"
    assert (dest / "b.png").read_bytes() == b"beta"
