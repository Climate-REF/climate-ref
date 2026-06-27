"""Unit tests for :mod:`climate_ref_core.regression.capture`."""

import json
from pathlib import Path

import pytest

from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.output_files import (
    PLACEHOLDER_OUTPUT_DIR,
    PLACEHOLDER_SOFTWARE_ROOT_DIR,
    PLACEHOLDER_TEST_DATA_DIR,
    PlaceholderMap,
)
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.regression.capture import (
    _COMMITTED_FLOAT_JSON_KWARGS,
    _UNROUNDED_COMMITTED_FILES,
    _contains_float,
    build_native_snapshot,
    capture_execution,
    materialise_native,
    write_committed_bundle,
)
from climate_ref_core.regression.manifest import COMMITTED_BUNDLE_FILES, NativeEntry, sha256_file
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
        source,
        regression_dir,
        placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir),
    )

    assert set(digests) == {"series.json", "diagnostic.json", "output.json"}
    written = (regression_dir / "diagnostic.json").read_text()
    assert str(output_dir) not in written
    assert PLACEHOLDER_OUTPUT_DIR in written
    assert PLACEHOLDER_TEST_DATA_DIR in written
    # Digest is over the sanitised bytes as they sit on disk.
    assert digests["diagnostic.json"] == sha256_file(regression_dir / "diagnostic.json")


def test_write_committed_bundle_rejects_unbound_placeholders(tmp_path):
    output_dir = (tmp_path / "scratch" / "frag").resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    source = _seed_execution(tmp_path / "scratch", "frag", output_dir=output_dir, test_data_dir=test_data_dir)
    regression_dir = tmp_path / "regression"

    with pytest.raises(ValueError, match="with_output"):
        write_committed_bundle(
            source, regression_dir, placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir)
        )


def _sig_figs(value: float) -> int:
    """Count the significant figures in ``value``'s shortest round-trip repr."""
    # ``repr`` gives Python's shortest decimal that round-trips to the same float,
    # so it reflects the value's true precision rather than padding it out.
    text = repr(value)
    mantissa = text.lstrip("-").split("e")[0]
    digits = mantissa.replace(".", "").lstrip("0").rstrip("0")
    return len(digits) if digits else 1


def _assert_floats_rounded(obj, max_sig_figs=7):
    """Recursively assert every float in ``obj`` has at most ``max_sig_figs`` figures."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        assert _sig_figs(obj) <= max_sig_figs, f"{obj!r} exceeds {max_sig_figs} sig figs"
    elif isinstance(obj, dict):
        for value in obj.values():
            _assert_floats_rounded(value, max_sig_figs)
    elif isinstance(obj, list):
        for item in obj:
            _assert_floats_rounded(item, max_sig_figs)


def test_write_committed_bundle_rounds_floats(tmp_path):
    output_dir = (tmp_path / "scratch" / "frag").resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    source = tmp_path / "scratch" / "frag"
    source.mkdir(parents=True)
    # Full-precision floats that round to fewer sig figs at write time.
    source_diag = {"PROVENANCE": {"score": 1.843240715970751, "rmse": 2.813496471229112}}
    (source / "diagnostic.json").write_text(json.dumps(source_diag))
    source_series = [{"dimensions": {"region": "global"}, "values": [1.761333624017425, 9.87654321]}]
    (source / "series.json").write_text(json.dumps(source_series))
    regression_dir = tmp_path / "regression"

    write_committed_bundle(
        source,
        regression_dir,
        placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir),
    )

    diag = json.loads((regression_dir / "diagnostic.json").read_text())
    series = json.loads((regression_dir / "series.json").read_text())
    _assert_floats_rounded(diag)
    _assert_floats_rounded(series)
    # The actual rounded values, not merely "<= 7 figures".
    assert diag["PROVENANCE"]["score"] == 1.843241
    assert series[0]["values"] == [1.761334, 9.876543]


def test_committed_float_classification_is_exhaustive():
    assert set(_COMMITTED_FLOAT_JSON_KWARGS) | _UNROUNDED_COMMITTED_FILES == set(COMMITTED_BUNDLE_FILES)
    assert set(_COMMITTED_FLOAT_JSON_KWARGS).isdisjoint(_UNROUNDED_COMMITTED_FILES)


def test_output_json_is_float_free_by_construction(tmp_path):
    # Codifies the invariant the output.json exclusion relies on: a representative output bundle,
    # serialised the way REF writes it natively, carries no float leaves.
    output = CMECOutput.model_validate(CMECOutput.create_template())
    output.update(
        "data",
        short_name="example",
        dict_content={
            "filename": "data/example.nc",
            "long_name": "Example output",
            "description": "An example data file",
        },
    )
    output.update(
        "html",
        short_name="index",
        dict_content={
            "filename": "index.html",
            "long_name": "Index page",
            "description": "The landing page",
        },
    )
    json_path = tmp_path / "output.json"
    output.dump_to_json(json_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))

    assert _contains_float(loaded) is False


def test_write_committed_bundle_leaves_output_json_bytes_unchanged(tmp_path):
    # output.json must pass through write_committed_bundle byte-for-byte (no key reorder / rewrite),
    # while the float-bearing artefacts are rounded.
    output_dir = (tmp_path / "scratch" / "frag").resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    source = tmp_path / "scratch" / "frag"
    source.mkdir(parents=True)
    # output.json written the native (Pydantic) way: indent=2, keys NOT sorted, no floats.
    CMECOutput.model_validate(CMECOutput.create_template()).dump_to_json(source / "output.json")
    source_output_bytes = (source / "output.json").read_bytes()
    # Float-bearing artefacts that rounding must rewrite.
    (source / "diagnostic.json").write_text(json.dumps({"PROVENANCE": {"score": 1.843240715970751}}))
    (source / "series.json").write_text(
        json.dumps([{"dimensions": {"region": "global"}, "values": [1.761333624017425]}])
    )
    regression_dir = tmp_path / "regression"

    write_committed_bundle(
        source,
        regression_dir,
        placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir),
    )

    # output.json is byte-identical to the copied source: not re-dumped, keys not reordered.
    assert (regression_dir / "output.json").read_bytes() == source_output_bytes
    # The float-bearing artefacts were rounded (proving the bundle was processed, not skipped).
    assert json.loads((regression_dir / "diagnostic.json").read_text())["PROVENANCE"]["score"] == 1.843241
    assert json.loads((regression_dir / "series.json").read_text())[0]["values"] == [1.761334]


def _leaky_provenance(software_root: Path, output_dir: Path) -> dict:
    """A CMEC provenance block: host/user fields to redact + absolute paths to placeholder."""
    return {
        "commandLine": (
            f"{software_root}/conda/pmp/bin/mean_climate_driver.py "
            f"-p {software_root}/params/pmp_param.py "
            f"--test_data_path {output_dir} --cmec"
        ),
        "date": "2026-06-24 12:30:54",
        "userId": "jared",
        "platform": {"Name": "gus", "OS": "Linux", "Version": "6.12.88+deb13-amd64"},
    }


def test_write_committed_bundle_redacts_and_placeholders_provenance(tmp_path):
    output_dir = (tmp_path / "scratch" / "frag").resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    software_root = (tmp_path / "software").resolve()
    source = tmp_path / "scratch" / "frag"
    source.mkdir(parents=True)
    # Metric bundle: leaky uppercase PROVENANCE + a float so rounding also runs.
    (source / "diagnostic.json").write_text(
        json.dumps(
            {"PROVENANCE": _leaky_provenance(software_root, output_dir), "RESULTS": {"score": 1.2345678901}}
        )
    )
    # Output bundle: leaky lowercase provenance, float-free, native indent=2 layout.
    output_obj = CMECOutput.create_template()
    output_obj["provenance"].update(_leaky_provenance(software_root, output_dir))
    (source / "output.json").write_text(json.dumps(output_obj, indent=2))
    (source / "series.json").write_text(json.dumps([]))
    regression_dir = tmp_path / "regression"

    digests = write_committed_bundle(
        source,
        regression_dir,
        placeholders=PlaceholderMap.for_baseline(
            test_data_dir=test_data_dir, software_root_dir=software_root
        ).with_output(output_dir),
    )

    for filename in ("diagnostic.json", "output.json"):
        text = (regression_dir / filename).read_text()
        # date and userId are redacted as structured fields...
        assert '"userId": "<USER>"' in text
        assert '"date": "<DATE>"' in text
        # ...the command line is kept but made portable via path placeholders (not nuked)...
        assert "<COMMAND_LINE>" not in text
        assert "mean_climate_driver.py" in text
        assert PLACEHOLDER_SOFTWARE_ROOT_DIR in text
        assert PLACEHOLDER_OUTPUT_DIR in text
        # ...and no personal / host / absolute-path / timestamp data survives.
        assert "jared" not in text
        assert "gus" not in text  # hostname (platform.Name) redacted
        assert "6.12.88+deb13-amd64" not in text  # kernel version (platform.Version) redacted
        assert str(software_root) not in text
        assert str(output_dir) not in text
        assert "2026-06-24 12:30:54" not in text
        # Digest is taken over the sanitised bytes exactly as they sit on disk.
        assert digests[filename] == sha256_file(regression_dir / filename)

    # Structured redaction survives float-rounding in the metric bundle.
    diag = json.loads((regression_dir / "diagnostic.json").read_text())
    assert diag["PROVENANCE"]["userId"] == "<USER>"
    assert diag["PROVENANCE"]["date"] == "<DATE>"
    # Host fields redacted; coarse OS kept as portable context.
    assert diag["PROVENANCE"]["platform"]["Name"] == "<HOSTNAME>"
    assert diag["PROVENANCE"]["platform"]["Version"] == "<HOST_VERSION>"
    assert diag["PROVENANCE"]["platform"]["OS"] == "Linux"
    assert diag["RESULTS"]["score"] == 1.234568

    # output.json is re-dumped byte-faithfully: provenance key order is preserved.
    out = json.loads((regression_dir / "output.json").read_text())
    assert list(out["provenance"].keys()) == list(output_obj["provenance"].keys())


def test_redaction_is_noop_without_provenance_fields(tmp_path):
    # A bundle with no host/user provenance fields must pass through untouched,
    # so the committed digest stays byte-stable for already-portable artefacts.
    output_dir = (tmp_path / "scratch" / "frag").resolve()
    test_data_dir = (tmp_path / "test-data").resolve()
    source = tmp_path / "scratch" / "frag"
    source.mkdir(parents=True)
    template = CMECOutput.create_template()
    (source / "output.json").write_text(json.dumps(template, indent=2))
    source_output_bytes = (source / "output.json").read_bytes()
    (source / "diagnostic.json").write_text(json.dumps({"RESULTS": {"score": 1.0}}))
    (source / "series.json").write_text(json.dumps([]))
    regression_dir = tmp_path / "regression"

    write_committed_bundle(
        source,
        regression_dir,
        placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir),
    )

    assert (regression_dir / "output.json").read_bytes() == source_output_bytes


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
        placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir),
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
        placeholders=PlaceholderMap.for_baseline(test_data_dir=test_data_dir).with_output(output_dir),
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
            placeholders=PlaceholderMap.for_baseline(test_data_dir=tmp_path / "td").with_output(
                tmp_path / "out"
            ),
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


@pytest.mark.parametrize(
    "relpath",
    ["../escape.nc", "../../etc/passwd", "/abs/path.nc", "sub/../../escape.nc"],
)
def test_materialise_native_rejects_path_traversal(tmp_path, relpath):
    store = LocalFilesystemStore(root=tmp_path / "store")
    native = {relpath: NativeEntry(sha256="0" * 64, size=1)}
    with pytest.raises(ValueError, match=r"Unsafe native path"):
        materialise_native(native, store, tmp_path / "dest")
