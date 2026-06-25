"""Unit tests for :mod:`climate_ref_core.output_files`."""

import pytest

from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.output_files import (
    PLACEHOLDER_OUTPUT_DIR,
    PLACEHOLDER_SOFTWARE_ROOT_DIR,
    PLACEHOLDER_TEST_DATA_DIR,
    copy_execution_outputs,
    copy_output_file,
    from_placeholders,
    to_placeholders,
)
from climate_ref_core.pycmec.output import CMECOutput


def test_sanitise_and_expand_round_trip(tmp_path):
    output_dir = tmp_path / "scratch" / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()

    original = f"path={output_dir}\ndata={test_data_dir}\n"
    (directory / "diagnostic.json").write_text(original)

    to_placeholders(directory, output_dir=output_dir, test_data_dir=test_data_dir)
    sanitised = (directory / "diagnostic.json").read_text()
    assert str(output_dir) not in sanitised
    assert str(test_data_dir) not in sanitised
    assert PLACEHOLDER_OUTPUT_DIR in sanitised
    assert PLACEHOLDER_TEST_DATA_DIR in sanitised

    from_placeholders(directory, output_dir=output_dir, test_data_dir=test_data_dir)
    assert (directory / "diagnostic.json").read_text() == original


def test_sanitise_software_root_round_trip(tmp_path):
    output_dir = tmp_path / "scratch" / "output"
    test_data_dir = tmp_path / "test-data"
    software_root_dir = tmp_path / "software"
    directory = tmp_path / "regression"
    directory.mkdir()

    # A provenance command line referencing the software root and the output dir.
    original = f"cmd={software_root_dir}/conda/bin/driver.py --out {output_dir}\n"
    (directory / "diagnostic.json").write_text(original)

    to_placeholders(
        directory,
        output_dir=output_dir,
        test_data_dir=test_data_dir,
        software_root_dir=software_root_dir,
    )
    sanitised = (directory / "diagnostic.json").read_text()
    assert str(software_root_dir) not in sanitised
    assert PLACEHOLDER_SOFTWARE_ROOT_DIR in sanitised
    assert PLACEHOLDER_OUTPUT_DIR in sanitised

    from_placeholders(
        directory,
        output_dir=output_dir,
        test_data_dir=test_data_dir,
        software_root_dir=software_root_dir,
    )
    assert (directory / "diagnostic.json").read_text() == original


def test_software_root_substitution_is_opt_in(tmp_path):
    # Without software_root_dir, no <SOFTWARE_ROOT_DIR> substitution occurs (back-compatible default).
    software_root_dir = tmp_path / "software"
    directory = tmp_path / "regression"
    directory.mkdir()
    original = f"cmd={software_root_dir}/conda/bin/driver.py\n"
    (directory / "diagnostic.json").write_text(original)

    to_placeholders(directory, output_dir=tmp_path / "out", test_data_dir=tmp_path / "td")

    assert (directory / "diagnostic.json").read_text() == original


def test_sanitise_only_touches_text_globs(tmp_path):
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()

    binary = f"{output_dir}".encode()
    (directory / "data.nc").write_bytes(binary)
    (directory / "meta.json").write_text(str(output_dir))

    to_placeholders(directory, output_dir=output_dir, test_data_dir=test_data_dir)

    # Binary file is untouched; JSON is rewritten.
    assert (directory / "data.nc").read_bytes() == binary
    assert (directory / "meta.json").read_text() == PLACEHOLDER_OUTPUT_DIR


def test_sanitise_longest_match_first(tmp_path):
    # test_data_dir is a prefix of output_dir; the longer path must win.
    test_data_dir = tmp_path / "base"
    output_dir = tmp_path / "base" / "scratch"
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / "a.json").write_text(str(output_dir))

    to_placeholders(directory, output_dir=output_dir, test_data_dir=test_data_dir)

    assert (directory / "a.json").read_text() == PLACEHOLDER_OUTPUT_DIR


@pytest.mark.parametrize("suffix", ("json", "txt", "yaml", "yml", "html", "xml"))
def test_sanitise_covers_all_text_globs(suffix, tmp_path):
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / f"artefact.{suffix}").write_text(str(output_dir))

    to_placeholders(directory, output_dir=output_dir, test_data_dir=test_data_dir)

    assert (directory / f"artefact.{suffix}").read_text() == PLACEHOLDER_OUTPUT_DIR


def test_sanitise_respects_custom_globs(tmp_path):
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()
    (directory / "config.cfg").write_text(str(output_dir))
    (directory / "skip.json").write_text(str(output_dir))

    to_placeholders(
        directory,
        output_dir=output_dir,
        test_data_dir=test_data_dir,
        globs=("*.cfg",),
    )

    # Only the custom glob is rewritten; the default JSON glob is left untouched.
    assert (directory / "config.cfg").read_text() == PLACEHOLDER_OUTPUT_DIR
    assert (directory / "skip.json").read_text() == str(output_dir)


def test_sanitise_leaves_unmatched_content_untouched(tmp_path):
    # A text file that contains neither path must be left byte-for-byte identical
    # (exercises the "no rewrite" branch that skips the write).
    output_dir = tmp_path / "output"
    test_data_dir = tmp_path / "test-data"
    directory = tmp_path / "regression"
    directory.mkdir()
    original = "nothing to replace here\n"
    target = directory / "untouched.json"
    target.write_text(original)
    before = target.stat().st_mtime_ns

    to_placeholders(directory, output_dir=output_dir, test_data_dir=test_data_dir)

    assert target.read_text() == original
    assert target.stat().st_mtime_ns == before


@pytest.mark.parametrize("filename", ("bundle.zip", "nested/bundle.zip"))
def test_copy_output_file_returns_relpath(filename, tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    src = scratch / fragment / filename
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()

    rel = copy_output_file(scratch, results, fragment, filename)

    assert (results / fragment / filename).exists()
    assert rel.as_posix() == filename


def test_copy_output_file_missing_raises(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    with pytest.raises(FileNotFoundError, match=r"Could not find missing\.json"):
        copy_output_file(scratch, results, "frag", "missing.json")


def test_copy_output_file_accepts_absolute_filename(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    src = scratch / fragment / "bundle.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.touch()

    # An absolute path under scratch/fragment is normalised back to a relative path.
    rel = copy_output_file(scratch, results, fragment, src)

    assert rel.as_posix() == "bundle.json"
    assert (results / fragment / "bundle.json").exists()


def test_copy_output_file_rejects_identical_directories(tmp_path):
    same = (tmp_path / "shared").resolve()
    with pytest.raises(ValueError, match="must differ"):
        copy_output_file(same, same, "frag", "bundle.json")


class _FakeResult:
    def __init__(self, metric_bundle_filename, output_bundle_filename=None, series_filename=None):
        self.metric_bundle_filename = metric_bundle_filename
        self.output_bundle_filename = output_bundle_filename
        self.series_filename = series_filename


def _seed(scratch, fragment, *names):
    for name in names:
        path = scratch / fragment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_copy_execution_outputs_curated_set(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, EXECUTION_LOG_FILENAME, "bundle.json", "series.json")

    result = _FakeResult(metric_bundle_filename="bundle.json", series_filename="series.json")
    copied = copy_execution_outputs(scratch, results, fragment, result, include_log=True)

    names = {p.as_posix() for p in copied}
    assert names == {EXECUTION_LOG_FILENAME, "bundle.json", "series.json"}
    for name in names:
        assert (results / fragment / name).exists()


def test_copy_execution_outputs_excludes_log_by_default(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json")

    result = _FakeResult(metric_bundle_filename="bundle.json")
    copied = copy_execution_outputs(scratch, results, fragment, result)

    assert [p.as_posix() for p in copied] == ["bundle.json"]
    assert not (results / fragment / EXECUTION_LOG_FILENAME).exists()


def test_copy_execution_outputs_requires_metric_bundle(tmp_path):
    result = _FakeResult(metric_bundle_filename=None)
    with pytest.raises(ValueError, match="without a metric bundle"):
        copy_execution_outputs(tmp_path / "scratch", tmp_path / "results", "frag", result)


def _write_output_bundle(scratch, fragment, bundle_filename, referenced):
    """Seed a CMEC output bundle plus the plots/data/html files it references."""
    cmec_output = CMECOutput(**CMECOutput.create_template())
    for attr, short_name, filename in referenced:
        cmec_output.update(
            attr,
            short_name=short_name,
            dict_content={"long_name": short_name, "filename": filename, "description": ""},
        )
        _seed(scratch, fragment, filename)
    cmec_output.dump_to_json(scratch / fragment / bundle_filename)


def test_copy_execution_outputs_includes_output_bundle_and_referenced_files(tmp_path):
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json")
    _write_output_bundle(
        scratch,
        fragment,
        "output.json",
        referenced=[
            ("plots", "fig1", "fig_1.png"),
            ("plots", "fig2", "nested/fig_2.png"),
            ("html", "index", "index.html"),
        ],
    )

    result = _FakeResult(metric_bundle_filename="bundle.json", output_bundle_filename="output.json")
    copied = copy_execution_outputs(scratch, results, fragment, result)

    names = {p.as_posix() for p in copied}
    assert names == {"bundle.json", "output.json", "fig_1.png", "nested/fig_2.png", "index.html"}
    for name in names:
        assert (results / fragment / name).exists()


def test_copy_execution_outputs_full_curated_set(tmp_path):
    # Exercises every branch together: log + metric + output bundle (+ referenced) + series.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, EXECUTION_LOG_FILENAME, "bundle.json", "series.json")
    _write_output_bundle(scratch, fragment, "output.json", referenced=[("data", "d1", "data_1.nc")])

    result = _FakeResult(
        metric_bundle_filename="bundle.json",
        output_bundle_filename="output.json",
        series_filename="series.json",
    )
    copied = copy_execution_outputs(scratch, results, fragment, result, include_log=True)

    names = {p.as_posix() for p in copied}
    assert names == {
        EXECUTION_LOG_FILENAME,
        "bundle.json",
        "output.json",
        "data_1.nc",
        "series.json",
    }
    for name in names:
        assert (results / fragment / name).exists()


def test_copy_execution_outputs_empty_output_bundle(tmp_path):
    # An output bundle that references no files still copies the bundle itself.
    scratch = (tmp_path / "scratch").resolve()
    results = (tmp_path / "results").resolve()
    fragment = "frag"
    _seed(scratch, fragment, "bundle.json")
    _write_output_bundle(scratch, fragment, "output.json", referenced=[])

    result = _FakeResult(metric_bundle_filename="bundle.json", output_bundle_filename="output.json")
    copied = copy_execution_outputs(scratch, results, fragment, result)

    assert {p.as_posix() for p in copied} == {"bundle.json", "output.json"}
