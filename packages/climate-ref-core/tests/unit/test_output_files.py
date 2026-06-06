"""Unit tests for :mod:`climate_ref_core.output_files`."""

import pytest

from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.output_files import (
    PLACEHOLDER_OUTPUT_DIR,
    PLACEHOLDER_TEST_DATA_DIR,
    copy_execution_outputs,
    copy_output_file,
    from_placeholders,
    to_placeholders,
)


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
