"""Unit tests for `climate_ref.results.artifacts`."""

import pytest

from climate_ref.results.artifacts import ArtifactsReader
from climate_ref_core.logging import EXECUTION_LOG_FILENAME


class TestOutputDirectory:
    def test_resolves_under_results_root(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        assert reader.output_directory("frag") == tmp_path / "frag"

    def test_nested_fragment(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        assert reader.output_directory("a/b/c") == tmp_path / "a" / "b" / "c"

    def test_escape_raises(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        with pytest.raises(ValueError):
            reader.output_directory("../escape")

    def test_absolute_fragment_raises(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        with pytest.raises(ValueError):
            reader.output_directory("/etc/passwd")

    def test_sibling_prefix_raises(self, tmp_path):
        # Guards against a naive string-prefix containment check: `results2` starts with
        # the same characters as `results` but is a sibling directory, not a subdirectory.
        results_root = tmp_path / "results"
        reader = ArtifactsReader(results_root)
        with pytest.raises(ValueError):
            reader.output_directory("../results2/x")


class TestLogFile:
    def test_resolves_under_output_directory(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        log_file = reader.log_file("frag")
        assert log_file == tmp_path / "frag" / EXECUTION_LOG_FILENAME
        assert log_file.name == "out.log"


class TestBundle:
    def test_none_bundle_path_returns_none(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        assert reader.bundle("frag", None) is None

    def test_resolves_under_output_directory(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        assert reader.bundle("frag", "cmec.json") == tmp_path / "frag" / "cmec.json"

    def test_escape_raises(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        with pytest.raises(ValueError):
            reader.bundle("frag", "../escape")


class TestOutputFile:
    def test_resolves_under_output_directory(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        assert reader.output_file("frag", "plot.png") == tmp_path / "frag" / "plot.png"

    def test_relative_escape_raises(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        with pytest.raises(ValueError):
            reader.output_file("frag", "../escape")

    def test_absolute_filename_raises(self, tmp_path):
        reader = ArtifactsReader(tmp_path)
        with pytest.raises(ValueError):
            reader.output_file("frag", "/etc/passwd")
