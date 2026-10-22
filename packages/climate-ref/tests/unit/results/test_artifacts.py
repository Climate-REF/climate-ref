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


class TestSymlinkContainment:
    @pytest.fixture
    def layout(self, tmp_path):
        results_root = tmp_path / "results"
        (results_root / "frag").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        return results_root, outside

    def test_symlinked_directory_pointing_outside_raises(self, layout):
        results_root, outside = layout
        (results_root / "link").symlink_to(outside)
        reader = ArtifactsReader(results_root)

        with pytest.raises(ValueError, match="symlink"):
            reader.output_directory("link")
        with pytest.raises(ValueError, match="symlink"):
            reader.output_file("link", "secret.txt")
        with pytest.raises(ValueError, match="symlink"):
            reader.log_file("link")

    def test_symlinked_file_pointing_outside_raises(self, layout):
        results_root, outside = layout
        (results_root / "frag" / "plot.png").symlink_to(outside / "secret.txt")
        reader = ArtifactsReader(results_root)

        with pytest.raises(ValueError, match="symlink"):
            reader.output_file("frag", "plot.png")

    def test_symlinked_bundle_pointing_outside_raises(self, layout):
        results_root, outside = layout
        (results_root / "frag" / "cmec.json").symlink_to(outside / "secret.txt")
        reader = ArtifactsReader(results_root)

        with pytest.raises(ValueError, match="symlink"):
            reader.bundle("frag", "cmec.json")

    def test_symlinked_log_pointing_outside_raises(self, layout):
        results_root, outside = layout
        (results_root / "frag" / EXECUTION_LOG_FILENAME).symlink_to(outside / "secret.txt")
        reader = ArtifactsReader(results_root)

        with pytest.raises(ValueError, match="symlink"):
            reader.log_file("frag")

    def test_symlink_within_root_is_allowed(self, layout):
        results_root, _ = layout
        (results_root / "frag" / "plot.png").write_bytes(b"png")
        (results_root / "alias").symlink_to(results_root / "frag")
        reader = ArtifactsReader(results_root)

        assert reader.output_file("alias", "plot.png") == results_root / "alias" / "plot.png"

    def test_symlinked_results_root_is_allowed(self, layout, tmp_path):
        results_root, _ = layout
        linked_root = tmp_path / "linked-results"
        linked_root.symlink_to(results_root)
        reader = ArtifactsReader(linked_root)

        assert reader.output_file("frag", "plot.png") == linked_root / "frag" / "plot.png"

    def test_missing_path_is_allowed(self, layout):
        results_root, _ = layout
        reader = ArtifactsReader(results_root)

        assert reader.output_file("missing", "plot.png") == results_root / "missing" / "plot.png"
