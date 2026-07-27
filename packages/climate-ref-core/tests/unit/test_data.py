"""Tests for the layered resolution of data files distributed with the REF."""

from pathlib import Path

import pytest

from climate_ref_core.data import (
    DataResourceError,
    FileResource,
    LayeredResource,
    PackagedResource,
    ResourceOrigin,
    resolve_cache_dir,
)

CV = PackagedResource("climate_ref_core.pycmec", "cv_cmip7_aft.yaml")


class TestPackagedResource:
    def test_exists(self):
        assert CV.exists()

    def test_missing_resource_does_not_exist(self):
        assert not PackagedResource("climate_ref_core.pycmec", "nope.yaml").exists()

    def test_missing_package_does_not_exist(self):
        assert not PackagedResource("climate_ref_core.not_a_package", "nope.yaml").exists()

    def test_read_text(self):
        assert "dimensions" in CV.read_text()

    def test_read_missing_raises(self):
        with pytest.raises(DataResourceError, match="Could not read"):
            PackagedResource("climate_ref_core.pycmec", "nope.yaml").read_text()

    def test_as_path(self):
        with CV.as_path() as path:
            assert path.is_file()
            assert path.read_text(encoding="utf-8") == CV.read_text()

    def test_as_path_does_not_swallow_body_errors(self):
        # An OSError raised by the caller must propagate unchanged, not be relabelled
        # as a resolution failure.
        with pytest.raises(OSError, match="from the body"):
            with CV.as_path():
                raise OSError("from the body")

    def test_str(self):
        assert str(CV) == "climate_ref_core.pycmec/cv_cmip7_aft.yaml"


class TestFileResource:
    def test_read_text(self, tmp_path):
        target = tmp_path / "a.yaml"
        target.write_text("contents", encoding="utf-8")

        assert FileResource(target).read_text() == "contents"

    def test_exists(self, tmp_path):
        assert not FileResource(tmp_path / "missing.yaml").exists()

    def test_read_missing_raises(self, tmp_path):
        with pytest.raises(DataResourceError, match="Could not read"):
            FileResource(tmp_path / "missing.yaml").read_text()

    def test_as_path_returns_the_path_itself(self, tmp_path):
        target = tmp_path / "a.yaml"
        target.write_text("contents", encoding="utf-8")

        with FileResource(target).as_path() as path:
            assert path == target

    def test_str(self, tmp_path):
        assert str(FileResource(tmp_path / "a.yaml")) == str(tmp_path / "a.yaml")


class TestLayeredResource:
    def test_falls_back_to_package(self):
        resource = LayeredResource(packaged=CV)

        assert resource.origin == ResourceOrigin.package
        assert resource.read_text() == CV.read_text()
        assert resource.describe() == f"{CV} (package)"

    def test_cache_wins_over_package(self, tmp_path):
        cache = tmp_path / "cached.yaml"
        cache.write_text("cached", encoding="utf-8")

        resource = LayeredResource(packaged=CV, cache=cache)

        assert resource.origin == ResourceOrigin.cache
        assert resource.read_text() == "cached"

    def test_absent_cache_falls_through(self, tmp_path):
        resource = LayeredResource(packaged=CV, cache=tmp_path / "missing.yaml")

        assert resource.origin == ResourceOrigin.package

    def test_cache_populated_later_is_picked_up(self, tmp_path):
        cache = tmp_path / "cached.yaml"
        resource = LayeredResource(packaged=CV, cache=cache)
        assert resource.origin == ResourceOrigin.package

        cache.write_text("cached", encoding="utf-8")

        assert resource.origin == ResourceOrigin.cache

    def test_override_wins_over_cache(self, tmp_path):
        cache = tmp_path / "cached.yaml"
        cache.write_text("cached", encoding="utf-8")
        override = tmp_path / "override.yaml"
        override.write_text("override", encoding="utf-8")

        resource = LayeredResource(packaged=CV, override=override, cache=cache)

        assert resource.origin == ResourceOrigin.override
        assert resource.read_text() == "override"

    def test_missing_override_raises_with_guidance(self, tmp_path):
        resource = LayeredResource(packaged=CV, override=tmp_path / "missing.yaml")

        with pytest.raises(DataResourceError, match="does not exist"):
            resource.read_text()

    def test_describe_never_raises_for_a_missing_override(self, tmp_path):
        resource = LayeredResource(packaged=CV, override=tmp_path / "missing.yaml")

        assert resource.describe() == f"{tmp_path / 'missing.yaml'} (missing)"


class TestResolveCacheDir:
    def test_default_root(self, monkeypatch, mocker):
        monkeypatch.delenv("REF_DATASET_CACHE_DIR", raising=False)
        mocker.patch(
            "climate_ref_core.data.platformdirs.user_cache_path",
            return_value=Path("/cache/climate_ref"),
        )

        assert resolve_cache_dir("grey_list") == Path("/cache/climate_ref/grey_list")

    def test_environment_variable_root(self, monkeypatch):
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", "/somewhere/else")

        assert resolve_cache_dir("grey_list") == Path("/somewhere/else/grey_list")

    def test_environment_variable_expansion(self, monkeypatch):
        monkeypatch.setenv("A_ROOT", "/expanded")
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", "$A_ROOT/cache")

        assert resolve_cache_dir("grey_list") == Path("/expanded/cache/grey_list")
