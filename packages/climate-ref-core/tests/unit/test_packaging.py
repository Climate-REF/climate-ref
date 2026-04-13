"""
Tests that guard the shape of the installed ``climate_ref_core`` package.
"""

import importlib.resources

import pytest
import yaml


def test_pycmec_package_data_is_importable():
    """``climate_ref_core.pycmec`` must be importable as a resource package."""
    files = importlib.resources.files("climate_ref_core.pycmec")
    assert files.is_dir()


@pytest.mark.parametrize("filename", ["cv_cmip7_aft.yaml"])
def test_bundled_data_files_resolve(filename):
    """Bundled package-data files must resolve and be readable from the wheel."""
    resource = importlib.resources.files("climate_ref_core.pycmec") / filename

    assert resource.is_file(), f"{filename} is missing from the installed climate_ref_core package"

    contents = resource.read_text(encoding="utf-8")
    assert contents, f"{filename} resolved but is empty"

    # The CV YAML must parse — a truncated or corrupt file would break startup.
    parsed = yaml.safe_load(contents)
    assert isinstance(parsed, dict)
