"""
Tests that guard the data files shipped inside the installed packages.

None of these files are configured explicitly in any `pyproject.toml`.
They ride on the hatchling default of including everything under `src/<package>`,
so a build backend change or a stray exclude would drop them silently.
Every file the REF reads out of a package at runtime is listed here.
"""

import importlib.resources

import pytest
import yaml

from climate_ref_core.data import PackagedResource

# Every data file the REF reads out of an installed package at runtime.
BUNDLED_FILES = [
    ("climate_ref_core.pycmec", "cv_cmip7_aft.yaml"),
    ("climate_ref_core", "data/cmip6_cmip7_variable_map.json"),
    ("climate_ref", "default_ignore_datasets.yaml"),
    ("climate_ref.dataset_registry", "obs4ref_reference.txt"),
    ("climate_ref.dataset_registry", "sample_data.txt"),
    ("climate_ref.dataset_registry", "quickstart.txt"),
    ("climate_ref_ilamb.dataset_registry", "ilamb.txt"),
    ("climate_ref_ilamb.dataset_registry", "ilamb_regions.txt"),
    ("climate_ref_ilamb.configure", "ilamb.yaml"),
    ("climate_ref_ilamb.configure", "iomb.yaml"),
    ("climate_ref_pmp.dataset_registry", "pmp_climatology.txt"),
    ("climate_ref_esmvaltool.dataset_registry", "data.txt"),
    ("climate_ref_esmvaltool", "recipes.txt"),
]


def test_pycmec_package_data_is_importable():
    """``climate_ref_core.pycmec`` must be importable as a resource package."""
    files = importlib.resources.files("climate_ref_core.pycmec")
    assert files.is_dir()


def _require(package: str) -> None:
    """Skip when the owning package is not installed, so core can be tested on its own."""
    pytest.importorskip(package.split(".", maxsplit=1)[0])


@pytest.mark.parametrize("package, resource", BUNDLED_FILES, ids=lambda value: value)
def test_bundled_data_files_resolve(package, resource):
    """Bundled package-data files must resolve and be readable from the wheel."""
    _require(package)
    packaged = PackagedResource(package, resource)

    assert packaged.exists(), f"{packaged} is missing from the installed package"

    contents = packaged.read_text()
    assert contents, f"{packaged} resolved but is empty"


@pytest.mark.parametrize(
    "package, resource",
    [
        ("climate_ref_core.pycmec", "cv_cmip7_aft.yaml"),
        ("climate_ref", "default_ignore_datasets.yaml"),
        ("climate_ref_ilamb.configure", "ilamb.yaml"),
        ("climate_ref_ilamb.configure", "iomb.yaml"),
    ],
    ids=lambda value: value,
)
def test_bundled_yaml_files_parse(package, resource):
    """A truncated or corrupt YAML file would break startup, so parse each one."""
    _require(package)
    parsed = yaml.safe_load(PackagedResource(package, resource).read_text())

    assert isinstance(parsed, dict)
