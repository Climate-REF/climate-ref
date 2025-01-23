"""
Testing utilities
"""

import importlib.resources
import os
import shutil
from pathlib import Path

import pooch
from loguru import logger


def _determine_test_directory() -> Path | None:
    expected = Path(__file__).parents[4] / "tests" / "test-data"

    if not expected.exists():  # pragma: no cover
        return None
    return expected


def _build_sample_data_registry(sample_data_version: str) -> pooch.Pooch:
    registry = pooch.create(
        path=pooch.os_cache("ref_sample_data"),
        base_url="https://raw.githubusercontent.com/CMIP-REF/ref-sample-data/refs/tags/{version}/data/",
        version=sample_data_version,
        env="REF_SAMPLE_DATA_DIR",
    )

    with (
        importlib.resources.files("cmip_ref")
        .joinpath("datasets")
        .joinpath("sample_data.txt")
        .open("rb") as fh
    ):
        registry.load_registry(fh)

    return registry


TEST_DATA_DIR = _determine_test_directory()
SAMPLE_DATA_VERSION = "v0.3.0"


def fetch_sample_data(version: str = SAMPLE_DATA_VERSION) -> None:
    """
    Fetch the sample data for the given version.

    Parameters
    ----------
    version
        The version tag of the sample data to fetch.

        This will fail if the version is not found in the sample data registry
        or if the sample data registry file is incompatible with this version.
    """
    sample_registry = _build_sample_data_registry(version)

    output_dir = None

    if TEST_DATA_DIR:  # pragma: no branch
        output_dir = TEST_DATA_DIR / "sample-data"
        if output_dir.exists():  # pragma: no branch
            logger.warning("Removing existing sample data")
            shutil.rmtree(output_dir)

    for key in sample_registry.registry.keys():
        fetch_file = sample_registry.fetch(key)

        if output_dir:  # pragma: no branch
            linked_file = output_dir / key
            linked_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating symlink for {linked_file}")
            os.symlink(fetch_file, linked_file)
