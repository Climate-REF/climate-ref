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
SAMPLE_DATA_VERSION = "v0.3.3"


def fetch_sample_data(
    version: str = SAMPLE_DATA_VERSION, force_cleanup: bool = False, symlink: bool = False
) -> None:
    """
    Fetch the sample data for the given version.

    Parameters
    ----------
    version
        The version tag of the sample data to fetch.

        This will fail if the version is not found in the sample data registry
        or if the sample data registry file is incompatible with this version.
    force_cleanup
        If True, remove any existing files
    symlink
        If True, symlink in the data otherwise copy the files

        The symlink approach is faster, but will fail when running with a non-local executor
        because the symlinks can't be followed.
    """

    if TEST_DATA_DIR is None:  # pragma: no cover
        logger.warning("Test data directory not found, skipping sample data fetch")
        return

    sample_registry = _build_sample_data_registry(version)

    output_dir = TEST_DATA_DIR / "sample-data"
    version_file = output_dir / "version.txt"
    existing_version = None

    if output_dir.exists():  # pragma: no branch
        if version_file.exists():  # pragma: no branch
            with open(version_file) as fh:
                existing_version = fh.read().strip()

        if force_cleanup or existing_version != SAMPLE_DATA_VERSION:  # pragma: no branch
            logger.warning("Removing existing sample data")
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for key in sample_registry.registry.keys():
        fetch_file = sample_registry.fetch(key)

        linked_file = output_dir / key
        linked_file.parent.mkdir(parents=True, exist_ok=True)
        if not linked_file.exists():  # pragma: no cover
            if symlink:
                logger.info(f"Linking {key} to {linked_file}")

                os.symlink(fetch_file, linked_file)
            else:
                logger.info(f"Copying {key} to {linked_file}")
                shutil.copy(fetch_file, linked_file)

    # Write out the current sample data version to the copying as complete
    with open(output_dir / "version.txt", "w") as fh:
        fh.write(SAMPLE_DATA_VERSION)
