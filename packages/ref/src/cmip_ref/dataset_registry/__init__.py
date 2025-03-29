"""
Data registries for non-published reference data

These data are placeholders until these data have been added to obs4MIPs.
The AR7 FT REF requires that reference datasets are openly licensed before it is included
in any published data catalogs.
"""

import importlib.resources
import os
import pathlib
import shutil

import pooch
from loguru import logger

DATA_VERSION = "v3.9"
"""
Default version identifier for the datasets

Changing this will bust any existing caches.
"""


def build_reference_data_registry(
    version: str = DATA_VERSION,
) -> pooch.Pooch:
    """
    Build a pooch registry of reference data associated with PMP that isn't currently in obs4MIPs.

    Currently we only have reference datasets published from PMP,
    but this may change in the future.

    Parameters
    ----------
    version : str
        The version of the data.

        Changing the version will invalidate the cache and force a re-download of the data.

    Returns
    -------
    pooch.Pooch
        The pooch registry.
    """
    registry = pooch.create(
        path=pooch.os_cache("pmp"),
        base_url="https://pcmdiweb.llnl.gov/pss/pmpdata/",
        version=version,
        env="REF_METRICS_PMP_DATA_DIR",
    )
    registry.load_registry(importlib.resources.open_binary("cmip_ref.dataset_registry", "pmp_reference.txt"))
    return registry


def fetch_all_files(registry: pooch.Pooch, output_dir: pathlib.Path, symlink: bool = False) -> None:
    """
    Fetch all files associated with a pooch registry and write them to an output directory.

    Pooch fetches, caches and validates the downloaded files.
    Subsequent calls to this function will not refetch any previously downloaded files.

    Parameters
    ----------
    registry
        Pooch directory containing a set of files that should be fetched.
    output_dir
        The root directory to write the files to.

        The directory will be created if it doesn't exist,
        and matching files will be overwritten.
    symlink
        If True, symlink all files to this directory.
        Otherwise, perform a copy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in registry.registry.keys():
        fetch_file = registry.fetch(key)

        linked_file = output_dir / key
        linked_file.parent.mkdir(parents=True, exist_ok=True)
        if not linked_file.exists():  # pragma: no cover
            if symlink:
                logger.info(f"Linking {key} to {linked_file}")

                os.symlink(fetch_file, linked_file)
            else:
                logger.info(f"Copying {key} to {linked_file}")
                shutil.copy(fetch_file, linked_file)
        else:
            logger.info(f"File {linked_file} already exists. Skipping.")
