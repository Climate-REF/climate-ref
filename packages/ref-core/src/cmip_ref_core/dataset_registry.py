"""
Data registries for non-published reference data

These data are placeholders until these data have been added to obs4MIPs.
The CMIP7 Assessment Fas Track REF requires that reference datasets are openly licensed
before it is included in any published data catalogs.
"""

import importlib.resources
import os
import pathlib
import shutil

import pooch
from loguru import logger
from rich.progress import track


def fetch_all_files(
    registry: pooch.Pooch,
    name: str,
    output_dir: pathlib.Path | None,
    symlink: bool = False,
) -> None:
    """
    Fetch all files associated with a pooch registry and write them to an output directory.

    Pooch fetches, caches and validates the downloaded files.
    Subsequent calls to this function will not refetch any previously downloaded files.

    Parameters
    ----------
    registry
        Pooch directory containing a set of files that should be fetched.
    name
        Name of the registry.
    output_dir
        The root directory to write the files to.

        The directory will be created if it doesn't exist,
        and matching files will be overwritten.

        If no directory is provided, the files will be fetched from the remote server,
        but not copied anywhere.
    symlink
        If True, symlink all files to this directory.
        Otherwise, perform a copy.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for key in track(registry.registry.keys(), description=f"Fetching {name} data"):
        fetch_file = registry.fetch(key)

        if output_dir is None:
            # Just warm the cache and move onto the next file
            continue

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


class DatasetRegistryManager:
    """
    A collection of reference datasets registries

    The REF requires additional reference datasets
    in addition to obs4MIPs data which can be downloaded via ESGF.
    Each provider may have different sets of reference data that are needed.
    These are provider-specific datasets are datasets not yet available in obs4MIPs,
    or are post-processed from obs4MIPs.

    A dataset registry consists of a file that contains a list of files and checksums,
    in combination with a base URL that is used to fetch the files.
    [Pooch](https://www.fatiando.org/pooch/latest/) is used within the DataRegistry
    to manage the caching, downloading and validation of the files.

    All datasets that are registered here are expected to be openly licensed and freely available.
    """

    def __init__(self) -> None:
        self._registries: dict[str, pooch.Pooch] = {}

    def __getitem__(self, item: str) -> pooch.Pooch:
        """
        Get a registry by name
        """
        return self._registries[item]

    def keys(self) -> list[str]:
        """
        Get the list of registry names
        """
        return list(self._registries.keys())

    def register(  # noqa: PLR0913
        self,
        name: str,
        base_url: str,
        package: str,
        resource: str,
        cache_name: str | None = None,
        version: str | None = None,
    ) -> None:
        """
        Register a new dataset registry

        This will create a new Pooch registry and add it to the list of registries.
        This is typically used by a provider to register a new collections of datasets at runtime.

        Parameters
        ----------
        name
            Name of the registry

            This is used to identify the registry
        base_url
            Commmon URL prefix for the files
        package
            Name of the package containing the registry resource.
        resource
            Name of the resource in the package that contains a list of files and checksums.

            This must be formatted in a way that is expected by pooch.
        version
            The version of the data.

            Changing the version will invalidate the cache and force a re-download of the data.
        cache_name
            Name to use to generate the cache directory.

            This defaults to the value of `name` if not provided.
        """
        if cache_name is None:
            cache_name = "ref"

        registry = pooch.create(
            path=pooch.os_cache(cache_name),
            base_url=base_url,
            version=version,
            env="REF_METRICS_DATA_DIR",
        )
        registry.load_registry(str(importlib.resources.files(package) / resource))
        self._registries[name] = registry


dataset_registry_manager = DatasetRegistryManager()
