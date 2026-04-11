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
import pooch.hashes
from loguru import logger
from rich.progress import track

from climate_ref_core.env import env

DATASET_URL = env.str("REF_DATASET_URL", default="https://obs4ref.climate-ref.org")


def _verify_hash_matches(fname: str | pathlib.Path, known_hash: str) -> bool:
    """
    Check if the hash of a file matches a known hash.

    Coverts hashes to lowercase before comparison to avoid system specific
    mismatches between hashes in the registry and computed hashes.

    This is a tweaked version of the `pooch.hashes.hash_matches` function with a custom error message.

    Parameters
    ----------
    fname
        The path to the file.
    known_hash
        The known hash. Optionally, prepend ``alg:`` to the hash to specify the
        hashing algorithm. Default is SHA256.

    Raises
    ------
    ValueError
        If the hash does not match.
    FileNotFoundError
        If the file does not exist.

    Returns
    -------
    bool
        True if the hash matches.
    """
    fname = pathlib.Path(fname)

    if not fname.exists():
        raise FileNotFoundError(f"File {fname!s} does not exist. Cannot verify hash.")

    algorithm = pooch.hashes.hash_algorithm(known_hash)
    new_hash = pooch.hashes.file_hash(str(fname), alg=algorithm)
    matches = new_hash.lower() == known_hash.rsplit(":", maxsplit=1)[-1].lower()
    if not matches:
        raise ValueError(
            f"{algorithm.upper()} hash of downloaded file ({fname!s}) does not match"
            f" the known hash: expected {known_hash} but got {new_hash}. "
            f"The file may have been corrupted or the known hash may be outdated. "
            f"Delete the file and try again."
        )
    return matches


def validate_registry_cache(
    registry: pooch.Pooch,
    name: str,
) -> list[str]:
    """
    Validate that all files in a registry are cached and have correct checksums.

    Parameters
    ----------
    registry
        Pooch registry to validate.
    name
        Name of the registry (for error messages).

    Returns
    -------
    list[str]
        List of error messages for any validation failures.
        Empty list if all files are valid.
    """
    errors: list[str] = []

    for key in registry.registry.keys():
        expected_hash = registry.registry[key]
        if not isinstance(expected_hash, str) or not expected_hash:  # pragma: no cover
            errors.append(f"{name}: No hash defined for {key}")
            continue

        # Check if file exists in cache
        cached_path = registry.abspath / key  # type: ignore[attr-defined]
        if not cached_path.exists():
            errors.append(f"{name}: File not cached: {key}")
            continue

        # Verify checksum
        try:
            _verify_hash_matches(cached_path, expected_hash)
        except ValueError as e:
            errors.append(f"{name}: {e}")

    return errors


def fetch_all_files(
    registry: pooch.Pooch,
    name: str,
    output_dir: pathlib.Path | None,
    symlink: bool = False,
    verify: bool = True,
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
    verify
        If True, verify the checksums of the local files against the registry.
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for key in track(registry.registry.keys(), description=f"Fetching {name} data"):
        fetch_file = registry.fetch(key)
        expected_hash = registry.registry[key]
        if not isinstance(expected_hash, str) or not expected_hash:  # pragma: no cover
            raise ValueError(f"Expected a hash for {key} but got {expected_hash}")

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
        if verify:
            _verify_hash_matches(linked_file, expected_hash)


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

    @staticmethod
    def _resolve_cache_dir(cache_name: str) -> pathlib.Path:
        """
        Resolve the cache directory for a registry.

        If the ``REF_DATASET_CACHE_DIR`` environment variable is set,
        use that as the root. Otherwise, fall back to the OS cache
        under ``climate_ref``.

        Parameters
        ----------
        cache_name
            Subdirectory name within the cache root.

        Returns
        -------
            The resolved cache directory path.
        """
        if env_cache_dir := os.environ.get("REF_DATASET_CACHE_DIR"):
            cache_dir = pathlib.Path(os.path.expandvars(env_cache_dir)).expanduser()
        else:
            cache_dir = pooch.os_cache("climate_ref")

        return cache_dir / cache_name

    @staticmethod
    def _migrate_cache(
        registry: pooch.Pooch,
        legacy_cache_dirs: list[pathlib.Path],
    ) -> None:
        """
        Migrate cached files from legacy cache directories to the current location.

        For each file in the registry,
        if it does not already exist at the
        current cache path but is found in one of the legacy directories,
        move it to the new location.

        Parameters
        ----------
        registry
            The Pooch registry whose cache may need migrating.
        legacy_cache_dirs
            Directories where files may have been cached previously.
        """
        new_root: pathlib.Path = registry.abspath  # type: ignore[attr-defined]

        for key in registry.registry:
            new_path = new_root / key
            if new_path.exists():
                continue

            for legacy_dir in legacy_cache_dirs:
                old_path = legacy_dir / key
                if old_path.exists():
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Migrating cached file {key}: {old_path} -> {new_path}")
                    shutil.move(str(old_path), str(new_path))
                    break

    def register(  # noqa: PLR0913
        self,
        name: str,
        base_url: str,
        package: str,
        resource: str,
        cache_name: str | None = None,
        version: str | None = None,
        legacy_cache_dirs: list[pathlib.Path] | None = None,
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
        legacy_cache_dirs
            Previous cache directories to migrate files from.

            If provided, any files that exist in a legacy directory but not in
            the current cache will be moved to the new location. This avoids
            re-downloading data after a cache layout change.
        """
        if cache_name is None:
            cache_name = name

        cache_path = self._resolve_cache_dir(cache_name)

        registry = pooch.create(
            path=cache_path,
            base_url=base_url,
            version=version,
            retry_if_failed=10,
        )
        registry.load_registry(str(importlib.resources.files(package) / resource))

        if legacy_cache_dirs:
            self._migrate_cache(registry, legacy_cache_dirs)

        self._registries[name] = registry


dataset_registry_manager = DatasetRegistryManager()
