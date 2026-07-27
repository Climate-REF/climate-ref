"""
Resolution of the data files that are distributed with the REF

Several files the REF depends on are shipped inside the wheels:
the controlled vocabulary, the grey list, and the dataset registry manifests.
Some of them also have a newer copy available over the network.

`LayeredResource` resolves one logical file across three layers:
an operator override, a cached download, and the copy shipped in the package.
The packaged copy is always present and always readable,
so resolution never requires the network or a writable filesystem.

Packaged files are read through `importlib.resources` rather than being converted
to a filesystem path, so an installation that is not unpacked on disk still works.
"""

import enum
import importlib.resources
import os
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

import platformdirs
from attrs import field, frozen

from climate_ref_core.exceptions import RefException


class DataResourceError(RefException):
    """Raised when a data file cannot be resolved or read."""


def resolve_cache_dir(cache_name: str) -> pathlib.Path:
    """
    Resolve a cache directory used to hold downloaded data

    If the ``REF_DATASET_CACHE_DIR`` environment variable is set, use that as the root.
    Otherwise, fall back to the OS cache under ``climate_ref``.

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
        cache_dir = platformdirs.user_cache_path("climate_ref")

    return cache_dir / cache_name


@frozen
class PackagedResource:
    """
    A data file shipped inside an installed package

    The file is always accessed through `importlib.resources`.
    It is never converted to a filesystem path unless a caller explicitly asks for one
    via `as_path`, which extracts it to a temporary location when needed.
    """

    package: str
    """Import name of the package that contains the file, for example ``climate_ref_core.pycmec``."""

    resource: str
    """Name of the file within that package, for example ``cv_cmip7_aft.yaml``."""

    def exists(self) -> bool:
        """
        Check whether the file is present in the installed package

        Returns
        -------
        :
            True if the file can be read.
        """
        try:
            return (importlib.resources.files(self.package) / self.resource).is_file()
        except (ModuleNotFoundError, FileNotFoundError):
            return False

    def read_text(self, encoding: str = "utf-8") -> str:
        """
        Read the file as text

        Parameters
        ----------
        encoding
            Text encoding of the file.

        Returns
        -------
        :
            The contents of the file.

        Raises
        ------
        DataResourceError
            If the file is missing from the installed package.
        """
        try:
            return (importlib.resources.files(self.package) / self.resource).read_text(encoding=encoding)
        except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            raise DataResourceError(
                f"Could not read {self} from the installed package. "
                "This usually means the package was built without its data files."
            ) from exc

    @contextmanager
    def open_text(self, encoding: str = "utf-8") -> Iterator[IO[str]]:
        """
        Open the file as a text stream

        Parameters
        ----------
        encoding
            Text encoding of the file.

        Yields
        ------
        :
            An open text stream positioned at the start of the file.
        """
        try:
            handle = (importlib.resources.files(self.package) / self.resource).open("r", encoding=encoding)
        except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            raise DataResourceError(f"Could not open {self} from the installed package.") from exc
        try:
            yield handle
        finally:
            handle.close()

    @contextmanager
    def as_path(self) -> Iterator[pathlib.Path]:
        """
        Expose the file as a filesystem path for the duration of the context

        Prefer `read_text` or `open_text`.
        Use this only for third-party APIs that insist on a path.
        The path may point at a temporary file that is removed on exit,
        so it must not be retained beyond the context.

        Yields
        ------
        :
            A path that exists for the duration of the context.
        """
        # Only the lookup is guarded. An exception raised in the caller's `with` body
        # propagates back in through this yield, and must not be relabelled.
        try:
            manager = importlib.resources.as_file(importlib.resources.files(self.package) / self.resource)
        except (ModuleNotFoundError, FileNotFoundError, OSError) as exc:
            raise DataResourceError(f"Could not materialise {self} as a filesystem path.") from exc
        with manager as path:
            yield path

    def __str__(self) -> str:
        return f"{self.package}/{self.resource}"


class ResourceOrigin(enum.Enum):
    """
    Which layer a resolved data file came from
    """

    override = "override"
    """An explicit path supplied by the operator via configuration or an environment variable."""

    cache = "cache"
    """A copy downloaded into the local cache."""

    package = "package"
    """The copy shipped inside the installed package."""


@frozen
class LayeredResource:
    """
    A data file resolved across an override, a cache, and the packaged copy

    Resolution happens on every access rather than being fixed at construction,
    so a cache that is populated after the object is built is picked up
    without the object being rebuilt.
    """

    packaged: PackagedResource
    """The copy shipped in the package. This is the floor, and is always available."""

    override: pathlib.Path | None = field(default=None)
    """An explicit path supplied by the operator. Takes precedence over everything else."""

    cache: pathlib.Path | None = field(default=None)
    """A local cache location. Used only when the file is actually present there."""

    def resolve(self) -> tuple[ResourceOrigin, pathlib.Path | PackagedResource]:
        """
        Determine which layer supplies the file

        Returns
        -------
        :
            The origin, and either a filesystem path or the packaged resource.

        Raises
        ------
        DataResourceError
            If an override was supplied but does not point at a readable file.
        """
        if self.override is not None:
            if not self.override.is_file():
                raise DataResourceError(
                    f"The configured file {self.override} does not exist. "
                    "Point it at an existing file, or remove the setting to use the copy "
                    f"shipped with the REF ({self.packaged})."
                )
            return ResourceOrigin.override, self.override

        if self.cache is not None and self.cache.is_file():
            return ResourceOrigin.cache, self.cache

        return ResourceOrigin.package, self.packaged

    @property
    def origin(self) -> ResourceOrigin:
        """
        The layer that currently supplies the file
        """
        return self.resolve()[0]

    def describe(self) -> str:
        """
        Describe where the file is being read from

        This never raises, so it is safe to use when building a log message
        for a resolution that has itself failed.

        Returns
        -------
        :
            A short description suitable for a log message.
        """
        try:
            origin, source = self.resolve()
        except DataResourceError:
            return f"{self.override} (missing)"
        return f"{source} ({origin.value})"

    def read_text(self, encoding: str = "utf-8") -> str:
        """
        Read the file as text from whichever layer supplies it

        Parameters
        ----------
        encoding
            Text encoding of the file.

        Returns
        -------
        :
            The contents of the file.
        """
        _, source = self.resolve()
        if isinstance(source, PackagedResource):
            return source.read_text(encoding=encoding)
        try:
            return source.read_text(encoding=encoding)
        except OSError as exc:
            raise DataResourceError(f"Could not read {source}.") from exc

    @contextmanager
    def open_text(self, encoding: str = "utf-8") -> Iterator[IO[str]]:
        """
        Open the file as a text stream from whichever layer supplies it

        Parameters
        ----------
        encoding
            Text encoding of the file.

        Yields
        ------
        :
            An open text stream positioned at the start of the file.
        """
        _, source = self.resolve()
        if isinstance(source, PackagedResource):
            with source.open_text(encoding=encoding) as handle:
                yield handle
        else:
            with source.open("r", encoding=encoding) as handle:
                yield handle

    @contextmanager
    def as_path(self) -> Iterator[pathlib.Path]:
        """
        Expose the file as a filesystem path for the duration of the context

        Yields
        ------
        :
            A path that exists for the duration of the context.
        """
        _, source = self.resolve()
        if isinstance(source, PackagedResource):
            with source.as_path() as path:
                yield path
        else:
            yield source
