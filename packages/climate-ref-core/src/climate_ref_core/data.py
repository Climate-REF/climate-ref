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
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from typing import Protocol

import platformdirs
from attrs import field, frozen

from climate_ref_core.exceptions import RefException

_RESOURCE_ERRORS = (ModuleNotFoundError, FileNotFoundError, OSError)


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
        cache_dir = pathlib.Path(os.path.expandvars(env_cache_dir))
        try:
            cache_dir = cache_dir.expanduser()
        except RuntimeError:
            # An unresolvable ``~user`` is left as-is rather than failing the caller.
            pass
    else:
        cache_dir = platformdirs.user_cache_path("climate_ref")

    return cache_dir / cache_name


class Resource(Protocol):
    """
    Something a data file can be read from

    This is the contract `LayeredResource` resolves to,
    and the narrowest one its consumers need.
    """

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read the file as text."""

    def describe(self) -> str:
        """Describe where the file is read from, for a log or error message."""


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

    def _traversable(self) -> importlib.resources.abc.Traversable:
        """
        Locate the file within the installed package.

        Existence is checked here rather than at each accessor,
        because `importlib.resources.as_file` happily hands back a path that does not exist.

        Returns
        -------
        :
            The located file.

        Raises
        ------
        DataResourceError
            If the file is not present in the installed package.
        """
        missing = DataResourceError(
            f"Could not read {self} from the installed package. "
            "This usually means the package was built without its data files."
        )
        try:
            located = importlib.resources.files(self.package) / self.resource
            if not located.is_file():
                raise missing
        except _RESOURCE_ERRORS as exc:
            raise missing from exc
        return located

    def exists(self) -> bool:
        """
        Check whether the file is present in the installed package

        Returns
        -------
        :
            True if the file can be read.
        """
        try:
            self._traversable()
        except DataResourceError:
            return False
        return True

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
        """
        traversable = self._traversable()
        try:
            return traversable.read_text(encoding=encoding)
        except _RESOURCE_ERRORS as exc:
            raise DataResourceError(f"Could not read {self} from the installed package.") from exc

    @contextmanager
    def as_path(self) -> Generator[pathlib.Path]:
        """
        Expose the file as a filesystem path for the duration of the context

        Prefer `read_text`.
        Use this only for third-party APIs that insist on a path.
        The path may point at a temporary file that is removed on exit,
        so it must not be retained beyond the context.

        Yields
        ------
        :
            A path that exists for the duration of the context.
        """
        traversable = self._traversable()
        with ExitStack() as stack:
            try:
                path = stack.enter_context(importlib.resources.as_file(traversable))
            except _RESOURCE_ERRORS as exc:
                raise DataResourceError(f"Could not materialise {self} as a filesystem path.") from exc
            # Only the lookup is guarded. An exception raised in the caller's `with`
            # block propagates back in through this yield and must not be relabelled.
            yield path

    def describe(self) -> str:
        """
        Describe where the file is read from

        Returns
        -------
        :
            The package and file name.
        """
        return str(self)

    def __str__(self) -> str:
        return f"{self.package}/{self.resource}"


@frozen
class FileResource:
    """
    A data file at a plain filesystem path
    """

    path: pathlib.Path

    def exists(self) -> bool:
        """
        Check whether the file exists

        Returns
        -------
        :
            True if the path points at a readable file.
        """
        return self.path.is_file()

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
        """
        try:
            return self.path.read_text(encoding=encoding)
        except OSError as exc:
            raise DataResourceError(f"Could not read {self}.") from exc

    @contextmanager
    def as_path(self) -> Generator[pathlib.Path]:
        """
        Expose the file as a filesystem path

        Yields
        ------
        :
            The path itself, which already exists on disk.
        """
        yield self.path

    def describe(self) -> str:
        """
        Describe where the file is read from

        Returns
        -------
        :
            The path.
        """
        return str(self)

    def __str__(self) -> str:
        return str(self.path)


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

    def resolve(self) -> tuple[ResourceOrigin, Resource]:
        """
        Determine which layer supplies the file

        Returns
        -------
        :
            The origin, and the resource that supplies the file.

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
            return ResourceOrigin.override, FileResource(self.override)

        if self.cache is not None and self.cache.is_file():
            return ResourceOrigin.cache, FileResource(self.cache)

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
        return self.resolve()[1].read_text(encoding=encoding)
