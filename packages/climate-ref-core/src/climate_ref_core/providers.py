"""
Interface for declaring a diagnostic provider.

This defines how diagnostic packages interoperate with the REF framework.
Each diagnostic package may contain multiple diagnostics.

Each diagnostic package must implement the `DiagnosticProvider` interface.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.resources
import os
import stat
import subprocess
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import requests
import yaml
from attrs import evolve
from loguru import logger

from climate_ref_core.constraints import IgnoreFacets
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.exceptions import InvalidDiagnosticException, InvalidProviderException

if TYPE_CHECKING:
    from climate_ref.config import Config


def _slugify(value: str) -> str:
    """
    Slugify a string.

    Parameters
    ----------
    value : str
        String to slugify.

    Returns
    -------
    str
        Slugified string.
    """
    return value.lower().replace(" ", "-")


class DiagnosticProvider:
    """
    The interface for registering and running diagnostics.

    Each package that provides diagnostics must implement this interface.
    """

    def __init__(self, name: str, version: str, slug: str | None = None) -> None:
        self.name = name
        self.slug = slug or _slugify(name)
        self.version = version

        self._diagnostics: dict[str, Diagnostic] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"

    def configure(self, config: Config) -> None:
        """
        Configure the provider.

        Parameters
        ----------
        config :
            A configuration.
        """
        logger.debug(
            f"Configuring provider {self.slug} using ignore_datasets_file {config.ignore_datasets_file}"
        )
        # The format of the configuration file is:
        # provider:
        #   diagnostic:
        #     source_type:
        #       - facet: value
        #       - other_facet: [other_value1, other_value2]
        ignore_datasets_all = yaml.safe_load(config.ignore_datasets_file.read_text(encoding="utf-8")) or {}
        ignore_datasets = ignore_datasets_all.get(self.slug, {})
        if unknown_slugs := {slug for slug in ignore_datasets} - {d.slug for d in self.diagnostics()}:
            logger.warning(
                f"Unknown diagnostics found in {config.ignore_datasets_file} "
                f"for provider {self.slug}: {', '.join(sorted(unknown_slugs))}"
            )

        known_source_types = {s.value for s in iter(SourceDatasetType)}
        for diagnostic in self.diagnostics():
            if diagnostic.slug in ignore_datasets:
                if unknown_source_types := set(ignore_datasets[diagnostic.slug]) - known_source_types:
                    logger.warning(
                        f"Unknown source types found in {config.ignore_datasets_file} for "
                        f"diagnostic '{diagnostic.slug}' by provider {self.slug}: "
                        f"{', '.join(sorted(unknown_source_types))}"
                    )
                data_requirements = (
                    r if isinstance(r, Sequence) else (r,) for r in diagnostic.data_requirements
                )
                diagnostic.data_requirements = tuple(
                    tuple(
                        evolve(
                            data_requirement,
                            constraints=tuple(
                                IgnoreFacets(facets)
                                for facets in ignore_datasets[diagnostic.slug].get(
                                    data_requirement.source_type.value, []
                                )
                            )
                            + data_requirement.constraints,
                        )
                        for data_requirement in requirement_collection
                    )
                    for requirement_collection in data_requirements
                )

    def diagnostics(self) -> list[Diagnostic]:
        """
        Iterate over the available diagnostics for the provider.

        Returns
        -------
        :
            Iterator over the currently registered diagnostics.
        """
        return list(self._diagnostics.values())

    def __len__(self) -> int:
        return len(self._diagnostics)

    def register(self, diagnostic: Diagnostic) -> None:
        """
        Register a diagnostic with the manager.

        Parameters
        ----------
        diagnostic :
            The diagnostic to register.
        """
        if not isinstance(diagnostic, Diagnostic):
            raise InvalidDiagnosticException(
                diagnostic, "Diagnostics must be an instance of the 'Diagnostic' class"
            )
        diagnostic.provider = self
        self._diagnostics[diagnostic.slug.lower()] = diagnostic

    def get(self, slug: str) -> Diagnostic:
        """
        Get a diagnostic by name.

        Parameters
        ----------
        slug :
            Name of the diagnostic (case-sensitive).

        Raises
        ------
        KeyError
            If the diagnostic with the given name is not found.

        Returns
        -------
        Diagnostic
            The requested diagnostic.
        """
        return self._diagnostics[slug.lower()]


def import_provider(fqn: str) -> DiagnosticProvider:
    """
    Import a provider by name

    Parameters
    ----------
    fqn
        Full package and attribute name of the provider to import

        For example: `climate_ref_example:provider` will use the `provider` attribute from the
        `climate_ref_example` package.

        If only a package name is provided, the default attribute name is `provider`.

    Raises
    ------
    InvalidProviderException
        If the provider cannot be imported

        If the provider isn't a valid `DiagnosticProvider`.

    Returns
    -------
    :
        DiagnosticProvider instance
    """
    if ":" not in fqn:
        fqn = f"{fqn}:provider"

    entrypoint = importlib.metadata.EntryPoint(name="provider", value=fqn, group="climate-ref.providers")

    try:
        provider = entrypoint.load()
        if not isinstance(provider, DiagnosticProvider):
            raise InvalidProviderException(fqn, f"Expected DiagnosticProvider, got {type(provider)}")
        return provider
    except ModuleNotFoundError:
        logger.error(f"Module '{fqn}' not found")
        raise InvalidProviderException(fqn, "Module not found")
    except AttributeError:
        logger.error(f"Provider '{fqn}' not found")
        raise InvalidProviderException(fqn, "Provider not found in module")


class CommandLineDiagnosticProvider(DiagnosticProvider):
    """
    A provider for diagnostics that can be run from the command line.
    """

    @abstractmethod
    def run(self, cmd: Iterable[str]) -> None:
        """
        Return the arguments for the command to run.
        """


MICROMAMBA_EXE_URL = (
    "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-{platform}-{arch}"
)
"""The URL to download the micromamba executable from."""


MICROMAMBA_MAX_AGE = datetime.timedelta(days=7)
"""Do not update if the micromamba executable is younger than this age."""


def _get_micromamba_url() -> str:
    """
    Build a platform specific URL from which to download micromamba.

    Based on the script at: https://micro.mamba.pm/install.sh

    """
    sysname = os.uname().sysname
    machine = os.uname().machine

    if sysname == "Linux":
        platform = "linux"
    elif sysname == "Darwin":
        platform = "osx"
    elif "NT" in sysname:
        platform = "win"
    else:
        platform = sysname

    arch = machine if machine in {"aarch64", "ppc64le", "arm64"} else "64"

    supported = {
        "linux-aarch64",
        "linux-ppc64le",
        "linux-64",
        "osx-arm64",
        "osx-64",
        "win-64",
    }
    if f"{platform}-{arch}" not in supported:
        msg = "Failed to detect your platform. Please set MICROMAMBA_EXE_URL to a valid location."
        raise ValueError(msg)

    return MICROMAMBA_EXE_URL.format(platform=platform, arch=arch)


class CondaDiagnosticProvider(CommandLineDiagnosticProvider):
    """
    A provider for diagnostics that can be run from the command line in a conda environment.

    Parameters
    ----------
    name
        The name of the provider.
    version
        The version of the provider.
    slug
        A slugified version of the name.
    repo
        URL of the git repository to install a development version of the package from.
    tag_or_commit
        Tag or commit to install from the `repo` repository.

    Attributes
    ----------
    env_vars
        Environment variables to set when running commands in the conda environment.
    url
        URL to install a development version of the package from.

    """

    def __init__(
        self,
        name: str,
        version: str,
        slug: str | None = None,
        repo: str | None = None,
        tag_or_commit: str | None = None,
    ) -> None:
        super().__init__(name, version, slug)
        self._conda_exe: Path | None = None
        self._prefix: Path | None = None
        self.url = f"git+{repo}@{tag_or_commit}" if repo and tag_or_commit else None
        self.env_vars: dict[str, str] = os.environ.copy()

    @property
    def prefix(self) -> Path:
        """Path where conda environments are stored."""
        if not isinstance(self._prefix, Path):
            msg = (
                "No prefix for conda environments configured. Please use the "
                "configure method to configure the provider or assign a value "
                "to prefix directly."
            )
            raise ValueError(msg)
        return self._prefix

    @prefix.setter
    def prefix(self, path: Path) -> None:
        self._prefix = path

    def configure(self, config: Config) -> None:
        """Configure the provider."""
        super().configure(config)
        self.prefix = config.paths.software / "conda"
        self.env_vars.setdefault("HOME", str(self.prefix))

    def _is_stale(self, path: Path) -> bool:
        """Check if a file is older than `MICROMAMBA_MAX_AGE`.

        Parameters
        ----------
        path
            The path to the file to check.

        Returns
        -------
            True if the file is older than `MICROMAMBA_MAX_AGE`, False otherwise.
        """
        creation_time = datetime.datetime.fromtimestamp(path.stat().st_ctime)
        age = datetime.datetime.now() - creation_time
        return age > MICROMAMBA_MAX_AGE

    def _install_conda(self, update: bool) -> Path:
        """Install micromamba in a specific location.

        Parameters
        ----------
        update:
            Update the micromamba executable if it is older than a week.

        Returns
        -------
            The path to the executable.

        """
        conda_exe = self.prefix / "micromamba"

        if not conda_exe.exists() or update or self._is_stale(conda_exe):
            logger.info("Installing conda")
            self.prefix.mkdir(parents=True, exist_ok=True)
            response = requests.get(_get_micromamba_url(), timeout=120, stream=True)
            response.raise_for_status()
            with conda_exe.open(mode="wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
            conda_exe.chmod(stat.S_IRWXU)
            logger.info("Successfully installed conda.")

        return conda_exe

    def get_conda_exe(self, update: bool = False) -> Path:
        """
        Get the path to a conda executable.
        """
        if self._conda_exe is None:
            self._conda_exe = self._install_conda(update)
        return self._conda_exe

    def get_environment_file(self) -> AbstractContextManager[Path]:
        """
        Return a context manager that provides the environment file as a Path.
        """
        # Because providers are instances, we have no way of retrieving the
        # module in which they are created, so get the information from the
        # first registered diagnostic instead.
        diagnostics = self.diagnostics()
        if len(diagnostics) == 0:
            msg = "Unable to determine the provider module, please register a diagnostic first."
            raise ValueError(msg)
        module = diagnostics[0].__module__.split(".")[0]
        lockfile = importlib.resources.files(module).joinpath("requirements").joinpath("conda-lock.yml")
        return importlib.resources.as_file(lockfile)

    @property
    def env_path(self) -> Path:
        """
        A unique path for storing the conda environment.
        """
        with self.get_environment_file() as file:
            suffix = hashlib.sha1(file.read_bytes(), usedforsecurity=False)
            if self.url is not None:
                suffix.update(bytes(self.url, encoding="utf-8"))
        return self.prefix / f"{self.slug}-{suffix.hexdigest()}"

    def create_env(self) -> None:
        """
        Create a conda environment.
        """
        logger.debug(f"Attempting to create environment at {self.env_path}")
        if self.env_path.exists():
            logger.info(f"Environment at {self.env_path} already exists, skipping.")
            return

        conda_exe = f"{self.get_conda_exe(update=True)}"
        with self.get_environment_file() as file:
            cmd = [
                conda_exe,
                "create",
                "--yes",
                "--file",
                f"{file}",
                "--prefix",
                f"{self.env_path}",
            ]
            logger.debug(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True, env=self.env_vars)  # noqa: S603

            if self.url is not None:
                logger.info(f"Installing development version of {self.slug} from {self.url}")
                cmd = [
                    conda_exe,
                    "run",
                    "--prefix",
                    f"{self.env_path}",
                    "pip",
                    "install",
                    "--no-deps",
                    self.url,
                ]
                logger.debug(f"Running {' '.join(cmd)}")
                subprocess.run(cmd, check=True, env=self.env_vars)  # noqa: S603

    def run(self, cmd: Iterable[str]) -> None:
        """
        Run a command.

        Parameters
        ----------
        cmd
            The command to run.

        Raises
        ------
        subprocess.CalledProcessError
            If the command fails

        """
        if not self.env_path.exists():
            msg = (
                f"Conda environment for provider `{self.slug}` not available at "
                f"{self.env_path}. Please install it by running the command "
                f"`ref providers create-env --provider {self.slug}`"
            )
            raise RuntimeError(msg)

        cmd = [
            f"{self.get_conda_exe(update=False)}",
            "run",
            "--prefix",
            f"{self.env_path}",
            *cmd,
        ]
        logger.info(f"Running '{' '.join(cmd)}'")
        try:
            # This captures the log output until the execution is complete
            # We could poll using `subprocess.Popen` if we want something more responsive
            res = subprocess.run(  # noqa: S603
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self.env_vars,
            )
            logger.info("Command output: \n" + res.stdout)
            logger.info("Command execution successful")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to run {cmd}")
            logger.error(e.stdout)
            raise e
