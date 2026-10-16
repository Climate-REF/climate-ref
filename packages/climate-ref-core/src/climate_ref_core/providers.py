"""
Interface for declaring a diagnostic provider.

This defines how diagnostic packages interoperate with the REF framework.
Each diagnostic package may contain multiple diagnostics.

Each diagnostic package must implement the `DiagnosticProvider` interface.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import importlib.metadata
import importlib.resources
import os
import stat
import subprocess
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import requests
import yaml
from attrs import evolve
from loguru import logger

from climate_ref_core.constraints import IgnoreFacets
from climate_ref_core.data import DataResourceError
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.exceptions import (
    CondaCommandError,
    InvalidDiagnosticException,
    InvalidProviderException,
)
from climate_ref_core.logging import execution_log_only


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

    def configure(self, config: Any) -> None:
        """
        Configure the provider.

        Parameters
        ----------
        config :
            A configuration.
        """
        grey_list = config.ignore_datasets_resource

        # The format of the configuration file is:
        # provider:
        #   diagnostic:
        #     source_type:
        #       - facet: value
        #       - other_facet: [other_value1, other_value2]
        #
        # The packaged copy is tried second so a mistyped `ignore_datasets_file`
        # cannot silently drop the grey list protections.
        ignore_datasets_all: dict[str, Any] = {}
        source = "nothing"
        for candidate in (grey_list, grey_list.packaged):
            try:
                ignore_datasets_all = yaml.safe_load(candidate.read_text()) or {}
            except (DataResourceError, yaml.YAMLError) as exc:
                logger.warning(f"Could not read the grey list from {candidate.describe()}: {exc}")
                continue
            if not isinstance(ignore_datasets_all, dict):
                logger.warning(f"The grey list at {candidate.describe()} is not a mapping, ignoring it")
                ignore_datasets_all = {}
                continue
            source = candidate.describe()
            break
        else:
            logger.warning(f"No grey list could be read for provider {self.slug}, ignoring no datasets")
        logger.debug(f"Configuring provider {self.slug} using the grey list from {source}")

        ignore_datasets = ignore_datasets_all.get(self.slug, {})
        if unknown_slugs := {slug for slug in ignore_datasets} - {d.slug for d in self.diagnostics()}:
            logger.warning(
                f"Unknown diagnostics found in {source} "
                f"for provider {self.slug}: {', '.join(sorted(unknown_slugs))}"
            )

        known_source_types = {s.value for s in iter(SourceDatasetType)}
        for diagnostic in self.diagnostics():
            if diagnostic.slug in ignore_datasets:
                if unknown_source_types := set(ignore_datasets[diagnostic.slug]) - known_source_types:
                    logger.warning(
                        f"Unknown source types found in {source} for "
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

    def setup(
        self,
        config: Any,
        *,
        db: Any = None,
        skip_env: bool = False,
        skip_data: bool = False,
    ) -> None:
        """
        Perform all setup required before offline execution.

        This calls setup_environment, fetch_data, and ingest_data in the correct order.
        Override individual hooks for fine-grained control.

        This method MUST be idempotent - safe to call multiple times.

        Parameters
        ----------
        config
            The application configuration
        db
            Optional database instance for data ingestion.

            If None, ingestion is skipped. This allows providers to be set up without the full
            climate-ref package (e.g., for environment setup or data fetching only).
        skip_env
            If True, skip environment setup (e.g., conda)
        skip_data
            If True, skip data fetching and ingestion
        """
        if not skip_env:
            self.setup_environment(config)
        if not skip_data:
            self.fetch_data(config)
            if db is not None:
                self.ingest_data(config, db)

    def setup_environment(self, config: Any) -> None:
        """
        Set up the execution environment (e.g., conda environment).

        Default implementation does nothing. Override in subclasses
        that require environment setup.

        This method MUST be idempotent.

        Parameters
        ----------
        config
            The application configuration
        """
        pass

    def fetch_data(self, config: Any) -> None:
        """
        Fetch all data required for offline execution.

        This includes reference datasets, climatology files, map files,
        recipes, or any other data the provider needs.

        Default implementation does nothing. Override in subclasses
        that require data fetching. Providers are responsible for
        determining what data they need and how to fetch it.

        Data should be downloaded to the pooch cache (via `fetch_all_files`
        with `output_dir=None`). Diagnostics can then access data via
        `registry.abspath`.

        This method MUST be idempotent.

        Parameters
        ----------
        config
            The application configuration
        """
        pass

    def ingest_data(self, config: Any, db: Any) -> None:
        """
        Ingest fetched data into the database.

        This is called after fetch_data to register any provider-specific
        datasets in the database. For example, PMP climatology data needs
        to be ingested so it can be used by diagnostics.

        Default implementation does nothing. Override in subclasses
        that have data to ingest.

        This method MUST be idempotent.

        Note: This method is only called when a database instance is provided
        to setup(). The database and ingestion utilities are part of the
        climate-ref package, not climate-ref-core. Provider implementations
        should handle ImportError gracefully if they need to import from
        climate-ref, allowing the provider to work standalone without
        the full climate-ref package installed.

        Parameters
        ----------
        config
            The application configuration
        db
            The database instance (from climate-ref package)
        """
        pass

    def validate_setup(self, config: Any) -> bool:
        """
        Validate that the provider is ready for offline execution.

        Returns True if setup is complete and valid, False otherwise.
        The default implementation logs and fails on any `check_health` problem.

        Parameters
        ----------
        config
            The application configuration

        Returns
        -------
        bool
            True if setup is valid and complete
        """
        problems = self.check_health(config)
        for problem in problems:
            logger.error(problem)
        return not problems

    def check_health(self, config: Any) -> list[str]:
        """
        Check that the provider can launch a diagnostic on this host right now.

        Unlike `validate_setup`, this must be cheap enough to run before every solve.
        The default implementation reports no problems.

        Parameters
        ----------
        config
            The application configuration

        Returns
        -------
        :
            A description of each problem found, empty when the provider is healthy
        """
        return []

    def get_data_path(self) -> Path | None:
        """
        Get the path where this provider's data is cached.

        Returns
        -------
        Path | None
            The data cache path, or None if the provider doesn't use cached data.
        """
        return None


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


@functools.cache
def provider_by_slug(slug: str) -> DiagnosticProvider:
    """
    Look up a registered provider by its slug

    Entry points can be enumerated without importing them,
    so only the provider that matches is ever loaded.
    The loaded provider is the module-level singleton,
    so it carries whatever configuration the current process has already applied to it.

    Parameters
    ----------
    slug
        Slug of the provider, for example `esmvaltool`.

    Raises
    ------
    InvalidProviderException
        If no registered provider has the given slug.

    Returns
    -------
    :
        DiagnosticProvider instance
    """
    entry_points = list(importlib.metadata.entry_points(group="climate-ref.providers"))

    # The entry point name is the provider slug by convention, so the name match is tried first
    # and the rest are only a fallback for a provider whose name and slug disagree.
    named_first = [ep for ep in entry_points if ep.name == slug]
    named_first += [ep for ep in entry_points if ep.name != slug]

    # A non-matching provider that fails to import is skipped rather than masking the one we are looking for,
    # while an import failure on the named provider itself propagates.
    for entry_point in named_first:
        try:
            provider = import_provider(entry_point.value)
        except InvalidProviderException:
            if entry_point.name == slug:
                raise
            logger.debug(f"Skipping provider {entry_point.name!r} while looking for {slug!r}")
            continue
        if provider.slug == slug:
            return provider

    available = ", ".join(sorted(ep.name for ep in entry_points)) or "[]"
    raise InvalidProviderException(slug, f"No provider with slug {slug!r}. Available: {available}")


def resolve_diagnostic(full_slug: str) -> Diagnostic:
    """
    Look up a diagnostic from its full slug

    Parameters
    ----------
    full_slug
        Slug of the form `{provider_slug}/{diagnostic_slug}`, as produced by
        [Diagnostic.full_slug][climate_ref_core.diagnostics.Diagnostic.full_slug].

    Raises
    ------
    InvalidProviderException
        If the slug is malformed or names a provider that is not installed.
    KeyError
        If the provider does not supply a diagnostic with that slug.

    Returns
    -------
    :
        The diagnostic registered with the named provider.
    """
    provider_slug, separator, diagnostic_slug = full_slug.partition("/")
    if not separator:
        raise InvalidProviderException(full_slug, "Expected a slug of the form 'provider/diagnostic'")

    return provider_by_slug(provider_slug).get(diagnostic_slug)


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

COMMAND_OUTPUT_TAIL_LINES = 50
"""Lines of a failed command's output that are logged to the console."""

MICROMAMBA_FAILURE_PREFIX = "critical libmamba"
"""Output line prefix written when micromamba itself fails, before the wrapped command runs."""


HEALTH_CHECK_TIMEOUT_SECONDS = 300
"""Upper bound on a provider health check, which can be slow on a cold network filesystem."""


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

    Attributes
    ----------
    env_overrides
        Environment variables overriding the process environment
        for commands run in the conda environment.
    pip_packages
        Pip packages to install (as URLs) with ``--no-deps --force-reinstall``
        after creating the conda environment.

    """

    def __init__(
        self,
        name: str,
        version: str,
        slug: str | None = None,
    ) -> None:
        super().__init__(name, version, slug)
        self._conda_exe: Path | None = None
        self._prefix: Path | None = None
        self.pip_packages: list[str] = []
        self.env_overrides: dict[str, str] = {}

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

    def _launch_env(self, *, use_lockfiles: bool = True) -> dict[str, str]:
        """
        Environment for a command launched in the conda environment.

        The live process environment is the base,
        so settings applied to a worker reach every command it launches,
        with the provider's overrides on top.
        HOME falls back to the conda prefix so micromamba has a writable directory.

        Parameters
        ----------
        use_lockfiles
            Whether micromamba should take its file locks.
            Disable this for commands that only read the environment.
            An explicit ``MAMBA_USE_LOCKFILES`` in the process environment
            or in `env_overrides` always wins.
        """
        env = {**os.environ, **self.env_overrides}
        if "HOME" not in env:
            env["HOME"] = str(self.prefix)
        if not use_lockfiles:
            env.setdefault("MAMBA_USE_LOCKFILES", "false")
        return env

    @property
    def conda_exe_path(self) -> Path:
        """
        Where the conda executable lives, whether or not it has been installed yet.

        Use this when only the location is needed.
        `get_conda_exe` installs the executable, which requires network access.
        """
        return self.prefix / "micromamba"

    def configure(self, config: Any) -> None:
        """Configure the provider."""
        super().configure(config)
        self.prefix = config.paths.software / "conda"

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
        conda_exe = self.conda_exe_path

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
            for pkg in self.pip_packages:
                suffix.update(bytes(pkg, encoding="utf-8"))
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
            subprocess.run(cmd, check=True, env=self._launch_env())  # noqa: S603

            for pkg in self.pip_packages:
                logger.info(f"Installing development version from {pkg}")
                cmd = [
                    conda_exe,
                    "run",
                    "--prefix",
                    f"{self.env_path}",
                    "pip",
                    "install",
                    "--no-deps",
                    "--force-reinstall",
                    pkg,
                ]
                logger.debug(f"Running {' '.join(cmd)}")
                subprocess.run(cmd, check=True, env=self._launch_env())  # noqa: S603

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
        env_path = self.env_path
        if not env_path.exists():
            msg = (
                f"Conda environment for provider `{self.slug}` not available at "
                f"{env_path}. Please install it by running the command "
                f"`ref providers create-env --provider {self.slug}`"
            )
            raise RuntimeError(msg)

        cmd = self._run_command(self.get_conda_exe(update=False), env_path, cmd)
        logger.info(f"Running '{' '.join(cmd)}'")
        try:
            # This captures the log output until the execution is complete
            # We could poll using `subprocess.Popen` if we want something more responsive
            res = self._launch(cmd)
            logger.info("Command output: \n" + res.stdout)
            logger.info("Command execution successful")
        except subprocess.CalledProcessError as e:
            output = e.stdout or ""
            message = f"Command exited with code {e.returncode}: {cmd}"
            # The full output goes to the execution log, the tail also to the console
            execution_log_only().error("Command output: \n" + output)
            tail = "\n".join(output.splitlines()[-COMMAND_OUTPUT_TAIL_LINES:])
            logger.error(f"{message}\nLast lines of output:\n{tail}")
            raise CondaCommandError(
                message,
                stdout=e.stdout,
                stderr=e.stderr,
                launch_failed=any(line.startswith(MICROMAMBA_FAILURE_PREFIX) for line in output.splitlines()),
            ) from e

    @staticmethod
    def _run_command(conda_exe: Path, env_path: Path, cmd: Iterable[str]) -> list[str]:
        return [str(conda_exe), "run", "--prefix", str(env_path), *cmd]

    def _launch(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=self._launch_env(use_lockfiles=False),
            **kwargs,
        )

    def setup_environment(self, config: Any) -> None:
        """Set up the conda environment."""
        self.create_env()

    def check_health(self, config: Any) -> list[str]:
        """
        Check that a command can be launched inside the conda environment.

        This runs a trivial command through `micromamba run`,
        the same way `run` launches a diagnostic,
        so it catches a missing environment or a launcher that fails before the diagnostic starts.
        """
        remedy = f"Run `ref providers setup --provider {self.slug}` to install it."
        conda_exe = self.conda_exe_path
        env_path = self.env_path
        if not conda_exe.exists():
            return [f"micromamba is not available at {conda_exe}. {remedy}"]
        if not env_path.exists():
            return [f"Conda environment for {self.slug} is not available at {env_path}. {remedy}"]

        cmd = self._run_command(conda_exe, env_path, ["python", "-c", "pass"])
        try:
            self._launch(cmd, timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        except subprocess.CalledProcessError as e:
            return [f"`{' '.join(cmd)}` failed with exit code {e.returncode}:\n{e.stdout}"]
        except subprocess.TimeoutExpired:
            return [f"`{' '.join(cmd)}` did not finish within {HEALTH_CHECK_TIMEOUT_SECONDS}s"]
        except OSError as e:
            return [f"`{' '.join(cmd)}` could not be launched: {e}"]
        return []
