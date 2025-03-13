"""
Interfaces for metrics providers.

This defines how metrics packages interoperate with the REF framework.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib
import os
import stat
import subprocess
from abc import abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from loguru import logger

from cmip_ref_core.exceptions import InvalidMetricException, InvalidProviderException
from cmip_ref_core.metrics import Metric

if TYPE_CHECKING:
    from cmip_ref.config import Config


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


class MetricsProvider:
    """
    Interface for that a metrics provider must implement.

    This provides a consistent interface to multiple different metrics packages.
    """

    def __init__(self, name: str, version: str, slug: str | None = None) -> None:
        self.name = name
        self.slug = slug or _slugify(name)
        self.version = version

        self._metrics: dict[str, Metric] = {}

    def configure(self, config: Config) -> None:
        """
        Configure the provider.

        Parameters
        ----------
        config :
            A configuration.
        """

    def metrics(self) -> list[Metric]:
        """
        Iterate over the available metrics for the provider.

        Returns
        -------
        :
            Iterator over the currently registered metrics.
        """
        return list(self._metrics.values())

    def __len__(self) -> int:
        return len(self._metrics)

    def register(self, metric: Metric) -> None:
        """
        Register a metric with the manager.

        Parameters
        ----------
        metric :
            The metric to register.
        """
        if not isinstance(metric, Metric):
            raise InvalidMetricException(metric, "Metrics must be an instance of the 'Metric' class")
        metric.provider = self
        self._metrics[metric.slug.lower()] = metric

    def get(self, slug: str) -> Metric:
        """
        Get a metric by name.

        Parameters
        ----------
        slug :
            Name of the metric (case-sensitive).

        Raises
        ------
        KeyError
            If the metric with the given name is not found.

        Returns
        -------
        Metric
            The requested metric.
        """
        return self._metrics[slug.lower()]


def import_provider(fqn: str) -> MetricsProvider:
    """
    Import a provider by name

    Parameters
    ----------
    fqn
        Full package and attribute name of the provider to import

        For example: `cmip_ref_metrics_example.provider` will use the `provider` attribute from the
        `cmip_ref_metrics_example` package.

        If only a package name is provided, the default attribute name is `provider`.

    Raises
    ------
    InvalidProviderException
        If the provider cannot be imported

        If the provider isn't a valid `MetricsProvider`.

    Returns
    -------
    :
        MetricsProvider instance
    """
    if "." in fqn:
        module, name = fqn.rsplit(".", 1)
    else:
        module = fqn
        name = "provider"

    try:
        imp = importlib.import_module(module)
        provider = getattr(imp, name)
        if not isinstance(provider, MetricsProvider):
            raise InvalidProviderException(fqn, f"Expected MetricsProvider, got {type(provider)}")
        return provider
    except ModuleNotFoundError:
        logger.error(f"Module '{fqn}' not found")
        raise InvalidProviderException(fqn, f"Module '{module}' not found")
    except AttributeError:
        logger.error(f"Provider '{fqn}' not found")
        raise InvalidProviderException(fqn, f"Provider '{name}' not found in {module}")


class CommandLineMetricsProvider(MetricsProvider):
    """
    A metrics provider for metrics that can be run from the command line.
    """

    @abstractmethod
    def run(self, cmd: Iterable[str]) -> None:
        """
        Return the command to run.
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


class CondaMetricsProvider(CommandLineMetricsProvider):
    """
    A provider for metrics that can be run from the command line in a conda environment.
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
        self.prefix = config.paths.software / "conda"

    def _install_conda(self, update: bool) -> Path:
        """Install micromamba in a temporary location.

        Parameters
        ----------
        update:
            Update the micromamba executable if it is older than a week.

        Returns
        -------
            The path to the executable.

        """
        conda_exe = self.prefix / "micromamba"

        if conda_exe.exists() and update:
            # Only update if the executable is older than `MICROMAMBA_MAX_AGE`.
            creation_time = datetime.datetime.fromtimestamp(conda_exe.stat().st_ctime)
            age = datetime.datetime.now() - creation_time
            if age < MICROMAMBA_MAX_AGE:
                update = False

        if not conda_exe.exists() or update:
            logger.info("Installing conda")
            self.prefix.mkdir(parents=True, exist_ok=True)
            response = requests.get(_get_micromamba_url(), timeout=120)
            response.raise_for_status()
            with conda_exe.open(mode="wb") as file:
                file.write(response.content)
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
        # first registered metric instead.
        metrics = self.metrics()
        if len(metrics) == 0:
            msg = "Unable to determine the provider module, please register a metric first."
            raise ValueError(msg)
        module = metrics[0].__module__.split(".")[0]
        lockfile = importlib.resources.files(module).joinpath("requirements").joinpath("conda-lock.yml")
        return importlib.resources.as_file(lockfile)

    @property
    def env_path(self) -> Path:
        """
        A unique path for storing the conda environment.
        """
        with self.get_environment_file() as file:
            suffix = hashlib.sha1(file.read_bytes(), usedforsecurity=False).hexdigest()
        return self.prefix / f"{self.slug}-{suffix}"

    def create_env(self) -> None:
        """
        Create a conda environment.
        """
        logger.debug(f"Attempting to create environment at {self.env_path}")
        if self.env_path.exists():
            logger.info(f"Environment at {self.env_path} already exists, skipping.")
            return

        with self.get_environment_file() as file:
            cmd = [
                f"{self.get_conda_exe(update=True)}",
                "create",
                "--yes",
                "--file",
                f"{file}",
                "--prefix",
                f"{self.env_path}",
            ]
            logger.debug(f"Running {' '.join(cmd)}")
            subprocess.run(cmd, check=True)  # noqa: S603

    def run(self, cmd: Iterable[str]) -> None:
        """
        Run a command.

        Parameters
        ----------
        cmd :
            The command to run.

        """
        self.create_env()

        cmd = [
            f"{self.get_conda_exe(update=False)}",
            "run",
            "--prefix",
            f"{self.env_path}",
            *cmd,
        ]
        logger.info(f"Running {' '.join(cmd)}")
        subprocess.run(cmd, check=True)  # noqa: S603
        logger.info(f"Successfully ran {cmd}")
