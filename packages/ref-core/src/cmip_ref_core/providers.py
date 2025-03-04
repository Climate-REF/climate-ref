"""
Interfaces for metrics providers.

This defines how metrics packages interoperate with the REF framework.
"""

import hashlib
import importlib
import shutil
import subprocess
from abc import abstractmethod
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType

from loguru import logger

from cmip_ref_core.exceptions import InvalidMetricException, InvalidProviderException
from cmip_ref_core.metrics import Metric


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
        metric : Metric
            The metric to register.
        """
        if not isinstance(metric, Metric):
            raise InvalidMetricException(metric, "Metrics must be an instance of the 'Metric' class")
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
        provider.module = imp  # TODO: fix this hack
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


class CondaMetricsProvider(CommandLineMetricsProvider):
    """
    A provider for metrics that can be run from the command line in a conda environment.
    """

    prefix: Path = Path.cwd() / "envs"  # TODO: make this configurable
    module: ModuleType

    @property
    def conda_exe(self) -> Path:
        """
        The path to the conda executable.
        """
        if result := shutil.which("conda"):
            return Path(result)
        msg = "No conda executable available."
        raise ValueError(msg)

    def get_environment_file(self) -> AbstractContextManager[Path]:
        """
        Return a context manager that provides the environment file as a Path.
        """
        return importlib.resources.files(self.module).joinpath("requirements", "conda-lock.yml")

    @property
    def env_path(self) -> Path:
        """
        A unique path for storing the conda environment.
        """
        with self.get_environment_file() as file:
            suffix = hashlib.sha1(file.read_bytes(), usedforsecurity=False).hexdigest()
        return self.prefix / f"{self.name}-{self.version}-{suffix}"

    def create_env(self) -> None:
        """
        Create a conda environment.
        """
        if self.env_path.exists():
            logger.info(f"Environment at {self.env_path} already exists, skipping.")
            return

        with self.get_environment_file() as file:
            cmd = [
                f"{self.conda_exe}",
                "create",
                "--file",
                f"{file}",
                "--prefix",
                f"{self.env_path}",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603

    def run(self, cmd: Iterable[str]) -> None:
        """
        Run a command.

        Parameters
        ----------
        cmd :
            The command to run.

        """
        cmd = [
            f"{self.conda_exe}",
            "run",
            "--prefix",
            f"{self.env_path}",
            *cmd,
        ]
        logger.info(f"Running {cmd}")
        subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
        logger.info(f"Successfully ran {cmd}")
