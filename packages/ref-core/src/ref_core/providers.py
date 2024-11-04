"""
Interfaces for metrics providers.

This defines how metrics packages interoperate with the REF framework.
"""

from ref_core.metrics import Metric


class MetricsProvider:
    """
    Interface for that a metrics provider must implement.

    This provides a consistent interface to multiple different metrics packages.
    """

    def __init__(self, name: str, version: str) -> None:
        self.name = name
        self.version = version

        self._metrics: dict[str, Metric] = {}

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
            raise ValueError("Metric must be an instance of Metric")
        self._metrics[metric.name.lower()] = metric

    def __iter__(self):
        return iter(self._metrics.values())

    def get(self, name: str) -> Metric:
        """
        Get a metric by name.

        Parameters
        ----------
        name : str
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
        return self._metrics[name.lower()]
