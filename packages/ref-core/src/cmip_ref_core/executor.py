from typing import Protocol, runtime_checkable

from cmip_ref_core.metrics import (
    Metric,
    MetricResult,
    ProposedMetricExecutionDefinition,
)


@runtime_checkable
class Executor(Protocol):
    """
    An executor is responsible for running a metric.

    The metric may be run locally in the same process or in a separate process or container.

    Notes
    -----
    This is an extremely basic interface and will be expanded in the future, as we figure out
    our requirements.
    """

    name: str

    def run_metric(self, metric: Metric, definition: ProposedMetricExecutionDefinition) -> MetricResult:
        """
        Execute a metric

        Parameters
        ----------
        metric
            Metric to run
        definition
            Definition of the information needed to execute a metric

            This definition needs to be translated into a concrete `MetricExecutionDefinition`
            before running the metric.

            This definition describes which datasets are required to run the metric and where
            the output should be stored.

        Returns
        -------
        :
            Results from running the metric
        """
        ...
