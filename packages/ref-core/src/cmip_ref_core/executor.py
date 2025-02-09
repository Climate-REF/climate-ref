from collections.abc import Callable
from typing import Protocol, runtime_checkable

from cmip_ref.models import MetricExecutionResult
from cmip_ref_core.metrics import (
    Metric,
    MetricResult,
    ProposedMetricExecutionDefinition,
)
from cmip_ref_core.providers import MetricsProvider

ExecutionCompletedCallback = Callable[[ProposedMetricExecutionDefinition, MetricResult], None]
"""
Callback for when an execution is completed

This will be called when an execution is completed, either successfully or with an error.
"""


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
    is_async: bool

    def run_metric(
        self,
        provider: MetricsProvider,
        metric: Metric,
        definition: ProposedMetricExecutionDefinition,
        metric_execution_result: MetricExecutionResult | None = None,
    ) -> MetricResult | None:
        """
        Execute a metric

        Parameters
        ----------
        provider
            Provider for the metric
        metric
            Metric to run
        definition
            Definition of the information needed to execute a metric

            This definition needs to be translated into a concrete `MetricExecutionDefinition`
            before running the metric.

            This definition describes which datasets are required to run the metric and where
            the output should be stored.
        metric_execution_result
            Result of the metric execution

            This is a database object that contains the results of the execution.
            If provided, it will be updated with the results of the execution.
            This may happen asynchronously, so the results may not be immediately available.

        Returns
        -------
        :
            If the executor is synchronous, the result of the execution will be returned directly,
            otherwise, this is return None and the result will need to be retrieved from the
            metric_execution_result object.

            /// admonition | Note
            In future, we may return a `Future` object that can be used to retrieve the result,
            but that requires some additional work to implement.
            ///
        """
        ...
