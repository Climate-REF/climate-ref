from typing import Any

from ref_core.metrics import Configuration, Metric, MetricResult, TriggerInfo


class LocalExecutor:
    """
    Run a metric locally, in-process.

    This is mainly useful for debugging and testing.
    The production executor will run the metric in a separate process or container,
    the exact manner of which is yet to be determined.
    """

    name: str = "local"

    def run_metric(
        self, metric: Metric, configuration: Configuration, trigger: TriggerInfo | None, **kwargs: Any
    ) -> MetricResult:
        """
        Run a metric in process

        Parameters
        ----------
        metric
            Metric to run
        configuration
            Configuration to run the metric with
        trigger
            Information about the dataset that triggered the metric run
        kwargs
            Additional keyword arguments for the executor

        Returns
        -------
        :
            Results from running the metric
        """
        return metric.run(configuration=configuration, trigger=trigger)
