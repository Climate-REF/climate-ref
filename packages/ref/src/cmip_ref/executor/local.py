from loguru import logger

from cmip_ref.config import Config
from cmip_ref.models import MetricExecutionResult
from cmip_ref_core.metrics import Metric, MetricResult, ProposedMetricExecutionDefinition


class LocalExecutor:
    """
    Run a metric locally, in-process.

    This is mainly useful for debugging and testing.
    The production executor will run the metric in a separate process or container,
    the exact manner of which is yet to be determined.
    """

    name = "local"
    is_async = False

    def __init__(self, config: Config | None = None):
        self.config = Config.default() if config is None else config

    def run_metric(
        self,
        metric: Metric,
        definition: ProposedMetricExecutionDefinition,
        metric_execution_result: MetricExecutionResult | None = None,
    ) -> MetricResult:
        """
        Run a metric in process

        Parameters
        ----------
        metric
            Metric to run
        definition
            A description of the information needed for this execution of the metric

        Returns
        -------
        :
            Results from running the metric
        """
        concrete_definition = definition.to_metric_execution_definition(
            data_directory=self.config.paths.data,
            scratch_directory=self.config.paths.scratch,
        )
        concrete_definition.output_directory.mkdir(parents=True, exist_ok=True)

        try:
            return metric.run(definition=concrete_definition)
            # TODO: Copy results to the output directory
        except Exception:
            logger.exception(f"Error running metric {metric.slug}")
            return MetricResult.build_from_failure(concrete_definition)
