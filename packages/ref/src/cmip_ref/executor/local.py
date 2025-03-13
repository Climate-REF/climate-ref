from typing import Any

from loguru import logger

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.executor import handle_execution_result
from cmip_ref.models import MetricExecutionResult as MetricExecutionResultModel
from cmip_ref_core.metrics import Metric, MetricExecutionDefinition, MetricExecutionResult
from cmip_ref_core.providers import MetricsProvider


class LocalExecutor:
    """
    Run a metric locally, in-process.

    This is mainly useful for debugging and testing.
    The production executor will run the metric in a separate process or container,
    the exact manner of which is yet to be determined.
    """

    name = "local"

    def __init__(
        self, *, database: Database | None = None, config: Config | None = None, **kwargs: Any
    ) -> None:
        if config is None:
            config = Config.default()
        if database is None:
            database = Database.from_config(config, run_migrations=False)

        self.database = database
        self.config = config

    def run_metric(
        self,
        provider: MetricsProvider,
        metric: Metric,
        definition: MetricExecutionDefinition,
        metric_execution_result: MetricExecutionResultModel | None = None,
    ) -> None:
        """
        Run a metric in process

        Parameters
        ----------
        provider
            The provider of the metric
        metric
            Metric to run
        definition
            A description of the information needed for this execution of the metric
        metric_execution_result
            A database model representing the execution of the metric.
            If provided, the result will be updated in the database when completed.
        """
        definition.output_directory.mkdir(parents=True, exist_ok=True)

        try:
            result = metric.run(definition=definition)
        except Exception:
            logger.exception(f"Error running metric {metric.slug}")
            result = MetricExecutionResult.build_from_failure(definition)

        if metric_execution_result:
            handle_execution_result(self.config, self.database, metric_execution_result, result)

    def join(self, timeout: float) -> None:
        """
        Wait for all metrics to finish

        This returns immediately because the local executor runs metrics synchronously.

        Parameters
        ----------
        timeout
            Timeout in seconds (Not used)
        """
        return
