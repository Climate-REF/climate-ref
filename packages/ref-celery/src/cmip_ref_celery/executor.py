from loguru import logger

from cmip_ref.config import Config
from cmip_ref.models import MetricExecutionResult
from cmip_ref_celery.app import app
from cmip_ref_celery.tasks import generate_task_name
from cmip_ref_core.exceptions import InvalidExecutorException
from cmip_ref_core.executor import ExecutionCompletedCallback, Executor
from cmip_ref_core.metrics import Metric, ProposedMetricExecutionDefinition
from cmip_ref_core.providers import MetricsProvider


class CeleryExecutor(Executor):
    """
    Run a metric asynchronously

    Celery is an asynchronous task queue/job queue based on distributed message passing.
    Celery uses a message broker to distribute tasks across a cluster of worker nodes.
    The worker nodes are responsible for executing the tasks.
    The message broker used in this case is [Redis](https://github.com/redis/redis).

    The worker node may be running on the same machine as the client or on a different machine,
    either natively or via a docker container.

    This interface is still a work in progress.
    We cannot resume tasks that are in progress if the process terminates.
    That should be possible tracking some additional state in the database.
    """

    name = "celery"
    is_async = True

    def __init__(self, config: Config | None = None):
        self.config = Config.default() if config is None else config
        self._callbacks: dict[str, ExecutionCompletedCallback] = {}

        if self.config.paths.allow_out_of_tree_datasets:
            raise InvalidExecutorException(self, "CeleryExecutor does not support out of tree datasets")

    def run_metric(
        self,
        provider: MetricsProvider,
        metric: Metric,
        definition: ProposedMetricExecutionDefinition,
        metric_execution_result: MetricExecutionResult | None = None,
    ) -> None:
        """
        Run a metric calculation asynchronously

        This will queue the metric to be run by a Celery worker.
        The results will be stored in the database when the task completes if `metric_execution_result`
        is specified.

        No result will be returned from this function.
        Instead, you can peridically check the status of the task in the database.

        Parameters
        ----------
        provider
            Provider for the metric
        metric
            Metric to run
        definition
            A description of the information needed for this execution of the metric

            This include relative paths to the data files,
            which will be converted to absolute paths when being executed
        metric_execution_result
            Result of the metric execution

            This is a database object that contains the results of the execution.
            If provided, it will be updated with the results of the execution.
            This may happen asynchronously, so the results may not be immediately available.
        """
        from cmip_ref_celery.worker_tasks import handle_result

        name = generate_task_name(provider, metric)

        async_result = app.send_task(
            name,
            args=[
                definition,
            ],
            queue=provider.slug,
            link=handle_result.s(metric_execution_result.id).set(queue="celery")
            if metric_execution_result
            else None,
        )
        logger.debug(f"Celery task {async_result.id} submitted")
