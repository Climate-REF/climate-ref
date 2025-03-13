from celery import current_app
from loguru import logger

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.executor import handle_execution_result
from cmip_ref.models import MetricExecutionResult as MetricExecutionResultModel
from cmip_ref_core.metrics import MetricExecutionResult


@current_app.task
def handle_result(result: MetricExecutionResult, metric_execution_result_id: int) -> None:
    """
    Handle the result of a metric execution

    This function is called when a metric execution is completed.

    Parameters
    ----------
    metric_execution_result_id
        The unique identifier for the metric execution
    result
        The result of the metric execution
    """
    logger.info(f"Handling result for metric execution {metric_execution_result_id} + {result}")

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    with db.session.begin():
        metric_execution_result = db.session.get(MetricExecutionResultModel, metric_execution_result_id)

        if metric_execution_result is None:
            logger.error(f"Metric execution result {metric_execution_result_id} not found")
            return

        handle_execution_result(config, db, metric_execution_result, result)
