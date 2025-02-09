from celery import current_app
from loguru import logger

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.models import MetricExecutionResult


@current_app.task
def handle_result(metric_execution_id: int, **kwargs) -> None:
    """
    Handle the result of a metric execution

    This function is called when a metric execution is completed.

    Parameters
    ----------
    metric_execution_id
        The unique identifier for the metric execution
    result
        The result of the metric execution
    """
    logger.info(f"Handling result for metric execution {metric_execution_id}")
    logger.info(f"Received: {kwargs}")

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    with db.session.begin():
        metric_execution_result = db.session.get(MetricExecutionResult, metric_execution_id)

        if metric_execution_result is None:
            logger.error(f"Metric execution result {metric_execution_id} not found")
            return
        logger.info(f"{metric_execution_result} found")
