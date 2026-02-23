"""
Celery worker tasks for handling diagnostic execution executions.
"""

from celery import current_app
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.executor import handle_execution_result
from climate_ref.models import Execution
from climate_ref_core.diagnostics import ExecutionResult


@current_app.task(max_retries=0)
def handle_result(result: ExecutionResult, execution_id: int) -> None:
    """
    Handle the result of a diagnostic execution

    This function is called when a diagnostic execution is completed successfully.

    Parameters
    ----------
    result
        The result of the diagnostic execution
    execution_id
        The unique identifier for the diagnostic execution
    """
    logger.info(f"Handling result for execution {execution_id} + {result}")

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    with db.session.begin():
        execution = db.session.get(Execution, execution_id)

        if execution is None:
            logger.error(f"Execution {execution_id} not found")
            return

        handle_execution_result(config, db, execution, result)


@current_app.task(max_retries=0)
def handle_failure(task_id: str, execution_id: int) -> None:
    """
    Handle a failed or killed diagnostic task

    This is called via ``link_error`` when the diagnostic task fails, is killed
    by a time limit, or the worker process is lost.

    It marks the corresponding ``Execution`` row as failed
    so it does not remain in an indeterminate state.

    Since this callback is triggered by infrastructure-level failures
    (worker crash, OOM kill, time limit), the execution group's dirty flag
    is left as-is so the execution will be retried on the next solve.

    Parameters
    ----------
    task_id
        The Celery task UUID of the failed task
    execution_id
        The unique identifier for the diagnostic execution
    """
    logger.error(
        f"Task {task_id} failed for execution {execution_id} "
        f"(system-level failure, will be retried on next solve)"
    )

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    with db.session.begin():
        execution = db.session.get(Execution, execution_id)

        if execution is None:
            logger.error(f"Execution {execution_id} not found")
            return

        execution.mark_failed()
        # Deliberately not clearing dirty - this is a system-level failure
        # (worker killed, OOM, time limit) so the execution should be retried
        logger.info(f"Marked execution {execution_id} as failed (retryable)")
