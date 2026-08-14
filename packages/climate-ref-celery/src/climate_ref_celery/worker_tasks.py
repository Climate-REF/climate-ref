"""
Celery worker tasks for handling diagnostic execution executions.
"""

import os

from celery import current_app
from celery.signals import worker_process_shutdown
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.executor import handle_execution_result
from climate_ref.models import Execution
from climate_ref_core.diagnostics import ExecutionResult


class _WorkerDatabase:
    """
    The database a worker process uses, opened once and reused across tasks

    A worker handles many tasks, so opening an engine per task churns a connection each time.
    The record of which process opened it means a forked child opens its own
    rather than using connections it inherited, which are not safe to share across processes.
    """

    def __init__(self) -> None:
        self._opened: tuple[int, str, Database] | None = None

    def get(self, config: Config) -> Database:
        """
        Open this process's database, or return the one already open for this URL

        Parameters
        ----------
        config
            REF configuration describing where the database lives.

        Returns
        -------
        :
            The database for this process.
        """
        key = (os.getpid(), config.db.database_url)
        if self._opened is not None and self._opened[:2] == key:
            return self._opened[2]

        self.close()
        self._opened = (*key, Database.from_config(config, run_migrations=False))
        return self._opened[2]

    def close(self) -> None:
        """
        Release the connections held by this process

        A database opened by another process is dropped rather than closed,
        because its connections belong to the process that opened them.
        """
        if self._opened is not None and self._opened[0] == os.getpid():
            self._opened[2].close()
        self._opened = None


_worker_database = _WorkerDatabase()


@worker_process_shutdown.connect
def _close_worker_database(**kwargs: object) -> None:  # pragma: no cover
    """Release this process's database connections as the worker process goes away."""
    _worker_database.close()


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
    db = _worker_database.get(config)

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
    db = _worker_database.get(config)

    with db.session.begin():
        execution = db.session.get(Execution, execution_id)

        if execution is None:
            logger.error(f"Execution {execution_id} not found")
            return

        execution.mark_failed()
        # Deliberately not clearing dirty - this is a system-level failure
        # (worker killed, OOM, time limit) so the execution should be retried
        logger.info(f"Marked execution {execution_id} as failed (retryable)")
