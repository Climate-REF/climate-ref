import concurrent.futures
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from loguru import logger
from tqdm import tqdm

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Execution
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import ExecutionError
from climate_ref_core.executor import execute_locally
from climate_ref_core.logging import initialise_logging

from .result_handling import (
    ExecutionFuture,
    mark_execution_failed,
    process_result,
)

__all__ = [
    "ExecutionFuture",
    "LocalExecutor",
    "execute_locally",
    "process_result",
]


def _process_initialiser() -> None:  # pragma: no cover
    # Setup the logging for the process
    # This replaces the loguru default handler
    try:
        logger.remove()
        config = Config.default()
        initialise_logging(
            level=config.log_level,
            format=config.log_format,
            log_directory=config.paths.log,
        )
    except Exception as e:
        # Don't raise an exception here as that would kill the process pool
        # We want to log the error and continue
        logger.error(f"Failed to add log handler: {e}")


def _process_run(definition: ExecutionDefinition, log_level: str) -> ExecutionResult:
    # This is a catch-all for any exceptions that occur in the process
    try:
        return execute_locally(definition=definition, log_level=log_level)
    except Exception:  # pragma: no cover
        # This isn't expected but if it happens we want to log the error before the process exits
        # Mark as retryable since this is an infrastructure-level failure
        logger.exception(f"Error running execution {definition.execution_slug()}")
        return ExecutionResult.build_from_failure(definition, retryable=True)


class LocalExecutor:
    """
    Run a diagnostic locally using a process pool.

    This performs the diagnostic executions in parallel using different processes.
    The maximum number of processes is determined by the `n` parameter and default to the number of CPUs.

    This executor is the default executor and is used when no other executor is specified.
    """

    name = "local"

    def __init__(
        self,
        *,
        database: Database | None = None,
        config: Config | None = None,
        n: int | None = None,
        pool: concurrent.futures.Executor | None = None,
        task_timeout: float = 6 * 60 * 60,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = Config.default()
        if database is None:
            database = Database.from_config(config, run_migrations=False)
        self.n = n

        self.database = database
        self.config = config

        # Per-task wall-clock budget (default 6 hours, matching the Celery
        # task_time_limit). Diagnostics that hang past this are considered lost
        # so the pool can recycle the slot rather than blocking ``join`` forever.
        # Set to ``0`` to disable.
        self.task_timeout = task_timeout

        if pool is not None:
            self.pool = pool
        else:
            self.pool = ProcessPoolExecutor(
                max_workers=n,
                initializer=_process_initialiser,
                # Explicitly set the context to "spawn" to avoid issues with hanging on MacOS
                mp_context=multiprocessing.get_context("spawn"),
            )
        self._results: list[ExecutionFuture] = []

    def run(
        self,
        definition: ExecutionDefinition,
        execution: Execution | None = None,
    ) -> None:
        """
        Run a diagnostic in process

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic
        execution
            A database model representing the execution of the diagnostic.
            If provided, the result will be updated in the database when completed.
        """
        # Submit the execution to the process pool
        # and track the future so we can wait for it to complete
        future = self.pool.submit(
            _process_run,
            definition=definition,
            log_level=self.config.log_level,
        )
        self._results.append(
            ExecutionFuture(
                future=future,
                definition=definition,
                execution_id=execution.id if execution else None,
                submitted_at=time.time(),
            )
        )

    def join(self, timeout: float) -> None:
        """
        Wait for all diagnostics to finish

        This will block until all diagnostics have completed or the timeout is reached.
        Each individual execution is also bounded by ``self.task_timeout`` so that a
        hung diagnostic cannot block the pool indefinitely. Outstanding executions
        are always marked as failed-retryable before this method returns or raises,
        so the next solve can pick them up rather than seeing them stuck with
        ``successful=None``.

        Parameters
        ----------
        timeout
            Overall wall-clock timeout in seconds for the whole join

        Raises
        ------
        TimeoutError
            If the overall timeout is reached
        """
        start_time = time.time()
        refresh_time = 0.5  # Time to wait between checking for completed tasks in seconds

        results = self._results
        t = tqdm(total=len(results), desc="Waiting for executions to complete", unit="execution")

        try:
            while results:
                now = time.time()

                # Iterate over a copy of the list and remove finished tasks
                for result in results[:]:
                    if result.future.done():
                        try:
                            execution_result = result.future.result(timeout=0)
                        except Exception as e:
                            # Something went wrong when attempting to run the execution
                            # This is likely a failure in the execution itself not the diagnostic
                            self._mark_failed(result, retryable=True)
                            results.remove(result)
                            raise ExecutionError(
                                f"Failed to execute {result.definition.execution_slug()!r}"
                            ) from e

                        assert execution_result is not None, "Execution result should not be None"
                        assert isinstance(execution_result, ExecutionResult), (
                            "Execution result should be of type ExecutionResult"
                        )

                        # Process the result in the main process
                        # The results should be committed after each execution
                        with self.database.session.begin():
                            execution = (
                                self.database.session.get(Execution, result.execution_id)
                                if result.execution_id
                                else None
                            )
                            process_result(self.config, self.database, execution_result, execution)
                        logger.debug(f"Execution completed: {result}")
                        t.update(n=1)
                        results.remove(result)
                        continue

                    # Per-task timeout: a runaway diagnostic cannot block the pool
                    # forever. Cancel its future and mark the row failed-retryable.
                    if (
                        self.task_timeout > 0
                        and result.submitted_at > 0
                        and now - result.submitted_at > self.task_timeout
                    ):
                        logger.error(
                            f"Execution {result.definition.execution_slug()!r} exceeded per-task "
                            f"timeout of {self.task_timeout}s; marking failed-retryable"
                        )
                        result.future.cancel()
                        self._mark_failed(result, retryable=True)
                        t.update(n=1)
                        results.remove(result)

                # Break early to avoid waiting for one more sleep cycle
                if len(results) == 0:
                    break

                elapsed_time = time.time() - start_time

                if elapsed_time > timeout:
                    self._fail_outstanding(results, t)
                    self.pool.shutdown(wait=False, cancel_futures=True)
                    raise TimeoutError("Not all tasks completed within the specified timeout")

                # Wait for a short time before checking for completed executions
                time.sleep(refresh_time)
        finally:
            t.close()

        logger.info("All executions completed successfully")

    def _mark_failed(self, result: ExecutionFuture, *, retryable: bool) -> None:
        mark_execution_failed(
            self.database,
            self.config,
            result.definition,
            result.execution_id,
            retryable=retryable,
        )

    def _fail_outstanding(self, results: list[ExecutionFuture], progress: Any) -> None:
        for outstanding in list(results):
            logger.warning(
                f"Execution {outstanding.definition.execution_slug()} did not complete within the timeout"
            )
            self._mark_failed(outstanding, retryable=True)
            progress.update(n=1)
            results.remove(outstanding)
