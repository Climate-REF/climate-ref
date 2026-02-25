import concurrent.futures
import multiprocessing
import time
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from attrs import define
from loguru import logger
from tqdm import tqdm

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Execution
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import ExecutionError
from climate_ref_core.executor import execute_locally
from climate_ref_core.logging import initialise_logging

from .result_handling import handle_execution_result


def process_result(
    config: Config, database: Database, result: ExecutionResult, execution: Execution | None
) -> None:
    """
    Process the result of a diagnostic execution

    Parameters
    ----------
    config
        The configuration object
    database
        The database object
    result
        The result of the diagnostic execution.

        This could have either been a success or failure.
    execution
        A database model representing the execution of the diagnostic.
    """
    if not result.successful:
        if execution is not None:  # pragma: no branch
            info_msg = (
                f"\nAdditional information about this execution can be viewed using: "
                f"ref executions inspect {execution.execution_group_id}"
            )
        else:
            info_msg = ""

        logger.exception(f"Error running {result.definition.execution_slug()}. {info_msg}")

    if execution:
        handle_execution_result(config, database, execution, result)


@define
class ExecutionFuture:
    """
    A container to hold the future and execution definition
    """

    future: Future[ExecutionResult]
    definition: ExecutionDefinition
    execution_id: int | None = None


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
        logger.exception("Error running diagnostic")
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
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = Config.default()
        if database is None:
            database = Database.from_config(config, run_migrations=False)
        self.n = n

        self.database = database
        self.config = config

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
            )
        )

    def join(self, timeout: float) -> None:
        """
        Wait for all diagnostics to finish

        This will block until all diagnostics have completed or the timeout is reached.
        If the timeout is reached, the method will return and raise an exception.

        Parameters
        ----------
        timeout
            Timeout in seconds

        Raises
        ------
        TimeoutError
            If the timeout is reached
        """
        start_time = time.time()
        refresh_time = 0.5  # Time to wait between checking for completed tasks in seconds

        results = self._results
        t = tqdm(total=len(results), desc="Waiting for executions to complete", unit="execution")

        try:
            while results:
                # Iterate over a copy of the list and remove finished tasks
                for result in results[:]:
                    if result.future.done():
                        try:
                            execution_result = result.future.result(timeout=0)
                        except Exception as e:
                            # Something went wrong when attempting to run the execution
                            # This is likely a failure in the execution itself not the diagnostic
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
                            process_result(self.config, self.database, result.future.result(), execution)
                        logger.debug(f"Execution completed: {result}")
                        t.update(n=1)
                        results.remove(result)

                # Break early to avoid waiting for one more sleep cycle
                if len(results) == 0:
                    break

                elapsed_time = time.time() - start_time

                if elapsed_time > timeout:
                    for result in results:
                        logger.warning(
                            f"Execution {result.definition.execution_slug()} "
                            f"did not complete within the timeout"
                        )
                    self.pool.shutdown(wait=False, cancel_futures=True)
                    raise TimeoutError("Not all tasks completed within the specified timeout")

                # Wait for a short time before checking for completed executions
                time.sleep(refresh_time)
        finally:
            t.close()

        logger.info("All executions completed successfully")
