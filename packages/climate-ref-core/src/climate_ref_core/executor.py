"""
Executor interface for running diagnostics
"""

import importlib
import shutil
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import DiagnosticError, InvalidExecutorException
from climate_ref_core.logging import redirect_logs


def _is_system_error(exc: BaseException) -> bool:
    """
    Check if an exception represents a system-level failure.

    System failures are retryable because running again with the same data
    may succeed (e.g. more memory available, disk space freed).
    Diagnostic logic errors are not retryable.

    Parameters
    ----------
    exc
        The exception to classify

    Returns
    -------
    :
        True if the exception is a system-level failure
    """
    return isinstance(exc, (MemoryError, OSError, SystemExit, KeyboardInterrupt))


def execute_locally(
    definition: ExecutionDefinition,
    log_level: str,
    raise_error: bool = False,
) -> ExecutionResult:
    """
    Run a diagnostic execution

    This is the chunk of work that should be executed by an executor.

    Parameters
    ----------
    definition
        A description of the information needed for this execution of the diagnostic
    log_level
        The log level to use for the execution
    """
    logger.info(f"Executing {definition.execution_slug()!r}")

    try:
        if definition.output_directory.exists():
            logger.warning(
                f"Output directory {definition.output_directory} already exists. "
                f"Removing the existing directory."
            )
            shutil.rmtree(definition.output_directory)
        definition.output_directory.mkdir(parents=True, exist_ok=True)

        with redirect_logs(definition, log_level):
            return definition.diagnostic.run(definition=definition)
    except Exception as e:
        # If the diagnostic fails, we want to log the error and return a failure result
        retryable = _is_system_error(e)
        if retryable:
            logger.error(
                f"System error running {definition.execution_slug()!r} (will be retried on next solve): {e}"
            )
        else:
            logger.exception(f"Diagnostic error running {definition.execution_slug()!r}")
        result = ExecutionResult.build_from_failure(definition, retryable=retryable)

        if raise_error:
            raise DiagnosticError(str(e), result) from e
        else:
            return result


@runtime_checkable
class Executor(Protocol):
    """
    An executor is responsible for running a diagnostic asynchronously

    The diagnostic may be run locally in the same process or in a separate process or container.

    Notes
    -----
    This is an extremely basic interface and will be expanded in the future, as we figure out
    our requirements.
    """

    name: str

    def __init__(self, **kwargs: Any) -> None: ...

    def run(
        self,
        definition: ExecutionDefinition,
        execution: Any = None,
    ) -> None:
        """
        Execute a diagnostic with a given definition

        No executions are returned from this method,
        as the execution may be performed asynchronously so executions may not be immediately available.

        /// admonition | Note
        In future, we may return a `Future` object that can be used to retrieve the result,
        but that requires some additional work to implement.
        ///

        Parameters
        ----------
        definition
            Definition of the information needed to execute a diagnostic

            This definition describes which datasets are required to run the diagnostic and where
            the output should be stored.
        execution
            The execution object to update with the results of the execution.

            This is a database object that contains the executions of the execution.
            If provided, it will be updated with the executions of the execution.
            This may happen asynchronously, so the executions may not be immediately available.

        Returns
        -------
        :
            Results from running the diagnostic
        """
        ...

    def join(self, timeout: float) -> None:
        """
        Wait for all executions to finish

        If the timeout is reached, the method will return and raise an exception.

        Parameters
        ----------
        timeout
            Maximum time to wait for all executions to finish in seconds

        Raises
        ------
        TimeoutError
            If the timeout is reached
        """


def import_executor_cls(fqn: str) -> type[Executor]:
    """
    Import an executor using a fully qualified module path

    Parameters
    ----------
    fqn
        Full package and attribute name of the executor to import

        For example: `climate_ref_example.executor` will use the `executor` attribute from the
        `climate_ref_example` package.

    Raises
    ------
    InvalidExecutorException
        If the executor cannot be imported

        If the executor isn't a valid `DiagnosticProvider`.

    Returns
    -------
    :
        Executor instance
    """
    module, attribute_name = fqn.rsplit(".", 1)

    try:
        imp = importlib.import_module(module)
        executor: type[Executor] = getattr(imp, attribute_name)

        if isinstance(executor, Exception):
            raise executor

        # We can't really check if the executor is a subclass of Executor here
        # Protocols can't be used with issubclass if they have non-method members
        # We have to check this at class instantiation time

        return executor
    except (ModuleNotFoundError, ImportError):
        logger.error(f"Package '{fqn}' not found")
        raise InvalidExecutorException(fqn, f"Module '{module}' not found")
    except AttributeError:
        logger.error(f"Provider '{fqn}' not found")
        raise InvalidExecutorException(fqn, f"Executor '{attribute_name}' not found in {module}")
