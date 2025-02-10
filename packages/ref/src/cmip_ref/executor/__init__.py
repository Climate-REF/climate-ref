"""
Execute metrics in different environments

We support running metrics in different environments, such as locally,
in a separate process, or in a container.
These environments are represented by `cmip_ref.executor.Executor` classes.

The simplest executor is the `LocalExecutor`, which runs the metric in the same process.
This is useful for local testing and debugging.
"""

import importlib
import pathlib
import shutil
from typing import TYPE_CHECKING

from loguru import logger

from cmip_ref_core.exceptions import InvalidExecutorException
from cmip_ref_core.executor import Executor

if TYPE_CHECKING:
    from cmip_ref.config import Config
    from cmip_ref.models import MetricExecutionResult
    from cmip_ref_core.metrics import MetricResult


def import_executor_cls(fqn: str) -> type[Executor]:
    """
    Import an executor using a fully qualified module path

    Parameters
    ----------
    fqn
        Full package and attribute name of the executor to import

        For example: `cmip_ref_metrics_example.executor` will use the `executor` attribute from the
        `cmip_ref_metrics_example` package.

    Raises
    ------
    cmip_ref_core.exceptions.InvalidExecutorException
        If the executor cannot be imported

        If the executor isn't a valid `MetricsProvider`.

    Returns
    -------
    :
        Executor instance
    """
    module, attribute_name = fqn.rsplit(".", 1)

    try:
        imp = importlib.import_module(module)
        executor: type[Executor] = getattr(imp, attribute_name)

        # We can't really check if the executor is a subclass of Executor here
        # Protocols can't be used with issubclass if they have non-method members
        # We have to check this at class instantiation time

        return executor
    except ModuleNotFoundError:
        logger.error(f"Package '{fqn}' not found")
        raise InvalidExecutorException(fqn, f"Module '{module}' not found")
    except AttributeError:
        logger.error(f"Provider '{fqn}' not found")
        raise InvalidExecutorException(fqn, f"Executor '{attribute_name}' not found in {module}")


def _copy_file_to_results(
    scratch_directory: pathlib.Path,
    results_directory: pathlib.Path,
    fragment: pathlib.Path | str,
    filename: pathlib.Path | str,
) -> None:
    """
    Copy a file from the scratch directory to the results directory

    Parameters
    ----------
    scratch_directory
        The directory where the file is currently located
    results_directory
        The directory where the file should be copied to
    fragment
        The fragment of the results directory where the file should be copied
    filename
        The name of the file to be copied
    """
    output_directory = results_directory / fragment

    if not (scratch_directory / fragment / filename).exists():
        raise FileNotFoundError(f"Could not find {filename} in {scratch_directory / fragment}")

    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    shutil.copy(scratch_directory / fragment / filename, output_directory / filename)


def handle_execution_result(
    config: "Config", metric_execution_result: "MetricExecutionResult", result: "MetricResult"
) -> None:
    """
    Handle the result of a metric execution

    This will update the metric execution result with the output of the metric execution.
    The output will be copied from the scratch directory to the results directory.

    Parameters
    ----------
    config
        The configuration to use
    metric_execution_result
        The metric execution result to update
    result
        The result of the metric execution, either successful or failed
    """
    if result.successful and result.bundle_filename is not None:
        logger.info(f"{metric_execution_result} successful")

        # TODO: Iterate over the files in the bundle and copy them
        # TODO: Add files to db
        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            metric_execution_result.output_fragment,
            result.bundle_filename,
        )

        metric_execution_result.mark_successful(result.bundle_filename)

        # TODO: This should check if the result is the most recent for the execution,
        # if so then update the dirty fields
        # i.e. if there are outstanding results don't make as clean
        metric_execution_result.metric_execution.dirty = False
    else:
        logger.info(f"{metric_execution_result} failed")
        metric_execution_result.mark_failed()
