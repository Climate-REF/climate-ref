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

from cmip_ref.database import Database
from cmip_ref.models.metric_execution import MetricExecutionResult, ResultOutput, ResultOutputType
from cmip_ref_core.exceptions import InvalidExecutorException
from cmip_ref_core.executor import Executor
from cmip_ref_core.metrics import MetricResult, ensure_relative_path
from cmip_ref_core.pycmec.output import CMECOutput, OutputDict

if TYPE_CHECKING:
    from cmip_ref.config import Config


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
    assert results_directory != scratch_directory  # noqa
    input_directory = scratch_directory / fragment
    output_directory = results_directory / fragment

    filename = ensure_relative_path(filename, input_directory)

    if not (input_directory / filename).exists():
        raise FileNotFoundError(f"Could not find {filename} in {input_directory}")

    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    shutil.copy(input_directory / filename, output_directory / filename)


def handle_execution_result(
    config: "Config",
    database: Database,
    metric_execution_result: "MetricExecutionResult",
    result: "MetricResult",
) -> None:
    """
    Handle the result of a metric execution

    This will update the metric execution result with the output of the metric execution.
    The output will be copied from the scratch directory to the results directory.

    Parameters
    ----------
    config
        The configuration to use
    database
        The active database session to use
    metric_execution_result
        The metric execution result DB object to update
    result
        The result of the metric execution, either successful or failed
    """
    if result.successful and result.metric_bundle_filename is not None:
        logger.info(f"{metric_execution_result} successful")

        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            metric_execution_result.output_fragment,
            result.metric_bundle_filename,
        )
        metric_execution_result.mark_successful(result.as_relative_path(result.metric_bundle_filename))

        if result.output_bundle_filename:
            _copy_file_to_results(
                config.paths.scratch,
                config.paths.results,
                metric_execution_result.output_fragment,
                result.output_bundle_filename,
            )

            # Extract the registered outputs
            # Copy the content to the output directory
            # Track in the db
            cmec_output_bundle = CMECOutput.load_from_json(
                result.to_output_path(result.output_bundle_filename)
            )
            _handle_outputs(
                cmec_output_bundle.plots,
                output_type=ResultOutputType.Plot,
                config=config,
                database=database,
                metric_execution_result=metric_execution_result,
            )
            _handle_outputs(
                cmec_output_bundle.data,
                output_type=ResultOutputType.Data,
                config=config,
                database=database,
                metric_execution_result=metric_execution_result,
            )
            _handle_outputs(
                cmec_output_bundle.html,
                output_type=ResultOutputType.HTML,
                config=config,
                database=database,
                metric_execution_result=metric_execution_result,
            )

        # TODO: This should check if the result is the most recent for the execution,
        # if so then update the dirty fields
        # i.e. if there are outstanding results don't make as clean
        metric_execution_result.metric_execution.dirty = False
    else:
        logger.error(f"{metric_execution_result} failed")
        metric_execution_result.mark_failed()


def _handle_outputs(
    outputs: dict[str, OutputDict] | None,
    output_type: ResultOutputType,
    config: "Config",
    database: Database,
    metric_execution_result: MetricExecutionResult,
) -> None:
    if outputs is None:
        return

    for key, output_info in outputs.items():
        filename = ensure_relative_path(
            output_info.filename, config.paths.scratch / metric_execution_result.output_fragment
        )

        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            metric_execution_result.output_fragment,
            filename,
        )
        database.session.add(
            ResultOutput(
                metric_execution_result_id=metric_execution_result.id,
                output_type=output_type,
                filename=str(filename),
                description=output_info.description,
                short_name=key,
                long_name=output_info.long_name,
            )
        )
