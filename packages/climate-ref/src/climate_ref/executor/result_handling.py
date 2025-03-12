"""
Execute diagnostics in different environments

We support running diagnostics in different environments, such as locally,
in a separate process, or in a container.
These environments are represented by `climate_ref.executor.Executor` classes.

The simplest executor is the `LocalExecutor`, which runs the diagnostic in the same process.
This is useful for local testing and debugging.
"""

import pathlib
import shutil
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import insert

from climate_ref.database import Database
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
from climate_ref.models.execution import Execution, ExecutionOutput, ResultOutputType
from climate_ref_core.diagnostics import ExecutionResult, ensure_relative_path
from climate_ref_core.exceptions import ResultValidationError
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputDict

if TYPE_CHECKING:
    from climate_ref.config import Config

EXECUTION_LOG_FILENAME = "out.log"


def _copy_file_to_results(
    scratch_directory: pathlib.Path,
    results_directory: pathlib.Path,
    fragment: pathlib.Path | str,
    filename: pathlib.Path | str,
) -> None:
    """
    Copy a file from the scratch directory to the executions directory

    Parameters
    ----------
    scratch_directory
        The directory where the file is currently located
    results_directory
        The directory where the file should be copied to
    fragment
        The fragment of the executions directory where the file should be copied
    filename
        The name of the file to be copied
    """
    assert results_directory != scratch_directory
    input_directory = scratch_directory / fragment
    output_directory = results_directory / fragment

    filename = ensure_relative_path(filename, input_directory)

    if not (input_directory / filename).exists():
        raise FileNotFoundError(f"Could not find {filename} in {input_directory}")

    output_filename = output_directory / filename
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(input_directory / filename, output_filename)


def _process_execution_scalar(
    database: Database,
    result: "ExecutionResult",
    execution: Execution,
    cv: CV,
) -> None:
    """
    Process the scalar values from the execution result and store them in the database

    This also validates the scalar values against the controlled vocabulary
    """
    # Load the metric bundle from the file
    cmec_metric_bundle = CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))

    # Check that the diagnostic values conform with the controlled vocabulary
    try:
        cv.validate_metrics(cmec_metric_bundle)
    except (ResultValidationError, AssertionError):
        # TODO: Remove once we have settled on a controlled vocabulary
        logger.exception("Diagnostic values do not conform with the controlled vocabulary")
        # execution.mark_failed()

    # Perform a bulk insert of scalar values
    # The current implementation will swallow the exception, but display a log message
    try:
        scalar_values = [
            {
                "execution_id": execution.id,
                "value": result.value,
                "attributes": result.attributes,
                **result.dimensions,
            }
            for result in cmec_metric_bundle.iter_results()
        ]
        logger.debug(f"Ingesting {len(scalar_values)} scalar values for execution {execution.id}")
        if scalar_values:
            # Perform this in a nested transaction to rollback if something goes wrong
            # We will lose the metric values for a given execution, but not the whole execution
            with database.session.begin_nested():
                database.session.execute(
                    insert(ScalarMetricValue),
                    scalar_values,
                )
    # This is a broad exception catch to ensure we log any issues
    except Exception:
        logger.exception("Something went wrong when ingesting diagnostic values")


def _process_execution_series(
    config: "Config",
    database: Database,
    result: "ExecutionResult",
    execution: Execution,
    cv: CV,
) -> None:
    """
    Process the series values from the execution result and store them in the database

    This also copies the series values file from the scratch directory to the results directory
    and validates the series values against the controlled vocabulary.
    """
    assert result.series_filename, "Series filename must be set in the result"

    _copy_file_to_results(
        config.paths.scratch,
        config.paths.results,
        execution.output_fragment,
        result.series_filename,
    )

    # Load the series values from the file
    series_values_path = result.to_output_path(result.series_filename)
    series_values = TSeries.load_from_json(series_values_path)

    try:
        cv.validate_metrics(series_values)
    except (ResultValidationError, AssertionError):
        # TODO: Remove once we have settled on a controlled vocabulary
        logger.exception("Diagnostic values do not conform with the controlled vocabulary")
        # execution.mark_failed()

    # Perform a bulk insert of series values
    try:
        series_values_content = [
            {
                "execution_id": execution.id,
                "values": series_result.values,
                "attributes": series_result.attributes,
                "index": series_result.index,
                "index_name": series_result.index_name,
                **series_result.dimensions,
            }
            for series_result in series_values
        ]
        logger.debug(f"Ingesting {len(series_values)} series values for execution {execution.id}")
        if series_values:
            # Perform this in a nested transaction to rollback if something goes wrong
            # We will lose the metric values for a given execution, but not the whole execution
            with database.session.begin_nested():
                database.session.execute(
                    insert(SeriesMetricValue),
                    series_values_content,
                )
    except Exception:
        logger.exception("Something went wrong when ingesting diagnostic series values")


def handle_execution_result(
    config: "Config",
    database: Database,
    execution: Execution,
    result: "ExecutionResult",
) -> None:
    """
    Handle the result of a diagnostic execution

    This will update the diagnostic execution result with the output of the diagnostic execution.
    The output will be copied from the scratch directory to the executions directory.

    Parameters
    ----------
    config
        The configuration to use
    database
        The active database session to use
    execution
        The diagnostic execution result DB object to update
    result
        The result of the diagnostic execution, either successful or failed
    """
    # Always copy log data to the results directory
    try:
        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            execution.output_fragment,
            EXECUTION_LOG_FILENAME,
        )
    except FileNotFoundError:
        logger.error(
            f"Could not find log file {EXECUTION_LOG_FILENAME} in scratch directory: {config.paths.scratch}"
        )
        execution.mark_failed()
        return

    if not result.successful or result.metric_bundle_filename is None:
        logger.error(f"{execution} failed")
        execution.mark_failed()
        return

    logger.info(f"{execution} successful")

    _copy_file_to_results(
        config.paths.scratch,
        config.paths.results,
        execution.output_fragment,
        result.metric_bundle_filename,
    )

    if result.output_bundle_filename:
        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            execution.output_fragment,
            result.output_bundle_filename,
        )
        _handle_output_bundle(
            config,
            database,
            execution,
            result.to_output_path(result.output_bundle_filename),
        )

    cv = CV.load_from_file(config.paths.dimensions_cv)

    if result.series_filename:
        # Process the series values if they are present
        # This will ingest the series values into the database
        _process_execution_series(config=config, database=database, result=result, execution=execution, cv=cv)

    # Process the scalar values
    # This will ingest the scalar values into the database
    _process_execution_scalar(database=database, result=result, execution=execution, cv=cv)

    # TODO: This should check if the result is the most recent for the execution,
    # if so then update the dirty fields
    # i.e. if there are outstanding executions don't make as clean
    execution.execution_group.dirty = False

    # Finally, mark the execution as successful
    execution.mark_successful(result.as_relative_path(result.metric_bundle_filename))


def _handle_output_bundle(
    config: "Config",
    database: Database,
    execution: Execution,
    cmec_output_bundle_filename: pathlib.Path,
) -> None:
    # Extract the registered outputs
    # Copy the content to the output directory
    # Track in the db
    cmec_output_bundle = CMECOutput.load_from_json(cmec_output_bundle_filename)
    _handle_outputs(
        cmec_output_bundle.plots,
        output_type=ResultOutputType.Plot,
        config=config,
        database=database,
        execution=execution,
    )
    _handle_outputs(
        cmec_output_bundle.data,
        output_type=ResultOutputType.Data,
        config=config,
        database=database,
        execution=execution,
    )
    _handle_outputs(
        cmec_output_bundle.html,
        output_type=ResultOutputType.HTML,
        config=config,
        database=database,
        execution=execution,
    )


def _handle_outputs(
    outputs: dict[str, OutputDict] | None,
    output_type: ResultOutputType,
    config: "Config",
    database: Database,
    execution: Execution,
) -> None:
    outputs = outputs or {}

    for key, output_info in outputs.items():
        filename = ensure_relative_path(
            output_info.filename, config.paths.scratch / execution.output_fragment
        )

        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            execution.output_fragment,
            filename,
        )
        database.session.add(
            ExecutionOutput.build(
                execution_id=execution.id,
                output_type=output_type,
                filename=str(filename),
                description=output_info.description,
                short_name=key,
                long_name=output_info.long_name,
                dimensions=output_info.dimensions or {},
            )
        )
