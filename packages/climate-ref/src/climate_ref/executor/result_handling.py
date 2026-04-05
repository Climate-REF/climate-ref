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
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputDict

if TYPE_CHECKING:
    from climate_ref.config import Config


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


def ingest_scalar_values(
    database: Database,
    result: "ExecutionResult",
    execution: Execution,
    cv: CV,
) -> None:
    """
    Load, validate, and bulk-insert scalar metric values.

    Parameters
    ----------
    database
        The active database session to use
    result
        The execution result containing the metric bundle filename
    execution
        The execution record to associate values with
    cv
        The controlled vocabulary to validate against

    Notes
    -----
    Callers are responsible for transaction boundaries; this function does not
    open a nested transaction or catch exceptions.
    """
    cmec_metric_bundle = CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))

    try:
        cv.validate_metrics(cmec_metric_bundle)
    except (ResultValidationError, AssertionError):
        # TODO: Remove once we have settled on a controlled vocabulary
        logger.warning(
            "Diagnostic scalar values do not conform with the controlled vocabulary", exc_info=True
        )

    new_values = []
    for metric_result in cmec_metric_bundle.iter_results():
        new_values.append(
            {
                "execution_id": execution.id,
                "value": metric_result.value,
                "attributes": metric_result.attributes,
                **metric_result.dimensions,
            }
        )

    logger.debug(f"Ingesting {len(new_values)} scalar values for execution {execution.id}")

    if new_values:
        database.session.execute(insert(ScalarMetricValue), new_values)


def ingest_series_values(
    database: Database,
    result: "ExecutionResult",
    execution: Execution,
    cv: CV,
) -> None:
    """
    Load, validate, and bulk-insert series metric values.

    Parameters
    ----------
    database
        The active database session to use
    result
        The execution result containing the series filename
    execution
        The execution record to associate values with
    cv
        The controlled vocabulary to validate against

    Notes
    -----
    Callers are responsible for transaction boundaries; this function does not
    open a nested transaction or catch exceptions.
    """
    assert result.series_filename, "Series filename must be set in the result"

    series_values_path = result.to_output_path(result.series_filename)
    series_values = TSeries.load_from_json(series_values_path)

    try:
        cv.validate_metrics(series_values)
    except (ResultValidationError, AssertionError):
        # TODO: Remove once we have settled on a controlled vocabulary
        logger.warning(
            "Diagnostic series values do not conform with the controlled vocabulary", exc_info=True
        )

    new_values = []
    for series_result in series_values:
        new_values.append(
            {
                "execution_id": execution.id,
                "values": series_result.values,
                "attributes": series_result.attributes,
                "index": series_result.index,
                "index_name": series_result.index_name,
                **series_result.dimensions,
            }
        )

    logger.debug(f"Ingesting {len(new_values)} series values for execution {execution.id}")

    if new_values:
        database.session.execute(insert(SeriesMetricValue), new_values)


def ingest_execution_result(
    database: Database,
    execution: Execution,
    result: "ExecutionResult",
    cv: CV,
    *,
    output_base_path: pathlib.Path,
) -> None:
    """
    Ingest a successful execution result into the database.

    Registers output entries and ingests scalar and series metric values.

    Parameters
    ----------
    database
        The active database session to use
    execution
        The execution record to associate results with
    result
        The successful execution result
    cv
        The controlled vocabulary to validate metrics against
    output_base_path
        Primary base directory for resolving output filenames

    Notes
    -----
    Callers are responsible for:

    * File copying (scratch -> results)
    * Transaction boundaries
    * Marking the execution as successful (``execution.mark_successful()``)
    * Setting the dirty flag on the execution group
    """
    if result.output_bundle_filename:
        cmec_output_bundle = CMECOutput.load_from_json(result.to_output_path(result.output_bundle_filename))
        for attr, output_type in [
            ("plots", ResultOutputType.Plot),
            ("data", ResultOutputType.Data),
            ("html", ResultOutputType.HTML),
        ]:
            register_execution_outputs(
                database,
                execution,
                getattr(cmec_output_bundle, attr),
                output_type=output_type,
                base_path=output_base_path,
            )

    if result.series_filename:
        ingest_series_values(
            database=database,
            result=result,
            execution=execution,
            cv=cv,
        )

    ingest_scalar_values(
        database=database,
        result=result,
        execution=execution,
        cv=cv,
    )


def register_execution_outputs(
    database: Database,
    execution: Execution,
    outputs: "dict[str, OutputDict] | None",
    output_type: ResultOutputType,
    *,
    base_path: pathlib.Path,
) -> None:
    """
    Register output entries in the database.

    Each entry in ``outputs`` is resolved relative to ``base_path``.

    Parameters
    ----------
    database
        The active database session to use
    execution
        The execution record to associate outputs with
    outputs
        Mapping of short name to ``OutputDict`` (may be None)
    output_type
        The type of output being registered
    base_path
        Base directory for resolving relative filenames

    Notes
    -----
    Callers are responsible for transaction boundaries.
    """
    for key, output_info in (outputs or {}).items():
        filename = ensure_relative_path(output_info.filename, base_path)
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
            f"Could not find log file {EXECUTION_LOG_FILENAME} in scratch directory: {config.paths.scratch}. "
            f"This is likely a system error (will be retried on next solve)."
        )
        execution.mark_failed()
        # Missing log file suggests the process was killed before writing output,
        # so leave dirty=True for retry
        return

    if not result.successful or result.metric_bundle_filename is None:
        execution.mark_failed()
        if result.retryable:
            logger.error(f"{execution} failed due to a system error (will be retried on next solve)")
            # Leave dirty=True so the execution is retried on next solve
        else:
            logger.error(f"{execution} failed due to a diagnostic error")
            execution.execution_group.dirty = False
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
        _copy_output_bundle_files(
            config,
            execution,
            result.to_output_path(result.output_bundle_filename),
        )

    if result.series_filename:
        _copy_file_to_results(
            config.paths.scratch,
            config.paths.results,
            execution.output_fragment,
            result.series_filename,
        )

    # Ingest outputs and metrics into the database via the shared ingestion path
    cv = CV.load_from_file(config.paths.dimensions_cv)
    try:
        with database.session.begin_nested():
            ingest_execution_result(
                database,
                execution,
                result,
                cv,
                output_base_path=config.paths.scratch / execution.output_fragment,
            )
    except Exception:
        logger.exception("Something went wrong when ingesting execution result")

    # TODO: This should check if the result is the most recent for the execution,
    # if so then update the dirty fields
    # i.e. if there are outstanding executions don't make as clean
    execution.execution_group.dirty = False

    # Finally, mark the execution as successful
    execution.mark_successful(result.as_relative_path(result.metric_bundle_filename))


def _copy_output_bundle_files(
    config: "Config",
    execution: Execution,
    cmec_output_bundle_filename: pathlib.Path,
) -> None:
    """Copy output bundle referenced files (plots, data, html) from scratch to results."""
    cmec_output_bundle = CMECOutput.load_from_json(cmec_output_bundle_filename)
    scratch_base = config.paths.scratch / execution.output_fragment

    for attr in ("plots", "data", "html"):
        for output_info in (getattr(cmec_output_bundle, attr) or {}).values():
            filename = ensure_relative_path(output_info.filename, scratch_base)
            _copy_file_to_results(
                config.paths.scratch,
                config.paths.results,
                execution.output_fragment,
                filename,
            )
