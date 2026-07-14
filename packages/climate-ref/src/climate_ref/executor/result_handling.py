"""
Execute diagnostics in different environments

We support running diagnostics in different environments, such as locally,
in a separate process, or in a container.
These environments are represented by `climate_ref.executor.Executor` classes.

The simplest executor is the `LocalExecutor`, which runs the diagnostic in the same process.
This is useful for local testing and debugging.
"""

import pathlib
from collections.abc import Sequence
from concurrent.futures import Future
from typing import TYPE_CHECKING, get_args

from attrs import define
from loguru import logger
from sqlalchemy import insert

from climate_ref.database import Database
from climate_ref.models import ScalarMetricValue, SeriesIndex, SeriesMetricValue
from climate_ref.models.execution import Execution, ExecutionOutput, ResultOutputType
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult, ensure_relative_path
from climate_ref_core.exceptions import ResultValidationError
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.metric_values import MetricValueKind
from climate_ref_core.metric_values import ScalarMetricValue as TScalar
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.output_files import copy_execution_outputs, copy_output_file
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputDict

if TYPE_CHECKING:
    from climate_ref.config import Config

# The role of every metric value must be one of these at ingest. This is enforced
# hard (not warn-only) because ``kind`` is the authoritative role signal: a missing
# or unexpected value cannot be silently defaulted without misclassifying the value.
# Derived from the single source of truth so it cannot drift from the model field.
_VALID_KINDS = frozenset(get_args(MetricValueKind))


def _validated_kind_and_dimensions(value: "TScalar | TSeries") -> tuple[str, dict[str, str]]:
    """
    Return the value's validated ``kind`` and its dimensions with ``kind`` removed.

    ``kind`` is persisted in its own column, so it is dropped from the free dimensions
    in case a producer also carried it there. Rejects an unknown role hard rather than
    silently defaulting it.
    """
    kind = value.kind
    if kind not in _VALID_KINDS:
        raise ResultValidationError(
            f"Invalid metric-value kind {kind!r}; expected one of {sorted(_VALID_KINDS)}"
        )
    dimensions = {k: v for k, v in value.dimensions.items() if k != "kind"}
    return kind, dimensions


@define
class ExecutionFuture:
    """A container linking a submitted future to its execution metadata."""

    future: Future[ExecutionResult]
    """
    The future representing the asynchronous execution of this task.

    This future has a base-class of ``concurrent.futures.Future``,
    but the concrete class of the instance will depend on the executor.
    """

    definition: ExecutionDefinition
    """
    The execution definition associated with this future.
    """

    execution_id: int | None = None
    """
    The ID of the execution associated with this future, or ``None`` if not yet assigned.
    """

    submitted_at: float = 0.0
    """
    Wall-clock time (``time.time()``) at which this future was submitted to the executor.
    """

    started_at: float | None = None
    """
    Wall-clock time (``time.time()``) at which a worker was first observed running this future,
    or ``None`` while it is still queued.

    The per-task timeout is budgeted against this rather than ``submitted_at``
    so that time spent waiting in the pool queue is not counted as execution time.
    """


def process_result(
    config: "Config",
    database: Database,
    result: ExecutionResult,
    execution: Execution | None,
) -> None:
    """Process the result of a diagnostic execution, persisting outcome to the DB."""
    if not result.successful:
        if execution is not None:  # pragma: no branch
            info_msg = (
                f"\nAdditional information about this execution can be viewed using: "
                f"ref executions inspect {execution.execution_group_id}"
            )
        else:
            info_msg = ""
        logger.error(f"Error running {result.definition.execution_slug()}. {info_msg}")

    if execution:
        handle_execution_result(config, database, execution, result)


def mark_execution_failed(
    database: Database,
    config: "Config",
    definition: ExecutionDefinition,
    execution_id: int | None,
    *,
    retryable: bool,
) -> None:
    """Persist a failed result for an outstanding execution.

    Used when an executor abandons a future (per-task timeout, overall timeout,
    worker crash) so the corresponding ``Execution`` row never stays stuck in
    ``successful=None`` state.
    """
    try:
        failure_result = ExecutionResult.build_from_failure(definition, retryable=retryable)
        with database.session.begin():
            execution = database.session.get(Execution, execution_id) if execution_id else None
            process_result(config, database, failure_result, execution)
    except Exception:
        logger.exception(f"Failed to record failure for {definition.execution_slug()!r}")


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
        kind, dimensions = _validated_kind_and_dimensions(metric_result)
        new_values.append(
            {
                "execution_id": execution.id,
                "value": metric_result.value,
                "attributes": metric_result.attributes,
                "kind": kind,
                **dimensions,
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

    # Resolve (deduplicate) the shared index axes for this batch up front,
    # so each distinct index is stored once in ``index_axis``
    # and referenced by id rather than duplicated on every series row.
    digest_by_series = [
        SeriesIndex.compute_hash(series_result.index_name, series_result.index)
        for series_result in series_values
    ]
    axis_payload_by_hash: dict[str, tuple[str | None, Sequence[float | int | str]]] = {}
    for series_result, digest in zip(series_values, digest_by_series, strict=True):
        axis_payload_by_hash.setdefault(digest, (series_result.index_name, series_result.index))

    axis_id_by_hash = SeriesIndex.bulk_get_or_create(database.session, axis_payload_by_hash)

    new_values = []
    for series_result, digest in zip(series_values, digest_by_series, strict=True):
        kind, dimensions = _validated_kind_and_dimensions(series_result)
        row = {
            "execution_id": execution.id,
            "values": series_result.values,
            "attributes": series_result.attributes,
            "index_id": axis_id_by_hash[digest],
            "kind": kind,
            **dimensions,
        }
        if kind == "reference":
            # Stable content hash so an identical observation ingested by different
            # executions deduplicates to the same reference_id.
            row["reference_id"] = SeriesMetricValue.compute_reference_id(
                series_result.values,
                series_result.index,
                dimensions.get("reference_source_id"),
            )
        new_values.append(row)

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
    *,
    update_dirty: bool = True,
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
    update_dirty
        Whether to update the execution group's dirty flag.
        Set to False for reingest which should not alter pending-work state.
    """
    # Always copy log data to the results directory
    try:
        copy_output_file(
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
        if update_dirty:
            # Missing log file suggests the process was killed before writing output,
            # so set dirty=True for retry rather than assuming it was already set.
            execution.execution_group.dirty = True
        return

    if not result.successful or result.metric_bundle_filename is None:
        execution.mark_failed()
        if result.retryable:
            logger.error(f"{execution} failed due to a system error (will be retried on next solve)")
            if update_dirty:
                # A hash-change rerun starts from dirty=False, so set it explicitly
                # rather than assuming it was already True.
                execution.execution_group.dirty = True
        else:
            logger.error(f"{execution} failed due to a diagnostic error")
            if update_dirty:
                execution.execution_group.dirty = False
        return

    logger.info(f"{execution} successful")

    # Copy the curated subset of outputs that REF persists for a successful execution
    # (metric bundle, output bundle and the files it references, series)
    # from scratch to results.
    # Only the files in results are served by the API.
    copy_execution_outputs(
        config.paths.scratch,
        config.paths.results,
        execution.output_fragment,
        result,
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
        # The diagnostic ran, but persisting its results failed
        # (malformed bundle, CV/schema mismatch, DB error).
        # The savepoint already rolled back the partial inserts.
        # Record this as a failed execution and leave the group dirty
        # so the next solve retries it — never report success with no metric values.
        logger.exception("Something went wrong when ingesting execution result")
        execution.mark_failed()
        return

    # TODO: This should check if the result is the most recent for the execution,
    # if so then update the dirty fields
    # i.e. if there are outstanding executions don't make as clean
    if update_dirty:
        execution.execution_group.dirty = False

    # Finally, mark the execution as successful
    execution.mark_successful(result.as_relative_path(result.metric_bundle_filename))
