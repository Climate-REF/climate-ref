"""
Reingest existing execution results without re-running diagnostics.

This module provides functionality to re-run ``build_execution_result()`` and
re-ingest the results into the database for executions that have already completed.
This is useful when new series definitions or metadata extraction logic has been added
and you want to apply it to existing outputs without re-executing the diagnostics.

Three reingest modes are supported:

* **additive** -- preserve existing metric values, insert only values with
  dimension signatures not already present for the execution
* **replace** -- delete all existing metric values and outputs, then re-ingest
* **versioned** -- create a new ``Execution`` record under the same ``ExecutionGroup``
  with its own output directory, leaving the original execution untouched
"""

import enum
import hashlib
import pathlib
import shutil
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger
from sqlalchemy import delete, insert

from climate_ref.datasets import get_slug_column
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
    ResultOutputType,
    execution_datasets,
    get_execution_group_and_latest_filtered,
)
from climate_ref.models.metric_value import MetricValue
from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult, ensure_relative_path
from climate_ref_core.exceptions import ResultValidationError
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputDict

if TYPE_CHECKING:
    from climate_ref.config import Config
    from climate_ref.database import Database
    from climate_ref.models.dataset import Dataset
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.diagnostics import Diagnostic


class ReingestMode(enum.Enum):
    """Mode for reingesting execution results."""

    additive = "additive"
    replace = "replace"
    versioned = "versioned"


def reconstruct_execution_definition(
    config: "Config",
    execution: Execution,
    diagnostic: "Diagnostic",
) -> ExecutionDefinition:
    """
    Reconstruct an ``ExecutionDefinition`` from database state.

    This rebuilds the definition that was originally used to produce the execution,
    using the execution's stored datasets, output fragment, and the live diagnostic
    object from the provider registry.

    Parameters
    ----------
    config
        Application configuration (provides ``paths.results``)
    execution
        The database ``Execution`` record to reconstruct from
    diagnostic
        The live ``Diagnostic`` instance resolved from the provider registry

    Returns
    -------
    :
        A reconstructed ``ExecutionDefinition`` pointing at the results directory
    """
    execution_group = execution.execution_group

    # Build DatasetCollection per source type from the execution's linked datasets
    datasets_by_type: dict[SourceDatasetType, list[Any]] = defaultdict(list)
    for dataset in execution.datasets:
        datasets_by_type[dataset.dataset_type].append(dataset)

    collection: dict[SourceDatasetType | str, DatasetCollection] = {}
    for source_type, ds_list in datasets_by_type.items():
        slug_column = get_slug_column(source_type)

        # Build a DataFrame from the DB dataset records and their files
        rows = []
        for dataset in ds_list:
            # Get all attributes from the polymorphic dataset model
            dataset_attrs = _extract_dataset_attributes(dataset)
            for file in dataset.files:
                row = {
                    **dataset_attrs,
                    "path": file.path,
                    "start_time": file.start_time,
                    "end_time": file.end_time,
                }
                if hasattr(file, "tracking_id"):
                    row["tracking_id"] = file.tracking_id
                rows.append((dataset.id, row))

        if rows:
            index = [r[0] for r in rows]
            data = [r[1] for r in rows]
            df = pd.DataFrame(data, index=index)
        else:
            df = pd.DataFrame()

        # Retrieve the selector for this source type from the execution group
        selector_key = source_type.value
        selector = tuple(tuple(pair) for pair in execution_group.selectors.get(selector_key, []))

        collection[source_type] = DatasetCollection(
            datasets=df,
            slug_column=slug_column,
            selector=selector,
        )

    # Point at the scratch directory -- the caller is expected to copy results
    # to scratch before calling build_execution_result() to avoid mutating
    # the live results tree.
    output_directory = config.paths.scratch / execution.output_fragment

    return ExecutionDefinition(
        diagnostic=diagnostic,
        key=execution_group.key,
        datasets=ExecutionDatasetCollection(collection),
        root_directory=config.paths.scratch,
        output_directory=output_directory,
    )


def _extract_dataset_attributes(dataset: "Dataset") -> dict[str, object]:
    """
    Extract all column values from a polymorphic dataset model as a dict.

    Introspects the SQLAlchemy mapper to get all mapped columns for the concrete
    dataset type (e.g. CMIP6Dataset, Obs4MIPsDataset).
    """
    attrs = {}
    # Get columns from the concrete mapper (handles polymorphic inheritance)
    mapper = type(dataset).__mapper__
    for column in mapper.columns:
        col_name = column.key
        # Skip internal/FK columns
        if col_name in ("id", "dataset_type"):
            continue
        val = getattr(dataset, col_name, None)
        if val is not None:
            attrs[col_name] = val
    return attrs


def _handle_reingest_output_bundle(
    config: "Config",
    database: "Database",
    execution: Execution,
    cmec_output_bundle_filename: pathlib.Path,
) -> None:
    """
    Process the output bundle for reingest (no file copy, files already in results dir).
    """
    cmec_output_bundle = CMECOutput.load_from_json(cmec_output_bundle_filename)
    _handle_reingest_outputs(
        cmec_output_bundle.plots,
        output_type=ResultOutputType.Plot,
        config=config,
        database=database,
        execution=execution,
    )
    _handle_reingest_outputs(
        cmec_output_bundle.data,
        output_type=ResultOutputType.Data,
        config=config,
        database=database,
        execution=execution,
    )
    _handle_reingest_outputs(
        cmec_output_bundle.html,
        output_type=ResultOutputType.HTML,
        config=config,
        database=database,
        execution=execution,
    )


def _handle_reingest_outputs(
    outputs: dict[str, OutputDict] | None,
    output_type: "ResultOutputType",
    config: "Config",
    database: "Database",
    execution: Execution,
) -> None:
    """Register outputs in the DB without copying files (they are already in place)."""
    outputs = outputs or {}
    results_base = config.paths.results / execution.output_fragment

    for key, output_info in outputs.items():
        filename = ensure_relative_path(output_info.filename, results_base)
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


def _process_reingest_series(
    database: "Database",
    result: ExecutionResult,
    execution: Execution,
    cv: CV,
) -> None:
    """
    Process series values for reingest.

    Like ``_process_execution_series`` but skips the file copy since the series
    file is already in the results directory. Unlike the normal ingestion path,
    errors are raised rather than swallowed so that the caller can roll back.
    """
    assert result.series_filename, "Series filename must be set in the result"

    # Load the series values directly from the results directory
    series_values_path = result.to_output_path(result.series_filename)
    series_values = TSeries.load_from_json(series_values_path)

    try:
        cv.validate_metrics(series_values)
    except (ResultValidationError, AssertionError):
        logger.exception("Diagnostic values do not conform with the controlled vocabulary")

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
        database.session.execute(
            insert(SeriesMetricValue),
            series_values_content,
        )


def _process_reingest_scalar(
    database: "Database",
    result: ExecutionResult,
    execution: Execution,
    cv: CV,
) -> None:
    """
    Process scalar values for reingest.

    Like ``_process_execution_scalar`` but raises on insertion errors
    so the caller can roll back.
    """
    cmec_metric_bundle = CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))

    try:
        cv.validate_metrics(cmec_metric_bundle)
    except (ResultValidationError, AssertionError):
        logger.exception("Diagnostic values do not conform with the controlled vocabulary")

    scalar_values = [
        {
            "execution_id": execution.id,
            "value": metric_result.value,
            "attributes": metric_result.attributes,
            **metric_result.dimensions,
        }
        for metric_result in cmec_metric_bundle.iter_results()
    ]
    logger.debug(f"Ingesting {len(scalar_values)} scalar values for execution {execution.id}")
    if scalar_values:
        database.session.execute(
            insert(ScalarMetricValue),
            scalar_values,
        )


def _get_existing_metric_dimensions(
    database: "Database", execution: Execution
) -> set[tuple[tuple[str, str], ...]]:
    """
    Get the dimension signatures of all existing metric values for an execution.

    Each signature is a sorted tuple of (dimension_name, value) pairs, making
    it suitable for set-based deduplication.
    """
    sigs: set[tuple[tuple[str, str], ...]] = set()
    for mv in database.session.query(MetricValue).filter(MetricValue.execution_id == execution.id).all():
        dims = tuple(sorted(mv.dimensions.items()))
        sigs.add(dims)
    return sigs


def _process_reingest_scalar_additive(
    database: "Database",
    result: ExecutionResult,
    execution: Execution,
    cv: CV,
) -> None:
    """
    Process scalar values for additive reingest.

    Only inserts values whose dimension signatures are not already present
    for this execution.
    """
    cmec_metric_bundle = CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))

    try:
        cv.validate_metrics(cmec_metric_bundle)
    except (ResultValidationError, AssertionError):
        logger.exception("Diagnostic values do not conform with the controlled vocabulary")

    existing = _get_existing_metric_dimensions(database, execution)

    new_values = []
    for metric_result in cmec_metric_bundle.iter_results():
        dims = tuple(sorted(metric_result.dimensions.items()))
        if dims not in existing:
            new_values.append(
                {
                    "execution_id": execution.id,
                    "value": metric_result.value,
                    "attributes": metric_result.attributes,
                    **metric_result.dimensions,
                }
            )

    logger.debug(
        f"Additive: {len(new_values)} new scalar values "
        f"(skipped {len(list(cmec_metric_bundle.iter_results())) - len(new_values)} existing) "
        f"for execution {execution.id}"
    )
    if new_values:
        database.session.execute(insert(ScalarMetricValue), new_values)


def _process_reingest_series_additive(
    database: "Database",
    result: ExecutionResult,
    execution: Execution,
    cv: CV,
) -> None:
    """
    Process series values for additive reingest.

    Only inserts series whose dimension signatures are not already present
    for this execution.
    """
    assert result.series_filename, "Series filename must be set in the result"

    series_values_path = result.to_output_path(result.series_filename)
    series_values = TSeries.load_from_json(series_values_path)

    try:
        cv.validate_metrics(series_values)
    except (ResultValidationError, AssertionError):
        logger.exception("Diagnostic values do not conform with the controlled vocabulary")

    existing = _get_existing_metric_dimensions(database, execution)

    new_values = []
    for series_result in series_values:
        dims = tuple(sorted(series_result.dimensions.items()))
        if dims not in existing:
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

    logger.debug(
        f"Additive: {len(new_values)} new series values "
        f"(skipped {len(series_values) - len(new_values)} existing) "
        f"for execution {execution.id}"
    )
    if new_values:
        database.session.execute(insert(SeriesMetricValue), new_values)


def _copy_results_to_scratch(
    config: "Config",
    output_fragment: str,
) -> pathlib.Path:
    """
    Copy an execution's results directory to scratch for safe re-extraction.

    ``build_execution_result()`` may write files (CMEC bundles) into its
    ``definition.output_directory``.  Running it directly against the live
    results tree would mutate the original outputs before the DB savepoint
    has a chance to roll back on failure.  Copying to scratch first keeps
    the results tree unchanged until we decide to commit.

    Returns the scratch output directory.
    """
    src = config.paths.results / output_fragment
    dst = config.paths.scratch / output_fragment
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def _delete_execution_results(database: "Database", execution: Execution) -> None:
    """Delete all metric values and outputs for an execution."""
    database.session.execute(delete(ExecutionOutput).where(ExecutionOutput.execution_id == execution.id))
    # MetricValue uses single-table inheritance so we delete from the base table
    database.session.execute(delete(MetricValue).where(MetricValue.execution_id == execution.id))


def _apply_reingest_mode(  # noqa: PLR0913
    config: "Config",
    database: "Database",
    execution: Execution,
    result: ExecutionResult,
    mode: ReingestMode,
    cv: CV,
) -> Execution:
    """
    Apply mode-specific DB mutations inside a savepoint.

    Returns the target execution (may differ from ``execution`` in versioned mode).
    """
    target_execution = execution
    execution_group = execution.execution_group

    with database.session.begin_nested():
        if mode == ReingestMode.versioned:
            version_hash = hashlib.sha1(  # noqa: S324
                f"{execution.output_fragment}-reingest-{execution.id}".encode()
            ).hexdigest()[:12]

            target_execution = Execution(
                execution_group=execution_group,
                dataset_hash=execution.dataset_hash,
                output_fragment=f"{execution.output_fragment}_v{version_hash}",
            )
            database.session.add(target_execution)
            database.session.flush()

            for dataset in execution.datasets:
                database.session.execute(
                    execution_datasets.insert().values(
                        execution_id=target_execution.id,
                        dataset_id=dataset.id,
                    )
                )
        elif mode == ReingestMode.replace:
            _delete_execution_results(database, target_execution)

        if result.output_bundle_filename:
            if mode != ReingestMode.additive:
                database.session.execute(
                    delete(ExecutionOutput).where(ExecutionOutput.execution_id == target_execution.id)
                )
            _handle_reingest_output_bundle(
                config,
                database,
                target_execution,
                result.to_output_path(result.output_bundle_filename),
            )

        _ingest_metrics(database, result, target_execution, cv, additive=mode == ReingestMode.additive)

        if mode == ReingestMode.versioned:
            assert result.metric_bundle_filename is not None
            target_execution.mark_successful(result.as_relative_path(result.metric_bundle_filename))

    return target_execution


def _ingest_metrics(
    database: "Database",
    result: ExecutionResult,
    execution: Execution,
    cv: CV,
    *,
    additive: bool,
) -> None:
    """Ingest scalar and series metric values, using additive dedup when requested."""
    if result.series_filename:
        if additive:
            _process_reingest_series_additive(database=database, result=result, execution=execution, cv=cv)
        else:
            _process_reingest_series(database=database, result=result, execution=execution, cv=cv)

    if additive:
        _process_reingest_scalar_additive(database=database, result=result, execution=execution, cv=cv)
    else:
        _process_reingest_scalar(database=database, result=result, execution=execution, cv=cv)


def _copy_scratch_to_results(
    config: "Config",
    scratch_dir: pathlib.Path,
    target_execution: Execution,
    mode: ReingestMode,
    original_results_dir: pathlib.Path,
) -> None:
    """Copy re-extracted files from scratch to the results tree after DB success."""
    target_results_dir = config.paths.results / target_execution.output_fragment
    if mode == ReingestMode.versioned:
        if target_results_dir.exists():
            shutil.rmtree(target_results_dir)
        shutil.copytree(scratch_dir, target_results_dir)
    else:
        shutil.copytree(scratch_dir, original_results_dir, dirs_exist_ok=True)


def reingest_execution(
    config: "Config",
    database: "Database",
    execution: Execution,
    provider_registry: "ProviderRegistry",
    mode: ReingestMode = ReingestMode.additive,
) -> bool:
    """
    Reingest an existing execution.

    Re-runs ``build_execution_result()`` and processes the results into the database.

    Parameters
    ----------
    config
        Application configuration
    database
        Database instance
    execution
        The ``Execution`` record to reingest
    provider_registry
        Registry of active providers (used to resolve the live diagnostic)
    mode
        Reingest mode: additive, replace, or versioned

    Returns
    -------
    :
        True if reingest was successful, False if it was skipped due to an error
    """
    execution_group = execution.execution_group
    diagnostic_model = execution_group.diagnostic
    provider_slug = diagnostic_model.provider.slug
    diagnostic_slug = diagnostic_model.slug

    try:
        diagnostic = provider_registry.get_metric(provider_slug, diagnostic_slug)
    except KeyError:
        logger.error(
            f"Could not resolve diagnostic {provider_slug}/{diagnostic_slug} "
            f"from provider registry. Skipping execution {execution.id}."
        )
        return False

    results_dir = config.paths.results / execution.output_fragment

    # Verify output directory exists
    if not results_dir.exists():
        logger.error(f"Output directory does not exist: {results_dir}. Skipping execution {execution.id}.")
        return False

    # Copy the results directory to scratch so that build_execution_result()
    # can write CMEC bundles without mutating the live results tree.
    # If anything fails, the original files remain untouched.
    scratch_dir = _copy_results_to_scratch(config, execution.output_fragment)

    # Reconstruct the definition pointing at the scratch copy
    definition = reconstruct_execution_definition(config, execution, diagnostic)

    # Re-run build_execution_result on the scratch copy
    try:
        result = diagnostic.build_execution_result(definition)
    except Exception:
        logger.exception(
            f"build_execution_result failed for execution {execution.id} "
            f"({provider_slug}/{diagnostic_slug}). Skipping."
        )
        return False

    if not result.successful or result.metric_bundle_filename is None:
        logger.warning(
            f"build_execution_result returned unsuccessful result for execution {execution.id}. Skipping."
        )
        return False

    cv = CV.load_from_file(config.paths.dimensions_cv)

    # All mode-specific mutations happen inside a single savepoint so that
    # any failure rolls back everything, preserving the original DB state.
    try:
        target_execution = _apply_reingest_mode(
            config=config,
            database=database,
            execution=execution,
            result=result,
            mode=mode,
            cv=cv,
        )
    except Exception:
        logger.exception(
            f"Ingestion failed for execution {execution.id} "
            f"({provider_slug}/{diagnostic_slug}). Rolling back changes."
        )
        return False

    # DB transaction succeeded -- copy files from scratch to the results tree.
    _copy_scratch_to_results(config, scratch_dir, target_execution, mode, results_dir)

    logger.info(
        f"Successfully reingested execution {execution.id} "
        f"({provider_slug}/{diagnostic_slug}) in {mode.value} mode"
    )
    return True


def get_executions_for_reingest(
    database: "Database",
    *,
    execution_group_ids: Sequence[int] | None = None,
    provider_filters: list[str] | None = None,
    diagnostic_filters: list[str] | None = None,
    include_failed: bool = False,
) -> list[tuple[ExecutionGroup, Execution]]:
    """
    Query executions eligible for reingest.

    Parameters
    ----------
    database
        Database instance
    execution_group_ids
        If provided, only include these execution group IDs
    provider_filters
        Filter by provider slug (substring, case-insensitive)
    diagnostic_filters
        Filter by diagnostic slug (substring, case-insensitive)
    include_failed
        If True, also include failed executions

    Returns
    -------
    :
        List of (ExecutionGroup, latest Execution) tuples
    """
    # Use the existing filtered query
    results = get_execution_group_and_latest_filtered(
        database.session,
        diagnostic_filters=diagnostic_filters,
        provider_filters=provider_filters,
        successful=None if include_failed else True,
    )

    # Filter by execution group IDs if provided
    if execution_group_ids:
        id_set = set(execution_group_ids)
        results = [(eg, ex) for eg, ex in results if eg.id in id_set]

    # Filter out entries with no execution
    return [(eg, ex) for eg, ex in results if ex is not None]
