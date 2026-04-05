"""
Reingest existing execution results without re-running diagnostics.

This module provides functionality to re-run ``build_execution_result()`` and
re-ingest the results into the database for executions that have already completed.
This is useful when new series definitions or metadata extraction logic has been added
and you want to apply it to existing outputs without re-executing the diagnostics.

Reingest always creates a new ``Execution`` record under the same ``ExecutionGroup``
with its own output directory, leaving the original execution untouched.
Results are treated as immutable.
"""

import shutil
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from climate_ref.datasets import get_slug_column
from climate_ref.executor.fragment import allocate_output_fragment
from climate_ref.executor.result_handling import handle_execution_result
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    execution_datasets,
    get_execution_group_and_latest_filtered,
)
from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)
from climate_ref_core.diagnostics import ExecutionDefinition
from climate_ref_core.logging import EXECUTION_LOG_FILENAME

if TYPE_CHECKING:
    from pathlib import Path

    from climate_ref.config import Config
    from climate_ref.database import Database
    from climate_ref.models.dataset import Dataset
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.diagnostics import Diagnostic, ExecutionResult


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
        A reconstructed ``ExecutionDefinition`` pointing at the scratch directory
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


def _validate_path_containment(path: "Path", base: "Path", label: str) -> None:
    """
    Check that *path* stays within *base* after resolving symlinks and ``..`` segments.

    Raises
    ------
    ValueError
        If the resolved *path* escapes *base*
    """
    if not path.resolve().is_relative_to(base.resolve()):
        msg = f"Computed {label} path {path} escapes {base}."
        raise ValueError(msg)


def _build_execution_result(
    config: "Config",
    execution: Execution,
    provider_registry: "ProviderRegistry",
) -> "tuple[ExecutionResult, Path] | None":
    """
    Resolve the diagnostic, validate paths, and build the execution result.

    Returns the result and scratch directory on success, or None on failure.
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
        return None

    scratch_dir = config.paths.scratch / execution.output_fragment
    try:
        _validate_path_containment(scratch_dir, config.paths.scratch, "scratch")
    except ValueError:
        logger.error(f"Skipping execution {execution.id}: scratch path escapes base.")
        return None
    if not scratch_dir.exists():
        logger.error(f"Scratch directory does not exist: {scratch_dir}. Skipping execution {execution.id}.")
        return None

    definition = reconstruct_execution_definition(config, execution, diagnostic)

    try:
        result = diagnostic.build_execution_result(definition)
    except Exception:
        logger.exception(
            f"build_execution_result failed for execution {execution.id} "
            f"({provider_slug}/{diagnostic_slug}). Skipping."
        )
        return None

    if not result.successful or result.metric_bundle_filename is None:
        logger.warning(
            f"build_execution_result returned unsuccessful result for execution {execution.id}. Skipping."
        )
        return None

    return result, scratch_dir


def reingest_execution(
    config: "Config",
    database: "Database",
    execution: Execution,
    provider_registry: "ProviderRegistry",
) -> bool:
    """
    Reingest an existing execution.

    Re-runs ``build_execution_result()`` against the scratch directory
    (which contains the raw outputs from the original diagnostic run),
    creates a new ``Execution`` record with a unique output fragment,
    copies results to the new location, and ingests metrics into the database.

    The original execution is left untouched.

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

    Returns
    -------
    :
        True if reingest was successful, False if it was skipped due to an error
    """
    built = _build_execution_result(config, execution, provider_registry)
    if built is None:
        return False
    result, scratch_dir = built

    execution_group = execution.execution_group

    # Allocate a new output fragment with a timestamp suffix
    new_fragment = allocate_output_fragment(execution.output_fragment, config.paths.results)

    # Copy previous scratch to new location to avoid overwriting CMEC results
    new_scratch_dir = config.paths.scratch / new_fragment
    try:
        _validate_path_containment(new_scratch_dir, config.paths.scratch, "scratch")
    except ValueError:
        logger.error(f"Skipping execution {execution.id}: new scratch path escapes base.")
        return False
    new_scratch_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scratch_dir, new_scratch_dir)

    # Ensure a log file exists in the new scratch location
    # handle_execution_result requires it and the original execution may not have produced one.
    log_file = new_scratch_dir / EXECUTION_LOG_FILENAME
    if not log_file.exists():
        log_file.write_text("Reingested from original execution\n")

    try:
        with database.session.begin_nested():
            # Create new Execution record
            new_execution = Execution(
                execution_group=execution_group,
                dataset_hash=execution.dataset_hash,
                output_fragment=new_fragment,
            )
            database.session.add(new_execution)
            database.session.flush()

            # Copy dataset links from the original execution
            for dataset in execution.datasets:
                database.session.execute(
                    execution_datasets.insert().values(
                        execution_id=new_execution.id,
                        dataset_id=dataset.id,
                    )
                )

        # Save and restore dirty so reingest does not alter the execution group's
        # pending-work state.
        saved_dirty = execution_group.dirty
        handle_execution_result(config, database, new_execution, result)
        execution_group.dirty = saved_dirty
    except Exception:
        logger.exception(f"Ingestion failed for execution {execution.id}. Rolling back changes.")
        # Clean up the new scratch and any partial results on failure.
        if new_scratch_dir.exists():
            shutil.rmtree(new_scratch_dir)
        new_results_dir = config.paths.results / new_fragment
        if new_results_dir.exists():
            shutil.rmtree(new_results_dir)
        return False

    logger.info(f"Successfully reingested execution {execution.id} -> new execution {new_execution.id}")
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

    Always selects the **oldest** (original) execution per group so that
    reingest uses the execution whose scratch directory actually exists.
    Reingested executions only have results directories, not scratch.

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
        List of (ExecutionGroup, oldest Execution) tuples
    """
    # Use the existing filtered query to identify matching execution groups
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

    # Filter out entries with no execution, then select the oldest per group.
    # ExecutionGroup.executions is ordered by created_at ascending,
    # so [0] is the original execution whose scratch directory exists.
    seen: set[int] = set()
    out: list[tuple[ExecutionGroup, Execution]] = []
    for eg, ex in results:
        if ex is None or eg.id in seen:
            continue
        seen.add(eg.id)
        oldest = eg.executions[0]
        if not include_failed and not oldest.successful:
            continue
        out.append((eg, oldest))
    return out
