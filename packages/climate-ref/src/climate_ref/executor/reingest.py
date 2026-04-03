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
from sqlalchemy import delete

from climate_ref.datasets import get_slug_column
from climate_ref.executor.result_handling import ingest_execution_result
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
    execution_datasets,
    get_execution_group_and_latest_filtered,
)
from climate_ref.models.metric_value import MetricValue
from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.pycmec.controlled_vocabulary import CV

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


def _copy_with_backup(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy *src* to *dst*, backing up *dst* first if it already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        backup = dst.with_suffix(dst.suffix + ".bak")
        shutil.copy2(dst, backup)
    shutil.copyfile(src, dst)


def _sync_reingest_files_to_results(
    config: "Config",
    src_fragment: str,
    dst_fragment: str,
    result: ExecutionResult,
    mode: ReingestMode,
) -> None:
    """
    Copy re-extracted CMEC bundles from scratch to the results directory.

    For **versioned** mode the entire scratch tree is copied to a new results
    directory (no backup needed).  For other modes only the CMEC bundle files
    are overwritten, with ``.bak`` copies of the previous versions.
    """
    src_dir = config.paths.scratch / src_fragment
    dst_dir = config.paths.results / dst_fragment

    if mode == ReingestMode.versioned:
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        assert result.metric_bundle_filename is not None
        _copy_with_backup(src_dir / result.metric_bundle_filename, dst_dir / result.metric_bundle_filename)
        if result.output_bundle_filename:
            _copy_with_backup(
                src_dir / result.output_bundle_filename, dst_dir / result.output_bundle_filename
            )
        if result.series_filename:
            _copy_with_backup(src_dir / result.series_filename, dst_dir / result.series_filename)


def _delete_execution_metric_values(database: "Database", execution: Execution) -> None:
    """Delete all metric values for an execution (outputs are handled separately)."""
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
            # Hash is intentionally deterministic for a given execution:
            # re-running versioned reingest on the same execution produces the
            # same base fragment, with _n{count} suffixes for repeats.
            version_hash = hashlib.sha1(  # noqa: S324
                f"{execution.output_fragment}-reingest-{execution.id}".encode()
            ).hexdigest()[:12]

            base_fragment = f"{execution.output_fragment}_v{version_hash}"
            existing_count = sum(
                1 for e in execution_group.executions if e.output_fragment.startswith(base_fragment)
            )
            version_suffix = f"_n{existing_count + 1}" if existing_count > 0 else ""

            target_execution = Execution(
                execution_group=execution_group,
                dataset_hash=execution.dataset_hash,
                output_fragment=f"{base_fragment}{version_suffix}",
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
            _delete_execution_metric_values(database, target_execution)

        # Outputs are always refreshed regardless of mode.
        # "Additive" only applies to metric values as keeping stale entries
        # would be more confusing than helpful.
        if result.output_bundle_filename:
            database.session.execute(
                delete(ExecutionOutput).where(ExecutionOutput.execution_id == target_execution.id)
            )

        existing = (
            _get_existing_metric_dimensions(database, target_execution)
            if mode == ReingestMode.additive
            else None
        )

        ingest_execution_result(
            database,
            target_execution,
            result,
            cv,
            output_base_path=config.paths.results / target_execution.output_fragment,
            output_fallback_path=config.paths.scratch / execution.output_fragment,
            existing_metrics=existing,
        )

        assert result.metric_bundle_filename is not None
        target_execution.mark_successful(result.as_relative_path(result.metric_bundle_filename))

    return target_execution


def reingest_execution(
    config: "Config",
    database: "Database",
    execution: Execution,
    provider_registry: "ProviderRegistry",
    mode: ReingestMode = ReingestMode.additive,
) -> bool:
    """
    Reingest an existing execution.

    Re-runs ``build_execution_result()`` against the scratch directory
    (which contains the raw outputs from the original diagnostic run) and re-ingests
    the results into the database.
    Updated CMEC bundles are then copied from scratch to the results directory,
    backing up previous versions.

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

    scratch_dir = config.paths.scratch / execution.output_fragment
    if not scratch_dir.exists():
        logger.error(f"Scratch directory does not exist: {scratch_dir}. Skipping execution {execution.id}.")
        return False

    definition = reconstruct_execution_definition(config, execution, diagnostic)

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

    # Copy re-extracted CMEC bundles from scratch to the results tree.
    # For non-versioned modes, existing files are backed up with a .bak suffix.
    _sync_reingest_files_to_results(
        config, execution.output_fragment, target_execution.output_fragment, result, mode
    )

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
