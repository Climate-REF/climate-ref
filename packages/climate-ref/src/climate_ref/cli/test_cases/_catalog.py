"""
Catalog construction helpers shared by the ``fetch`` and ``run`` commands.

These turn a diagnostic's test-data requests into a solved
:class:`~climate_ref_core.datasets.ExecutionDatasetCollection` and persist the
input catalog YAML next to the test case.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from climate_ref.datasets import get_dataset_adapter
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.exceptions import DatasetResolutionError

if TYPE_CHECKING:
    import pandas as pd

    from climate_ref.datasets import DatasetAdapter
    from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.esgf import ESGFRequest
    from climate_ref_core.testing import TestCase


def _build_catalog(dataset_adapter: DatasetAdapter, file_paths: list[Path]) -> pd.DataFrame:
    """
    Parse a list of datasets using a dataset adapter

    Parameters
    ----------
    file_paths
        List of files to build a catalog from

    Returns
    -------
    pd.DataFrame
        DataFrame catalog of datasets with metadata and paths
    """
    import pandas as pd

    # Collect unique parent directories since the adapter scans directories.
    # Sort for deterministic traversal/catalog order so baseline selection
    # (``_solve_test_case`` takes ``executions[0]``) is reproducible.
    parent_dirs = sorted({fp.parent for fp in file_paths}, key=lambda p: p.as_posix())

    catalog_dfs = []
    for parent_dir in parent_dirs:
        try:
            df = dataset_adapter.find_local_datasets(parent_dir)

            # Filter to only include the files we fetched
            fetched_files = {str(fp) for fp in file_paths}
            df = df[df["path"].isin(fetched_files)]
            if df.empty:
                logger.warning(f"No matching files found in catalog for {parent_dir}")
            catalog_dfs.append(df)
        except Exception as e:
            raise DatasetResolutionError(f"Failed to parse fetched datasets in {parent_dir}") from e

    if not catalog_dfs:
        return pd.DataFrame()
    return pd.concat(catalog_dfs, ignore_index=True)


def _pin_requests_to_catalog(
    requests: tuple[ESGFRequest, ...],
    recorded: dict[str, tuple[dict[str, Any], ...]],
) -> tuple[tuple[ESGFRequest, ...], set[str]]:
    """
    Pin a test case's requests to the datasets its catalog already records.

    A request the catalog records nothing for, or whose records do not carry what its pin
    needs, is kept as-is and resolves from its declared facets.

    Parameters
    ----------
    requests
        The test case's requests
    recorded
        The datasets recorded in the catalog, keyed by source type name

    Returns
    -------
    :
        Tuple of (requests, the source types that were pinned)
    """
    pinned_requests: list[ESGFRequest] = []
    pinned_source_types: set[str] = set()

    for request in requests:
        datasets = recorded.get(request.source_type)
        if not datasets:
            logger.debug(f"    No recorded datasets for {request.slug} ({request.source_type})")
            pinned_requests.append(request)
            continue

        pinned = request.pin_to_datasets(datasets)
        pinned_requests.append(pinned)
        if pinned is not request:
            logger.info(f"    Pinning {request.slug} to {len(datasets)} recorded dataset(s)")
            pinned_source_types.add(request.source_type)

    return tuple(pinned_requests), pinned_source_types


def _check_pinned_datasets_found(
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
    recorded: dict[str, tuple[dict[str, Any], ...]],
    pinned_source_types: set[str],
) -> None:
    """
    Check that every pinned dataset made it into the catalog.

    A dataset can go missing for more than one reason -- it may no longer be published,
    but it may equally have failed to download or, for CMIP7, to convert. Those failures
    are reported where they happen, so the error raised here points at them rather than
    naming a cause it cannot know.

    Raises
    ------
    DatasetResolutionError
        If a recorded dataset is missing, which would otherwise silently shrink the
        catalog and invalidate the test case's regression baseline.
    """
    for source_type_name in sorted(pinned_source_types):
        source = SourceDatasetType[source_type_name]
        catalog = data_catalog.get(source)
        slug_column = get_dataset_adapter(source.value).slug_column
        found = set(catalog[slug_column]) if catalog is not None and not catalog.empty else set()

        missing = [
            str(dataset[slug_column])
            for dataset in recorded[source_type_name]
            if dataset.get(slug_column) not in found
        ]
        if missing:
            raise DatasetResolutionError(
                f"{len(missing)} dataset(s) recorded in the catalog are missing from the "
                f"fetched data: {', '.join(missing)}. If the recorded datasets are "
                "out of date, re-resolve them from their declared facets with --regen."
            )


def _solve_test_case(
    diagnostic: Diagnostic,
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
) -> ExecutionDatasetCollection:
    """
    Solve for test case datasets by applying the diagnostic's data requirements.

    Runs the solver to determine which datasets from the catalog
    satisfy the diagnostic's requirements.
    """
    from climate_ref.solver import solve_executions

    executions = list(solve_executions(data_catalog, diagnostic, diagnostic.provider))

    if not executions:
        raise ValueError(f"No valid executions found for diagnostic {diagnostic.slug}")

    return executions[0].datasets


def _fetch_and_build_catalog(
    diag: Diagnostic,
    tc: TestCase,
    *,
    force: bool = False,
    regen: bool = False,
) -> tuple[ExecutionDatasetCollection, bool]:
    """
    Fetch test data and build catalog.

    This function:
    1. Fetches ESGF data using ESGFFetcher (files stored in intake-esgf cache)
    2. Uses CMIP6DatasetAdapter to create a data catalog
    3. Solves for datasets using the diagnostic's data requirements
    4. Writes catalog YAML to .catalogs/{provider}/{diagnostic}/{test_case}.yaml
    5. Returns the solved datasets and whether the catalog was written

    If the test case already has a catalog, its requests are pinned to the datasets that
    catalog records rather than re-resolved from the facets they declare, so a test case
    keeps the inputs its regression baseline was built from as ESGF publishes new
    versions. Use `regen=True` to resolve the declared facets again.

    By default, the catalog is only written if the content has changed.
    Use `force=True` to always write.

    Parameters
    ----------
    diag
        The diagnostic to fetch data for
    tc
        The test case to fetch data for
    force
        If True, always write the catalog even if unchanged
    regen
        If True, ignore an existing catalog's datasets and resolve the requests from
        their declared facets. This may change which datasets the test case uses.

    Returns
    -------
    :
        Tuple of (datasets, catalog_was_written)
    """
    from climate_ref.datasets import get_dataset_adapter
    from climate_ref_core.datasets import SourceDatasetType
    from climate_ref_core.esgf import ESGFFetcher
    from climate_ref_core.testing import (
        TestCasePaths,
        load_catalog_datasets,
        save_datasets_to_yaml,
    )

    fetcher = ESGFFetcher()

    paths = TestCasePaths.from_diagnostic(diag, tc.name)

    # Reuse the datasets an existing catalog records instead of the declared facets,
    # so repeating a fetch resolves the same data that the baseline was built from.
    recorded: dict[str, tuple[dict[str, Any], ...]] = {}
    if not regen and paths is not None and paths.catalog.exists():
        recorded = load_catalog_datasets(paths.catalog)

    requests = tc.requests or ()
    pinned_source_types: set[str] = set()
    if recorded:
        requests, pinned_source_types = _pin_requests_to_catalog(requests, recorded)

    # Fetch all requests - returns DataFrame with metadata + paths
    combined = fetcher.fetch_for_test_case(requests)

    if combined.empty:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Group paths by source type and use adapters to build proper catalog
    data_catalog: dict[SourceDatasetType, pd.DataFrame] = {}

    for source_type, group_df in combined.groupby("source_type"):
        # Requests name a source type by its enum name, adapters are keyed by its value.
        # An unrecognised name is skipped, and falls out as the "no datasets" error below.
        if source_type not in SourceDatasetType.__members__:
            continue

        file_paths = [Path(p) for p in group_df["path"].unique().tolist()]
        source = SourceDatasetType[str(source_type)]
        data_catalog[source] = _build_catalog(get_dataset_adapter(source.value), file_paths)

    if not data_catalog:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    _check_pinned_datasets_found(data_catalog, recorded, pinned_source_types)

    # Solve for datasets
    datasets = _solve_test_case(diag, data_catalog)

    # Write the catalog to the package-local test case directory,
    # and the local-paths sidecar to the dataset cache.
    catalog_written = False
    if paths:
        paths.create()
        catalog_written = save_datasets_to_yaml(datasets, paths.catalog, paths.catalog_paths, force=force)

    return datasets, catalog_written
