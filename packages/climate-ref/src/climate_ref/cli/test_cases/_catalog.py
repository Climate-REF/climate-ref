"""
Catalog construction helpers shared by the ``fetch`` and ``run`` commands.

These turn a diagnostic's test-data requests into a solved
:class:`~climate_ref_core.datasets.ExecutionDatasetCollection` and persist the
input catalog YAML next to the test case.
"""

# ``from __future__ import annotations`` keeps the heavy signature types
# (pandas, dataset adapters, diagnostics) under ``TYPE_CHECKING`` so importing
# this module -- which happens on every ``ref`` invocation -- stays cheap.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from climate_ref_core.exceptions import DatasetResolutionError

if TYPE_CHECKING:
    import pandas as pd

    from climate_ref.datasets import DatasetAdapter
    from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
    from climate_ref_core.diagnostics import Diagnostic
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
) -> tuple[ExecutionDatasetCollection, bool]:
    """
    Fetch test data and build catalog.

    This function:
    1. Fetches ESGF data using ESGFFetcher (files stored in intake-esgf cache)
    2. Uses CMIP6DatasetAdapter to create a data catalog
    3. Solves for datasets using the diagnostic's data requirements
    4. Writes catalog YAML to .catalogs/{provider}/{diagnostic}/{test_case}.yaml
    5. Returns the solved datasets and whether the catalog was written

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

    Returns
    -------
    :
        Tuple of (datasets, catalog_was_written)
    """
    from climate_ref.datasets import (
        CMIP6DatasetAdapter,
        CMIP7DatasetAdapter,
        Obs4MIPsDatasetAdapter,
        PMPClimatologyDatasetAdapter,
    )
    from climate_ref_core.datasets import SourceDatasetType
    from climate_ref_core.esgf import ESGFFetcher
    from climate_ref_core.testing import TestCasePaths, save_datasets_to_yaml

    fetcher = ESGFFetcher()

    # Fetch all requests - returns DataFrame with metadata + paths
    combined = fetcher.fetch_for_test_case(tc.requests)

    if combined.empty:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Group paths by source type and use adapters to build proper catalog
    data_catalog: dict[SourceDatasetType, pd.DataFrame] = {}

    for source_type, group_df in combined.groupby("source_type"):
        file_paths = [Path(p) for p in group_df["path"].unique().tolist()]

        if source_type == "CMIP6":
            data_catalog[SourceDatasetType.CMIP6] = _build_catalog(CMIP6DatasetAdapter(), file_paths)

        elif source_type == "CMIP7":
            data_catalog[SourceDatasetType.CMIP7] = _build_catalog(CMIP7DatasetAdapter(), file_paths)

        elif source_type == "obs4MIPs":
            data_catalog[SourceDatasetType.obs4MIPs] = _build_catalog(Obs4MIPsDatasetAdapter(), file_paths)

        elif source_type == "PMPClimatology":
            data_catalog[SourceDatasetType.PMPClimatology] = _build_catalog(
                PMPClimatologyDatasetAdapter(), file_paths
            )

    if not data_catalog:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Solve for datasets
    datasets = _solve_test_case(diag, data_catalog)

    # Write catalog YAML to package-local test case directory
    catalog_written = False
    paths = TestCasePaths.from_diagnostic(diag, tc.name)
    if paths:
        paths.create()
        catalog_written = save_datasets_to_yaml(datasets, paths.catalog, force=force)

    return datasets, catalog_written
