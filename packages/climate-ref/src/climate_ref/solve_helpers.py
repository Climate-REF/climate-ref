"""
Helpers for understanding and regression-testing the solver's behavior.

This module provides functions to:
- Generate parquet catalogs from local dataset directories
- Load parquet catalogs for solver testing
- Run the solver on catalogs and format results
- Produce regression-friendly output for pytest-regressions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from climate_ref.datasets import get_dataset_adapter
from climate_ref.provider_registry import ProviderRegistry
from climate_ref.solver import ExecutionSolver, SolveFilterOptions
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.providers import DiagnosticProvider


def generate_catalog(
    source_type: str,
    directories: list[Path],
    strip_path_prefix: str | None = None,
) -> pd.DataFrame:
    """
    Scan directories using the appropriate DatasetAdapter and concatenate results.

    Parameters
    ----------
    source_type
        Dataset source type (e.g. "cmip6", "obs4mips")
    directories
        List of directories to scan for datasets
    strip_path_prefix
        If provided, replace this prefix in path columns with ``{data_dir}``
        for portability

    Returns
    -------
    :
        DataFrame containing dataset metadata from all directories
    """
    adapter = get_dataset_adapter(source_type)
    frames = []
    for directory in directories:
        try:
            df = adapter.find_local_datasets(directory)
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.debug(f"Skipping directory {directory}: {exc}")
            continue
        if len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    catalog = pd.concat(frames, ignore_index=True)

    if strip_path_prefix and "path" in catalog.columns:
        catalog["path"] = (
            catalog["path"].astype(str).str.replace(strip_path_prefix, "{data_dir}", regex=False)
        )

    return catalog


def write_catalog_parquet(catalog: pd.DataFrame, output_path: Path) -> None:
    """
    Write a catalog DataFrame to parquet.

    Parameters
    ----------
    catalog
        DataFrame to write
    output_path
        Path for the output parquet file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(output_path, index=False)


def load_solve_catalog(catalog_dir: Path) -> dict[SourceDatasetType, pd.DataFrame] | None:
    """
    Load parquet catalog files from a directory.

    Looks for ``cmip6_catalog.parquet``, ``cmip7_catalog.parquet``,
    ``obs4mips_catalog.parquet``, and ``pmp_climatology_catalog.parquet``.

    Parameters
    ----------
    catalog_dir
        Directory containing parquet catalog files

    Returns
    -------
    :
        Mapping of source type to catalog DataFrame, or None if no catalogs found
    """
    if not catalog_dir.exists():
        return None

    catalog_files = {
        SourceDatasetType.CMIP6: "cmip6_catalog.parquet",
        SourceDatasetType.CMIP7: "cmip7_catalog.parquet",
        SourceDatasetType.obs4MIPs: "obs4mips_catalog.parquet",
        SourceDatasetType.PMPClimatology: "pmp_climatology_catalog.parquet",
    }

    result: dict[SourceDatasetType, pd.DataFrame] = {}
    for source_type, filename in catalog_files.items():
        path = catalog_dir / filename
        if path.exists():
            result[source_type] = pd.read_parquet(path)

    return result if result else None


def solve_to_results(
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
    providers: list[DiagnosticProvider],
    filters: SolveFilterOptions | None = None,
) -> list[dict[str, Any]]:
    """
    Run the solver on a data catalog and collect results into a sorted list of dicts.

    Parameters
    ----------
    data_catalog
        Mapping of source type to catalog DataFrame
    providers
        List of diagnostic providers to solve for
    filters
        Optional filters to restrict which diagnostics are solved

    Returns
    -------
    :
        Sorted list of result dicts, each with keys: ``provider``, ``diagnostic``,
        ``dataset_key``, ``selectors``, ``datasets``
    """
    registry = ProviderRegistry(providers=providers)
    solver = ExecutionSolver(provider_registry=registry, data_catalog=data_catalog)

    results = []
    for execution in solver.solve(filters=filters):
        datasets: dict[str, list[str]] = {}
        for source_type, ds_collection in execution.datasets.items():
            instance_ids = sorted(ds_collection.instance_id.unique().tolist())
            datasets[str(source_type.value)] = instance_ids

        results.append(
            {
                "provider": execution.provider.slug,
                "diagnostic": execution.diagnostic.slug,
                "dataset_key": execution.dataset_key,
                "selectors": execution.selectors,
                "datasets": datasets,
            }
        )

    results.sort(key=lambda r: (r["provider"], r["diagnostic"], r["dataset_key"]))
    return results


def format_solve_results_table(results: list[dict[str, Any]]) -> str:
    """
    Format solve results as a human-readable grouped text table.

    Groups by provider, then diagnostic, showing dataset_key and matched
    instance_ids per source type.

    Parameters
    ----------
    results
        Results from :func:`solve_to_results`

    Returns
    -------
    :
        Human-readable text representation
    """
    if not results:
        return "No executions found."

    lines: list[str] = []
    current_provider = None
    current_diagnostic = None
    diagnostic_count = set()
    provider_count = set()

    for r in results:
        provider_count.add(r["provider"])
        diagnostic_count.add((r["provider"], r["diagnostic"]))

        if r["provider"] != current_provider:
            current_provider = r["provider"]
            current_diagnostic = None
            lines.append(f"\n## {current_provider}")

        if r["diagnostic"] != current_diagnostic:
            current_diagnostic = r["diagnostic"]
            lines.append(f"\n  ### {current_diagnostic}")

        lines.append(f"    {r['dataset_key']}")
        for source_type, instance_ids in sorted(r["datasets"].items()):
            for iid in instance_ids:
                lines.append(f"      [{source_type}] {iid}")

    lines.append("")
    lines.append(
        f"Summary: {len(results)} executions, "
        f"{len(diagnostic_count)} diagnostics, "
        f"{len(provider_count)} providers"
    )
    return "\n".join(lines)


def format_solve_results_json(results: list[dict[str, Any]]) -> str:
    """
    Serialize solve results to JSON.

    Parameters
    ----------
    results
        Results from :func:`solve_to_results`

    Returns
    -------
    :
        JSON string of the results list
    """
    # Convert selectors (which contain tuples) to JSON-serializable form
    serializable = []
    for r in results:
        entry = dict(r)
        entry["selectors"] = {k: [[dim, val] for dim, val in v] for k, v in r["selectors"].items()}
        serializable.append(entry)
    return json.dumps(serializable, indent=2, sort_keys=True)


def solve_results_for_regression(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """
    Convert solve results to the dict format used by ``data_regression.check()``.

    Produces ``{dataset_key: {source_type: [instance_id, ...]}}``
    for use with ``data_regression.check()``.

    When called with results filtered to a single diagnostic (recommended),
    ``dataset_key`` is unique and no data is lost. If results span multiple
    diagnostics, duplicate ``dataset_key`` values will overwrite earlier entries.

    Parameters
    ----------
    results
        Results from :func:`solve_to_results`, ideally filtered to one diagnostic

    Returns
    -------
    :
        Dict keyed by ``dataset_key`` with source_type -> instance_id list values
    """
    output: dict[str, dict[str, list[str]]] = {}
    for r in results:
        output[r["dataset_key"]] = r["datasets"]
    return output
