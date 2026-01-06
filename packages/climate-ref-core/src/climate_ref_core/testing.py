"""
Test infrastructure classes for diagnostic testing.

This module provides classes for specifying test data requirements
and test cases for diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from attrs import field, frozen
from loguru import logger

from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    Selector,
    SourceDatasetType,
)
from climate_ref_core.esgf.base import ESGFRequest

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import Diagnostic


@frozen
class TestCase:
    """
    A single test case for a diagnostic.

    Test cases define specific scenarios for testing a diagnostic,
    including what datasets to use and optionally how to fetch them.
    """

    name: str
    """Name of the test case (e.g., 'default', 'edge-case-short-timeseries')."""

    description: str
    """Human-readable description of what this test case covers."""

    requests: tuple[ESGFRequest, ...] | None = None
    """Optional ESGF requests to fetch data for this test case."""

    datasets: ExecutionDatasetCollection | None = None
    """Explicit datasets to use (optional, takes precedence over datasets_file)."""

    datasets_file: str | None = None
    """Path to YAML file with dataset specification (relative to package)."""

    def resolve_datasets(
        self,
        data_catalog: dict[SourceDatasetType, pd.DataFrame] | None,
        diagnostic: Diagnostic,
        package_dir: Path | None = None,
    ) -> ExecutionDatasetCollection:
        """
        Resolve datasets from file, inline, or by solving.

        Parameters
        ----------
        data_catalog
            Data catalog to use for solving (if needed)
        diagnostic
            The diagnostic this test case is for
        package_dir
            Base directory for resolving relative dataset file paths

        Returns
        -------
        ExecutionDatasetCollection
            The resolved datasets for this test case
        """
        # 1. Explicit datasets take precedence
        if self.datasets is not None:
            return self.datasets

        # 2. Load from YAML file
        if self.datasets_file is not None:
            if package_dir is None:
                raise ValueError("package_dir required when using datasets_file")
            return load_datasets_from_yaml(package_dir / self.datasets_file)

        # 3. Fall back to solving
        if data_catalog is None:
            raise ValueError(
                f"Cannot resolve datasets for test case {self.name!r}: "
                "no explicit datasets, no datasets_file, and no data_catalog provided"
            )

        return solve_test_case(diagnostic, data_catalog)


@frozen
class TestDataSpecification:
    """
    Test data specification for a diagnostic.

    Contains multiple named test cases that can be used to test
    different scenarios for the diagnostic.
    """

    test_cases: tuple[TestCase, ...] = field(factory=tuple)
    """Collection of test cases for this diagnostic."""

    @property
    def default(self) -> TestCase:
        """
        Get the default test case.

        Returns
        -------
        TestCase
            The test case named 'default'

        Raises
        ------
        StopIteration
            If no default test case exists
        """
        return next(tc for tc in self.test_cases if tc.name == "default")

    def get_case(self, name: str) -> TestCase:
        """
        Get a test case by name.

        Parameters
        ----------
        name
            Name of the test case to retrieve

        Returns
        -------
        TestCase
            The matching test case

        Raises
        ------
        StopIteration
            If no test case with that name exists
        """
        return next(tc for tc in self.test_cases if tc.name == name)

    def has_case(self, name: str) -> bool:
        """
        Check if a test case with the given name exists.

        Parameters
        ----------
        name
            Name of the test case to check

        Returns
        -------
        bool
            True if the test case exists
        """
        return any(tc.name == name for tc in self.test_cases)

    @property
    def case_names(self) -> list[str]:
        """Get names of all test cases."""
        return [tc.name for tc in self.test_cases]


def load_datasets_from_yaml(path: Path) -> ExecutionDatasetCollection:
    """
    Load ExecutionDatasetCollection from a YAML file.

    The YAML file should have the following structure:

    ```yaml
    cmip6:  # or obs4mips, etc.
      slug_column: instance_id
      selector:
        source_id: ACCESS-ESM1-5
        member_id: r1i1p1f1
      datasets:
        - instance_id: CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.tas.gn.v20191115
          path: /path/to/file.nc
          variable_id: tas
          # ... other metadata
    ```

    Parameters
    ----------
    path
        Path to the YAML file

    Returns
    -------
    ExecutionDatasetCollection
        The loaded datasets
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    collections: dict[SourceDatasetType | str, DatasetCollection] = {}

    for source_type_str, source_data in data.items():
        source_type = SourceDatasetType(source_type_str)

        # Convert selector dict to tuple of tuples
        selector_dict = source_data.get("selector", {})
        selector: Selector = tuple(sorted(selector_dict.items()))

        # Convert datasets list to DataFrame
        datasets_list = source_data.get("datasets", [])
        datasets_df = pd.DataFrame(datasets_list)

        slug_column = source_data.get("slug_column", "instance_id")

        collections[source_type] = DatasetCollection(
            datasets=datasets_df,
            slug_column=slug_column,
            selector=selector,
        )

    return ExecutionDatasetCollection(collections)


def save_datasets_to_yaml(datasets: ExecutionDatasetCollection, path: Path) -> None:
    """
    Save ExecutionDatasetCollection to a YAML file.

    Parameters
    ----------
    datasets
        The datasets to save
    path
        Path to write the YAML file
    """
    data: dict[str, Any] = {}

    for source_type, collection in datasets.items():
        source_data: dict[str, Any] = {
            "slug_column": collection.slug_column,
            "selector": dict(collection.selector),
            "datasets": collection.datasets.to_dict(orient="records"),
        }
        data[source_type.value] = source_data

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved datasets to {path}")


def solve_test_case(
    diagnostic: Diagnostic,
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
) -> ExecutionDatasetCollection:
    """
    Solve for test case datasets by applying the diagnostic's data requirements.

    This function runs the solver to determine which datasets from the catalog
    satisfy the diagnostic's requirements.

    Parameters
    ----------
    diagnostic
        The diagnostic to solve for
    data_catalog
        Data catalog mapping source types to DataFrames

    Returns
    -------
    ExecutionDatasetCollection
        The solved datasets

    Raises
    ------
    ValueError
        If no valid execution can be found
    """
    # Import here to avoid circular dependency
    # The solver lives in climate-ref, not climate-ref-core
    # This function may need to be moved or the import resolved differently
    try:
        from climate_ref.solver import solve_executions  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "solve_test_case requires climate-ref package. Use explicit datasets or datasets_file instead."
        )

    executions = list(solve_executions(data_catalog, diagnostic, diagnostic.provider))

    if not executions:
        raise ValueError(f"No valid executions found for diagnostic {diagnostic.slug}")

    # Return the first execution's datasets
    return executions[0].datasets
