"""
Test infrastructure for diagnostic testing.

This module provides:
- TestCase and TestDataSpecification for defining test scenarios
- YAML serialization for dataset catalogs (with paths stored separately)
- RegressionValidator for validating pre-stored outputs
- Utilities for CMEC bundle validation
"""

from __future__ import annotations

import shutil
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
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.esgf.base import ESGFRequest
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import Diagnostic


@frozen
class TestCase:
    """
    A single test case for a diagnostic.

    Test cases define scenarios for testing, with data resolved via:
    - `requests`: ESGF requests to fetch data (use `ref test-cases fetch`)
    - `datasets_file`: Path to a pre-built catalog YAML file
    """

    name: str
    """Name of the test case (e.g., 'default', 'short-timeseries')."""

    description: str
    """Human-readable description of what this test case covers."""

    requests: tuple[ESGFRequest, ...] | None = None
    """Optional ESGF requests to fetch data for this test case."""

    datasets_file: str | None = None
    """Path to YAML file with dataset specification (relative to package)."""


@frozen
class TestDataSpecification:
    """
    Test data specification for a diagnostic.

    Contains multiple named test cases for testing different input datasets.
    """

    test_cases: tuple[TestCase, ...] = field(factory=tuple)
    """Collection of test cases for this diagnostic."""

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


def _get_paths_file(catalog_path: Path) -> Path:
    """Get the paths file path for a catalog file."""
    return catalog_path.with_suffix(".paths.yaml")


def load_datasets_from_yaml(path: Path) -> ExecutionDatasetCollection:
    """
    Load ExecutionDatasetCollection from a YAML file.

    The YAML file structure:

    ```yaml
    cmip6:
      slug_column: instance_id
      selector:
        source_id: ACCESS-ESM1-5
      datasets:
        - instance_id: CMIP6.CMIP...
          variable_id: tas
          # ... other metadata
    ```

    Paths are loaded from a separate `.paths.yaml` file if it exists,
    allowing the main catalog to be version-controlled while paths
    remain user-specific.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Load paths from separate file if it exists
    paths_file = _get_paths_file(path)
    paths_map: dict[str, str] = {}
    if paths_file.exists():
        with open(paths_file) as f:
            paths_map = yaml.safe_load(f) or {}

    collections: dict[SourceDatasetType | str, DatasetCollection] = {}

    for source_type_str, source_data in data.items():
        source_type = SourceDatasetType(source_type_str)
        selector_dict = source_data.get("selector", {})
        selector: Selector = tuple(sorted(selector_dict.items()))
        datasets_list = source_data.get("datasets", [])
        slug_column = source_data.get("slug_column", "instance_id")

        # Merge paths from paths file
        for dataset in datasets_list:
            instance_id = dataset.get(slug_column)
            if instance_id and instance_id in paths_map:
                dataset["path"] = paths_map[instance_id]

        collections[source_type] = DatasetCollection(
            datasets=pd.DataFrame(datasets_list),
            slug_column=slug_column,
            selector=selector,
        )

    return ExecutionDatasetCollection(collections)


def save_datasets_to_yaml(datasets: ExecutionDatasetCollection, path: Path) -> None:
    """
    Save ExecutionDatasetCollection to a YAML file.

    Paths are saved to a separate `.paths.yaml` file to allow the main
    catalog to be version-controlled while paths remain user-specific.

    Parameters
    ----------
    datasets
        The datasets to save
    path
        Path to write the YAML file
    """
    data: dict[str, Any] = {}
    paths_map: dict[str, str] = {}

    for source_type, collection in datasets.items():
        slug_column = collection.slug_column
        datasets_records = collection.datasets.to_dict(orient="records")

        # Extract paths to separate map
        for record in datasets_records:
            instance_id = record.get(slug_column)
            if instance_id and "path" in record:
                paths_map[instance_id] = record.pop("path")

        data[source_type.value] = {
            "slug_column": slug_column,
            "selector": dict(collection.selector),
            "datasets": datasets_records,
        }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    paths_file = _get_paths_file(path)
    with open(paths_file, "w") as f:
        yaml.dump(paths_map, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved catalog to {path} (paths: {paths_file})")


def get_test_case_regression_path(
    regression_data_dir: Path,
    provider_slug: str,
    diagnostic_slug: str,
    test_case_name: str,
) -> Path:
    """Get path to regression data for a test case."""
    return regression_data_dir / provider_slug / diagnostic_slug / test_case_name


def validate_cmec_bundles(diagnostic: Diagnostic, result: ExecutionResult) -> None:
    """
    Validate CMEC bundles in an execution result.

    Performs structural validation of the metric and output bundles.

    Raises
    ------
    AssertionError
        If the result is not successful or bundles are invalid
    """
    assert result.successful, f"Execution failed: {result}"

    # Validate metric bundle
    metric_bundle = CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))
    CMECMetric.model_validate(metric_bundle)

    # Check dimensions match diagnostic facets
    bundle_dimensions = tuple(metric_bundle.DIMENSIONS.root["json_structure"])
    assert diagnostic.facets == bundle_dimensions, (
        f"Bundle dimensions {bundle_dimensions} don't match diagnostic facets {diagnostic.facets}"
    )

    # Validate output bundle
    CMECOutput.load_from_json(result.to_output_path(result.output_bundle_filename))


@frozen
class RegressionValidator:
    """
    Validate diagnostic outputs from pre-stored regression data.

    Loads regression outputs and validates CMEC bundles without
    running the diagnostic. Suitable for fast CI validation.
    """

    diagnostic: Diagnostic
    test_case_name: str
    regression_data_dir: Path
    test_data_dir: Path | None = None

    def regression_path(self) -> Path:
        """Get path to regression data for this test case."""
        return get_test_case_regression_path(
            self.regression_data_dir,
            self.diagnostic.provider.slug,
            self.diagnostic.slug,
            self.test_case_name,
        )

    def has_regression_data(self) -> bool:
        """Check if regression data exists for this test case."""
        path = self.regression_path()
        return path.exists() and (path / "diagnostic.json").exists()

    def load_regression_definition(self, tmp_dir: Path) -> ExecutionDefinition:
        """
        Load regression data and create an ExecutionDefinition.

        Copies regression data to tmp_dir and replaces path placeholders.
        """
        regression_path = self.regression_path()
        if not regression_path.exists():
            raise FileNotFoundError(
                f"No regression data at {regression_path}. Run 'ref test-cases run --force-regen' first."
            )

        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(regression_path, output_dir, dirs_exist_ok=True)

        # Replace placeholders with actual paths
        test_data_dir_str = str(self.test_data_dir) if self.test_data_dir else ""
        for pattern in ("*.json", "*.txt", "*.yaml", "*.yml"):
            for file in output_dir.rglob(pattern):
                content = file.read_text()
                content = content.replace("<OUTPUT_DIR>", str(output_dir))
                content = content.replace("<TEST_DATA_DIR>", test_data_dir_str)
                file.write_text(content)

        return ExecutionDefinition(
            diagnostic=self.diagnostic,
            key=f"test-{self.test_case_name}",
            datasets=ExecutionDatasetCollection({}),
            output_directory=output_dir,
            root_directory=tmp_dir,
        )

    def validate(self, definition: ExecutionDefinition) -> None:
        """Validate CMEC bundles in the regression output."""
        result = self.diagnostic.build_execution_result(definition)
        result.to_output_path("out.log").touch()  # Log file not tracked in regression
        validate_cmec_bundles(self.diagnostic, result)
