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
import sys
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
    from _pytest.mark.structures import ParameterSet

    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.providers import DiagnosticProvider


class NetworkBlockedError(Exception):
    """Raised when network access is attempted while blocked."""

    pass


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


@frozen
class TestCasePaths:
    """
    Path resolver for test case data.

    Provides access to all paths within a test case directory:
    - catalog.yaml: Dataset metadata (tracked in git)
    - catalog.paths.yaml: Local file paths (gitignored)
    - regression/: Regression outputs (tracked in git)

    Can be constructed from:
    - A diagnostic + test case name (auto-resolves provider's test-data dir)
    - An explicit test_data_dir + diagnostic slug + test case name
    """

    root: Path
    """The test case directory (test_data_dir / diagnostic_slug / test_case_name)."""

    @classmethod
    def from_diagnostic(cls, diagnostic: Diagnostic, test_case: str) -> TestCasePaths | None:
        """
        Create from a diagnostic, auto-resolving the provider's test-data directory.

        Returns None if the provider's test-data directory cannot be determined
        (e.g., not a development checkout).

        Parameters
        ----------
        diagnostic
            The diagnostic to get paths for
        test_case
            Test case name (e.g., 'default')
        """
        test_data_dir = _get_provider_test_data_dir(diagnostic)
        if test_data_dir is None:
            return None
        return cls(root=test_data_dir / diagnostic.slug / test_case)

    @classmethod
    def from_test_data_dir(
        cls,
        test_data_dir: Path,
        diagnostic_slug: str,
        test_case: str,
    ) -> TestCasePaths:
        """
        Create from an explicit test data directory.

        Use this when you have a test_data_dir fixture (in tests) or
        know the base path explicitly.

        Parameters
        ----------
        test_data_dir
            Base test data directory (e.g., from test fixture)
        diagnostic_slug
            The diagnostic slug
        test_case
            Test case name (e.g., 'default')
        """
        return cls(root=test_data_dir / diagnostic_slug / test_case)

    @property
    def catalog(self) -> Path:
        """Path to catalog.yaml."""
        return self.root / "catalog.yaml"

    @property
    def catalog_paths(self) -> Path:
        """Path to catalog.paths.yaml (gitignored, contains local file paths)."""
        return self.root / "catalog.paths.yaml"

    @property
    def regression(self) -> Path:
        """Path to regression/ directory."""
        return self.root / "regression"

    @property
    def regression_catalog_hash(self) -> Path:
        """Path to catalog hash file in regression directory."""
        return self.regression / ".catalog_hash"

    @property
    def test_data_dir(self) -> Path:
        """Path to the test-data directory (parent of diagnostic slug dir)."""
        return self.root.parent.parent

    def exists(self) -> bool:
        """Check if the test case directory exists."""
        return self.root.exists()

    def create(self) -> None:
        """Create the test case directory if it doesn't exist."""
        self.root.mkdir(parents=True, exist_ok=True)


def _get_provider_test_data_dir(diag: Diagnostic) -> Path | None:
    """
    Get the test-data directory for a provider's package.

    Returns packages/climate-ref-{provider}/tests/test-data/ or None if unavailable.
    This will only work if working in a development checkout of the package.

    Parameters
    ----------
    diag
        The diagnostic to get the test data dir for
    """
    # TODO: Simplify once providers are in their own packages

    # Use the diagnostic's module to determine the provider package
    diagnostic_module_name = diag.__class__.__module__.split(".")[0]
    logger.debug(f"Looking up test data dir for diagnostic module: {diagnostic_module_name}")

    if diagnostic_module_name not in sys.modules:
        logger.debug(f"Module {diagnostic_module_name} not in sys.modules")
        return None

    diagnostic_module = sys.modules[diagnostic_module_name]
    if not hasattr(diagnostic_module, "__file__") or diagnostic_module.__file__ is None:
        logger.debug(f"Module {diagnostic_module_name} has no __file__ attribute")
        return None

    # Module: packages/climate-ref-{slug}/src/climate_ref_{slug}/__init__.py
    # Target: packages/climate-ref-{slug}/tests/test-data/
    module_path = Path(diagnostic_module.__file__)
    package_root = module_path.parent.parent.parent  # src -> climate-ref-{slug}
    tests_dir = package_root / "tests"

    # Only return path if tests/ exists (dev checkout)
    if not tests_dir.exists():
        logger.debug(f"Tests dir does not exist (not a dev checkout): {tests_dir}")
        return None

    test_data_dir = tests_dir / "test-data"
    logger.debug(f"Diagnostic module path: {module_path}")
    logger.debug(f"Derived test data dir: {test_data_dir} (exists: {test_data_dir.exists()})")

    return test_data_dir


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
          filename: tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc
          # ... other metadata
    ```

    Paths are loaded from a separate `.paths.yaml` file if it exists,
    allowing the main catalog to be version-controlled while paths
    remain user-specific. Multi-file datasets have multiple rows with
    paths keyed by `{instance_id}::{filename}`.
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
        if source_type_str == "_metadata":
            continue  # Skip metadata section
        source_type = SourceDatasetType(source_type_str)
        selector_dict = source_data.get("selector", {})
        selector: Selector = tuple(sorted(selector_dict.items()))
        datasets_list = source_data.get("datasets", [])
        slug_column = source_data.get("slug_column", "instance_id")

        # Merge paths from paths file using composite key
        for dataset in datasets_list:
            instance_id = dataset.get(slug_column)
            filename = dataset.get("filename")
            if instance_id and filename:
                # Try composite key first (new format for multi-file datasets)
                composite_key = f"{instance_id}::{filename}"
                if composite_key in paths_map:
                    dataset["path"] = paths_map[composite_key]
                elif instance_id in paths_map:
                    # Fall back to simple key for backward compatibility
                    dataset["path"] = paths_map[instance_id]
            elif instance_id and instance_id in paths_map:
                # Legacy format without filename
                dataset["path"] = paths_map[instance_id]

        collections[source_type] = DatasetCollection(
            datasets=pd.DataFrame(datasets_list),
            slug_column=slug_column,
            selector=selector,
        )

    return ExecutionDatasetCollection(collections)


def get_catalog_hash(path: Path) -> str | None:
    """
    Get the hash stored in an existing catalog file.

    Parameters
    ----------
    path
        Path to the catalog YAML file

    Returns
    -------
    :
        The hash string if found, None if file doesn't exist or has no hash
    """
    if not path.exists():
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return None
    hash_value = data.get("_metadata", {}).get("hash")
    return str(hash_value) if hash_value is not None else None


def catalog_changed_since_regression(paths: TestCasePaths) -> bool:
    """
    Check if the catalog has changed since regression data was generated.

    Returns True if:
    - No regression data exists (new test case)
    - No stored catalog hash exists (legacy regression data)
    - The catalog hash differs from the stored one

    Parameters
    ----------
    paths
        TestCasePaths for the test case

    Returns
    -------
    :
        True if regression should be regenerated, False otherwise
    """
    if not paths.regression.exists():
        return True  # No regression data, needs to run
    if not paths.regression_catalog_hash.exists():
        return True  # No stored hash, needs to run
    if not paths.catalog.exists():
        return True  # No catalog file, needs to run

    stored_hash = paths.regression_catalog_hash.read_text().strip()
    current_hash = get_catalog_hash(paths.catalog)

    return stored_hash != current_hash


def save_datasets_to_yaml(
    datasets: ExecutionDatasetCollection,
    path: Path,
    *,
    force: bool = False,
) -> bool:
    """
    Save ExecutionDatasetCollection to a YAML file.

    Paths are saved to a separate `.paths.yaml` file to allow the main
    catalog to be version-controlled while paths remain user-specific.

    Multi-file datasets (e.g., time-chunked data) are stored as multiple rows,
    one per file. Paths are keyed by `{instance_id}::{filename}` to support
    multiple files per dataset.

    By default, the catalog is only written if the content has changed
    (detected via hash comparison). Use `force=True` to always write.

    Parameters
    ----------
    datasets
        The datasets to save
    path
        Path to write the YAML file
    force
        If True, always write the catalog even if unchanged

    Returns
    -------
    :
        True if the catalog was written, False if skipped (unchanged)
    """
    # Compute the hash first to check if we need to write
    new_hash = datasets.hash

    if not force:
        existing_hash = get_catalog_hash(path)
        if existing_hash == new_hash:
            logger.info(f"Catalog unchanged, skipping write: {path}")
            return False

    data: dict[str, Any] = {
        "_metadata": {"hash": new_hash},
    }
    paths_map: dict[str, str] = {}

    for source_type, collection in datasets.items():
        slug_column = collection.slug_column
        datasets_records = collection.datasets.to_dict(orient="records")

        # Extract paths to separate map, keeping all rows (including multi-file datasets)
        filtered_records = []
        for record in datasets_records:
            instance_id = record.get(slug_column)
            if instance_id and "path" in record:  # pragma: no branch
                file_path = record.pop("path")
                filename = Path(file_path).name
                # Store filename in record for matching when loading
                record["filename"] = filename
                # Use composite key to support multiple files per instance_id
                paths_map[f"{instance_id}::{filename}"] = file_path
                # Sort fields within each record alphabetically
                sorted_record = dict(sorted(record.items()))
                filtered_records.append(sorted_record)

        # Sort records by instance_id, then by filename for stability
        filtered_records.sort(key=lambda r: (r.get(slug_column, ""), r.get("filename", "")))

        data[source_type.value] = {
            "slug_column": slug_column,
            "selector": dict(collection.selector),
            "datasets": filtered_records,
        }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    paths_file = _get_paths_file(path)
    with open(paths_file, "w") as f:
        yaml.dump(paths_map, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved catalog to {path} (paths: {paths_file})")
    return True


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

    The regression data is expected at:
    test_data_dir/{diagnostic}/{test_case}/regression/
    """

    diagnostic: Diagnostic
    test_case_name: str
    test_data_dir: Path

    @property
    def paths(self) -> TestCasePaths:
        """Get paths for this test case."""
        return TestCasePaths.from_test_data_dir(self.test_data_dir, self.diagnostic.slug, self.test_case_name)

    def has_regression_data(self) -> bool:
        """Check if regression data exists for this test case."""
        regression_path = self.paths.regression
        return regression_path.exists() and (regression_path / "diagnostic.json").exists()

    def load_regression_definition(self, tmp_dir: Path) -> ExecutionDefinition:
        """
        Load regression data and create an ExecutionDefinition.

        Copies regression data to tmp_dir and replaces path placeholders.
        """
        regression_path = self.paths.regression
        catalog_path = self.paths.catalog

        if not catalog_path.exists():
            raise FileNotFoundError(
                f"No catalog file at {catalog_path} for test case datasets. Run `ref test-cases fetch` first."
            )
        if not regression_path.exists():
            raise FileNotFoundError(
                f"No regression data at {regression_path}. Run 'ref test-cases run --force-regen' first."
            )

        output_dir = tmp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(regression_path, output_dir, dirs_exist_ok=True)

        # Replace placeholders with actual paths
        for pattern in ("*.json", "*.txt", "*.yaml", "*.yml"):
            for file in output_dir.rglob(pattern):
                content = file.read_text()
                content = content.replace("<OUTPUT_DIR>", str(output_dir))
                content = content.replace("<TEST_DATA_DIR>", str(self.test_data_dir))
                file.write_text(content)

        # Load datasets from catalog
        datasets: ExecutionDatasetCollection = load_datasets_from_yaml(catalog_path)

        return ExecutionDefinition(
            diagnostic=self.diagnostic,
            key=f"test-{self.test_case_name}",
            datasets=datasets,
            output_directory=output_dir,
            root_directory=tmp_dir,
        )

    def validate(self, definition: ExecutionDefinition) -> None:
        """Validate CMEC bundles in the regression output."""
        result = self.diagnostic.build_execution_result(definition)
        result.to_output_path("out.log").touch()  # Log file not tracked in regression
        validate_cmec_bundles(self.diagnostic, result)


def collect_test_case_params(provider: DiagnosticProvider) -> list[ParameterSet]:
    """
    Collect all diagnostic/test_case pairs from a provider for parameterized testing.

    Returns a list of pytest.param objects with (diagnostic, test_case_name) tuples,
    each with an id of "{diagnostic.slug}/{test_case.name}".

    Parameters
    ----------
    provider
        The diagnostic provider to collect test cases from

    Returns
    -------
    :
        List of pytest.param objects for use with @pytest.mark.parametrize

    Example
    -------
    ```python
    from climate_ref_core.testing import collect_test_case_params
    from my_provider import provider

    test_case_params = collect_test_case_params(provider)


    @pytest.mark.parametrize("diagnostic,test_case_name", test_case_params)
    def test_my_test(diagnostic, test_case_name): ...
    ```
    """
    import pytest  # noqa: PLC0415

    params: list[ParameterSet] = []
    for diagnostic in provider.diagnostics():
        if diagnostic.test_data_spec is None:
            continue
        for test_case in diagnostic.test_data_spec.test_cases:
            params.append(
                pytest.param(
                    diagnostic,
                    test_case.name,
                    id=f"{diagnostic.slug}/{test_case.name}",
                )
            )
    return params
