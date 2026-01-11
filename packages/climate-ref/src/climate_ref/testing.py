"""
Testing utilities for running and validating diagnostic test cases.

This module provides:
- Path resolution for package-local test data (catalogs, regression data)
- Sample data fetching utilities
- TestCaseRunner for executing diagnostics with test data
- Result validation helpers
"""

import shutil
import sys
from pathlib import Path

from attrs import define
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.executor import handle_execution_result
from climate_ref.models import Execution, ExecutionGroup
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.diagnostics import Diagnostic, ExecutionDefinition, ExecutionResult
from climate_ref_core.env import env
from climate_ref_core.exceptions import DatasetResolutionError, NoTestDataSpecError, TestCaseNotFoundError
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


def _determine_test_directory() -> Path | None:
    path = env.path("REF_TEST_DATA_DIR", default=Path(__file__).parents[4] / "tests" / "test-data")

    if not path.exists():  # pragma: no cover
        return None
    return path


TEST_DATA_DIR = _determine_test_directory()
"""Path to the centralised test data directory (for sample data)."""
SAMPLE_DATA_VERSION = "v0.7.4"


def _get_provider_test_data_dir(diag: Diagnostic, create: bool = False) -> Path | None:
    """
    Get the test-data directory for a provider's package.

    Returns packages/climate-ref-{provider}/tests/test-data/ or None if unavailable.
    This will only work if working in a development checkout of the package.

    Parameters
    ----------
    diag
        The diagnostic to get the test data dir for
    create
        If True, create the test-data directory if it doesn't exist
    """
    provider_module_name = diag.provider.__class__.__module__.split(".")[0]
    logger.debug(f"Looking up test data dir for provider module: {provider_module_name}")

    if provider_module_name not in sys.modules:
        logger.debug(f"Module {provider_module_name} not in sys.modules")
        return None

    provider_module = sys.modules[provider_module_name]
    if not hasattr(provider_module, "__file__") or provider_module.__file__ is None:
        logger.debug(f"Module {provider_module_name} has no __file__ attribute")
        return None

    # Module: packages/climate-ref-{slug}/src/climate_ref_{slug}/__init__.py
    # Target: packages/climate-ref-{slug}/tests/test-data/
    module_path = Path(provider_module.__file__)
    package_root = module_path.parent.parent.parent  # src -> climate-ref-{slug}
    tests_dir = package_root / "tests"

    # Only return path if tests/ exists (dev checkout)
    if not tests_dir.exists():
        logger.debug(f"Tests dir does not exist (not a dev checkout): {tests_dir}")
        return None

    test_data_dir = tests_dir / "test-data"
    logger.debug(f"Provider module path: {module_path}")
    logger.debug(f"Derived test data dir: {test_data_dir} (exists: {test_data_dir.exists()})")

    if create and not test_data_dir.exists():
        test_data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created test data dir: {test_data_dir}")

    return test_data_dir


def get_provider_catalog_path(diag: Diagnostic, test_case: str, create: bool = False) -> Path | None:
    """
    Get path to catalog file for a test case in the provider's package.

    Path: packages/climate-ref-{provider}/tests/test-data/catalogs/{diagnostic}/{test_case}.yaml

    Parameters
    ----------
    diag
        The diagnostic to get the catalog path for
    test_case
        Test case name (e.g., 'default')
    create
        If True, create the catalogs directory if it doesn't exist
    """
    test_data_dir = _get_provider_test_data_dir(diag, create=create)
    if test_data_dir is None:
        logger.debug("Could not determine provider test data dir")
        return None

    catalog_dir = test_data_dir / "catalogs"

    if create and not catalog_dir.exists():
        catalog_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created catalog dir: {catalog_dir}")
    elif not catalog_dir.exists():
        logger.debug(f"Catalog dir does not exist: {catalog_dir}")
        return None

    catalog_path = catalog_dir / diag.slug / f"{test_case}.yaml"
    logger.debug(f"Resolved catalog path: {catalog_path}")
    return catalog_path


def get_provider_regression_path(diag: Diagnostic, test_case: str, create: bool = False) -> Path | None:
    """
    Get path to regression data for a test case in the provider's package.

    Path: packages/climate-ref-{provider}/tests/test-data/regression/{provider}/{diagnostic}/{test_case}/

    Parameters
    ----------
    diag
        The diagnostic to get the regression path for
    test_case
        Test case name (e.g., 'default')
    create
        If True, create the regression directory structure if it doesn't exist
    """
    test_data_dir = _get_provider_test_data_dir(diag, create=create)
    if test_data_dir is None:
        return None

    regression_path = test_data_dir / "regression" / diag.provider.slug / diag.slug / test_case
    logger.debug(f"Resolved regression path: {regression_path}")
    return regression_path


def fetch_sample_data(force_cleanup: bool = False, symlink: bool = False) -> None:
    """
    Fetch the sample data for the given version.

    The sample data is produced in the [Climate-REF/ref-sample-data](https://github.com/Climate-REF/ref-sample-data)
    repository.
    This repository contains decimated versions of key datasets used by the diagnostics packages.
    Decimating these data greatly reduces the data volumes needed to run the test-suite.

    Parameters
    ----------
    force_cleanup
        If True, remove any existing files
    symlink
        If True, symlink in the data otherwise copy the files

        The symlink approach is faster, but will fail when running with a non-local executor
        because the symlinks can't be followed.
    """

    if TEST_DATA_DIR is None:  # pragma: no cover
        logger.warning("Test data directory not found, skipping sample data fetch")
        return

    sample_data_registry = dataset_registry_manager["sample-data"]

    output_dir = TEST_DATA_DIR / "sample-data"
    version_file = output_dir / "version.txt"
    existing_version = None

    if output_dir.exists():  # pragma: no branch
        if version_file.exists():  # pragma: no branch
            with open(version_file) as fh:
                existing_version = fh.read().strip()

        if force_cleanup or existing_version != SAMPLE_DATA_VERSION:  # pragma: no branch
            logger.warning("Removing existing sample data")
            shutil.rmtree(output_dir)

    fetch_all_files(sample_data_registry, "sample", output_dir, symlink)

    # Write out the current sample data version to the copying as complete
    with open(output_dir / "version.txt", "w") as fh:
        fh.write(SAMPLE_DATA_VERSION)


def validate_result(diagnostic: Diagnostic, config: Config, result: ExecutionResult) -> None:
    """
    Asserts the correctness of the result of a diagnostic execution

    This should only be used by the test suite as it will create a fake
    database entry for the diagnostic execution result.
    """
    # Add a fake execution/execution group in the Database
    database = Database.from_config(config)
    execution_group = ExecutionGroup(
        diagnostic_id=1, key=result.definition.key, dirty=True, selectors=result.definition.datasets.selectors
    )
    database.session.add(execution_group)
    database.session.flush()

    execution = Execution(
        execution_group_id=execution_group.id,
        dataset_hash=result.definition.datasets.hash,
        output_fragment=str(result.definition.output_fragment()),
    )
    database.session.add(execution)
    database.session.flush()

    assert result.successful

    # Validate bundles
    metric_bundle = CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))
    CMECMetric.model_validate(metric_bundle)
    bundle_dimensions = tuple(metric_bundle.DIMENSIONS.root["json_structure"])
    assert diagnostic.facets == bundle_dimensions
    CMECOutput.load_from_json(result.to_output_path(result.output_bundle_filename))

    # Create a fake log file if one doesn't exist
    if not result.to_output_path("out.log").exists():
        result.to_output_path("out.log").touch()

    # This checks if the bundles are valid
    handle_execution_result(config, database=database, execution=execution, result=result)


@define
class TestCaseRunner:
    """
    Helper class for running diagnostic test cases.

    This runner handles:
    - Running the diagnostic with pre-resolved datasets
    - Setting up the execution definition
    """

    config: Config
    datasets: ExecutionDatasetCollection | None = None

    def run(
        self,
        diagnostic: Diagnostic,
        test_case_name: str = "default",
        output_dir: Path | None = None,
    ) -> ExecutionResult:
        """
        Run a specific test case for a diagnostic.

        Parameters
        ----------
        diagnostic
            The diagnostic to run
        test_case_name
            Name of the test case to run (default: "default")
        output_dir
            Optional output directory for results

        Returns
        -------
        ExecutionResult
            The result of running the diagnostic

        Raises
        ------
        NoTestDataSpecError
            If the diagnostic has no test_data_spec
        TestCaseNotFoundError
            If the test case doesn't exist
        DatasetResolutionError
            If datasets cannot be resolved
        """
        if diagnostic.test_data_spec is None:
            raise NoTestDataSpecError(f"Diagnostic {diagnostic.slug} has no test_data_spec")

        if not diagnostic.test_data_spec.has_case(test_case_name):
            raise TestCaseNotFoundError(
                f"Test case {test_case_name!r} not found. Available: {diagnostic.test_data_spec.case_names}"
            )

        if self.datasets is None:
            raise DatasetResolutionError(
                "No datasets provided. Run 'ref test-cases fetch' first to build the catalog."
            )

        # Determine output directory
        if output_dir is None:
            output_dir = (
                self.config.paths.results
                / "test-cases"
                / diagnostic.provider.slug
                / diagnostic.slug
                / test_case_name
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create execution definition
        definition = ExecutionDefinition(
            diagnostic=diagnostic,
            key=f"test-{test_case_name}",
            datasets=self.datasets,
            output_directory=output_dir,
            root_directory=output_dir.parent,
        )

        # Run the diagnostic
        return diagnostic.run(definition)
