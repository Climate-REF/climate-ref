"""
Testing utilities for running and validating diagnostic test cases.

This module provides:
- Path resolution for package-local test data (catalogs, regression data)
- Sample data fetching utilities
- TestCaseRunner for executing diagnostics with test data
- Result validation helpers
"""

import shutil
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
from climate_ref_core.testing import (
    validate_cmec_bundles,
)


def _determine_test_directory() -> Path | None:
    path = env.path("REF_TEST_DATA_DIR", default=Path(__file__).parents[4] / "tests" / "test-data")

    if not path.exists():  # pragma: no cover
        return None
    return path


TEST_DATA_DIR = _determine_test_directory()
"""Path to the centralised test data directory (for sample data)."""
SAMPLE_DATA_VERSION = "v0.7.4"


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


def validate_result(
    diagnostic: Diagnostic, config: Config, result: ExecutionResult
) -> None:  # pragma: no cover
    """
    Asserts the correctness of the result of a diagnostic execution

    This should only be used by the test suite as it will create a fake
    database entry for the diagnostic execution result.
    """
    # TODO: Remove this function once we have moved to using RegressionValidator
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

    # Validate CMEC bundles
    validate_cmec_bundles(diagnostic, result)

    # Create a fake log file if one doesn't exist
    if not result.to_output_path("out.log").exists():
        result.to_output_path("out.log").touch()

    # Process and store the result
    # TODO: This is missing from RegressionValidator
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
