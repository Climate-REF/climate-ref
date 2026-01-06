"""
Testing utilities
"""

import shutil
from pathlib import Path

import pandas as pd
from attrs import define
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.executor import handle_execution_result
from climate_ref.models import Execution, ExecutionGroup
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from climate_ref_core.datasets import SourceDatasetType
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
    - Resolving test case datasets (from explicit datasets, YAML file, or solving)
    - Setting up the execution definition
    - Running the diagnostic
    """

    config: Config
    data_catalog: dict[SourceDatasetType, pd.DataFrame] | None = None

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

        test_case = diagnostic.test_data_spec.get_case(test_case_name)

        # Try to resolve datasets
        try:
            datasets = test_case.resolve_datasets(
                data_catalog=self.data_catalog,
                diagnostic=diagnostic,
                package_dir=None,
            )
        except ValueError as e:
            raise DatasetResolutionError(f"Could not resolve datasets: {e}") from e

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
            datasets=datasets,
            output_directory=output_dir,
            root_directory=output_dir.parent,
        )

        # Run the diagnostic
        return diagnostic.run(definition)
