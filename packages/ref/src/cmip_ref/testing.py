"""
Testing utilities
"""

import shutil
from pathlib import Path

from loguru import logger

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.executor import handle_execution_result
from cmip_ref.models import MetricExecutionResult as MetricExecutionResultModel
from cmip_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from cmip_ref_core.metrics import MetricExecutionResult
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


def _determine_test_directory() -> Path | None:
    expected = Path(__file__).parents[4] / "tests" / "test-data"

    if not expected.exists():  # pragma: no cover
        return None
    return expected


TEST_DATA_DIR = _determine_test_directory()
SAMPLE_DATA_VERSION = "v0.4.3"


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


def validate_result(config: Config, result: MetricExecutionResult) -> None:
    """
    Asserts the correctness of the result of a metric execution

    This should only be used by the test suite as it will create a fake
    database entry for the metric execution result.
    """
    # Add a fake item in the Database
    database = Database.from_config(config)
    metric_execution_result = MetricExecutionResultModel(
        metric_execution_group_id=1,
        dataset_hash=result.definition.metric_dataset.hash,
        output_fragment=str(result.definition.output_fragment()),
    )
    database.session.add(metric_execution_result)
    database.session.flush()

    assert result.successful

    # Validate bundles
    CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))
    CMECOutput.load_from_json(result.to_output_path(result.output_bundle_filename))

    # Create a fake log file if one doesn't exist
    if not result.to_output_path("out.log").exists():
        result.to_output_path("out.log").touch()

    # This checks if the bundles are valid
    handle_execution_result(
        config, database=database, metric_execution_result=metric_execution_result, result=result
    )
