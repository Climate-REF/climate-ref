"""
Testing utilities
"""

import shutil
from pathlib import Path

from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Execution, ExecutionGroup
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from climate_ref_core.diagnostics import Diagnostic, ExecutionResult
from climate_ref_core.env import env
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


def _determine_test_directory() -> Path | None:
    path = env.path("REF_TEST_DATA_DIR", default=Path(__file__).parents[4] / "tests" / "test-data")

    if not path.exists():  # pragma: no cover
        return None
    return path


TEST_DATA_DIR = _determine_test_directory()
SAMPLE_DATA_VERSION = "v0.7.1"


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

    # Import late to avoid importing executors,
    # some of which have on-import side effects, at package load time
    from climate_ref.executor import handle_execution_result

    # This checks if the bundles are valid
    handle_execution_result(config, database=database, execution=execution, result=result)
