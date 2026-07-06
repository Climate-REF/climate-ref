"""
Testing utilities for running and validating diagnostic test cases.

This module provides:
- Path resolution for package-local test data (catalogs, regression data)
- Sample data fetching utilities
- TestCaseRunner for executing diagnostics with test data
- Drift-checking helpers for test-case regression baselines
  (``assert_test_case_no_drift`` and the ``create_no_drift_test`` factory
  used by every provider's integration test module)
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from attrs import define
from loguru import logger

from climate_ref import SAMPLE_DATA_VERSION
from climate_ref.config import Config
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.diagnostics import Diagnostic, ExecutionDefinition, ExecutionResult
from climate_ref_core.env import env
from climate_ref_core.exceptions import DatasetResolutionError, NoTestDataSpecError, TestCaseNotFoundError
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.regression import Manifest
from climate_ref_core.testing import TestCasePaths, collect_test_case_params, load_datasets_from_yaml


def _determine_test_directory() -> Path | None:
    path = env.path("REF_TEST_DATA_DIR", default=Path(__file__).parents[4] / "tests" / "test-data")

    if not path.exists():  # pragma: no cover
        return None
    return path


TEST_DATA_DIR = _determine_test_directory()
"""Path to the centralised test data directory (for sample data)."""
# SAMPLE_DATA_VERSION is imported from climate_ref to avoid circular imports


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


def assert_test_case_no_drift(
    config: Config,
    diagnostic: Diagnostic,
    test_case_name: str,
    paths: TestCasePaths,
    work_dir: Path,
) -> None:
    """
    Execute a diagnostic test case and assert its committed bundle has not drifted.

    Runs the diagnostic against the fetched catalog using the same execute/build stages as
    ``ref test-cases run`` (see :mod:`climate_ref.cli.test_cases._stages`),
    rebuilds the committed bundle from the fresh execution,
    and compares it to the tracked ``regression/`` baseline within tolerance.

    Unlike ``ref test-cases replay`` this re-executes the diagnostic rather than replaying,
    so it also proves the diagnostic still runs and emits a valid bundle.

    Ingesting a committed bundle into the database is a separate concern,
    covered by the executor result-handling tests
    (``packages/climate-ref/tests/unit/executor/test_result_handling.py`` and friends).
    It is intentionally not re-checked here.

    Parameters
    ----------
    config
        The active configuration (provides the software root and results dir).
    diagnostic
        The diagnostic to run.
    test_case_name
        Name of the test case to run.
    paths
        Resolved paths for the test case (catalog + tracked regression baseline).
    work_dir
        A per-test scratch directory for the output slot and execution outputs.

    Raises
    ------
    AssertionError
        If the rebuilt committed bundle drifts from the tracked baseline.
    """
    from climate_ref.cli.test_cases._stages import (  # noqa: PLC0415
        baseline_placeholders,
        stage_build,
        stage_compare,
        stage_execute,
    )

    if diagnostic.test_data_spec is None:
        raise NoTestDataSpecError(f"Diagnostic {diagnostic.slug} has no test_data_spec")

    tc = diagnostic.test_data_spec.get_case(test_case_name)
    datasets = load_datasets_from_yaml(paths.catalog)

    slot = work_dir / "slot"
    slot.mkdir(parents=True, exist_ok=True)
    placeholders = baseline_placeholders(paths, config)

    # stage_execute runs the diagnostic (raising if it is not successful)
    # and copies the curated native set into the slot
    # stage_build rebuilds the committed bundle from it.
    source = stage_execute(
        config=config,
        diag=diagnostic,
        tc=tc,
        datasets=datasets,
        slot=slot,
        execution_dir=work_dir / "exec",
        clean=True,
    )
    stage_build(slot=slot, source=source, placeholders=placeholders)

    expected = Manifest.load(paths.manifest).committed
    failures, _ = stage_compare(slot=slot, paths=paths, slug=diagnostic.slug, expected=expected)
    assert not failures, (
        f"{diagnostic.provider.slug}/{diagnostic.slug}/{test_case_name}: committed bundle drift:\n"
        + "\n".join(failures)
    )


def create_no_drift_test(provider: DiagnosticProvider) -> Callable[..., None]:
    """
    Build the standard per-provider integration test for committed-bundle drift.

    Returns a pytest function parameterized with one case per diagnostic test case
    (via :func:`climate_ref_core.testing.collect_test_case_params`),
    marked ``slow`` and ``test_cases``.
    Each case configures the provider,
    resolves the package-local test-case paths,
    skips when the fetched catalog or committed baseline is missing,
    and delegates to :func:`assert_test_case_no_drift`.

    Requires ``ref test-cases fetch --provider <slug>`` to have been run first.

    Usage in a provider's ``tests/integration/test_diagnostics.py``::

        from climate_ref_example import provider

        from climate_ref.testing import create_no_drift_test

        test_run_test_cases = create_no_drift_test(provider)

    Parameters
    ----------
    provider
        The diagnostic provider whose test cases should be exercised.
    """
    import pytest  # noqa: PLC0415

    @pytest.mark.slow
    @pytest.mark.test_cases
    @pytest.mark.parametrize("diagnostic,test_case_name", collect_test_case_params(provider))
    def test_run_test_cases(
        diagnostic: Diagnostic,
        test_case_name: str,
        config: Config,
        tmp_path: Path,
    ) -> None:
        """Execute the test case end-to-end and assert the committed bundle has not drifted."""
        diagnostic.provider.configure(config)

        paths = TestCasePaths.from_diagnostic(diagnostic, test_case_name)
        if paths is None:
            pytest.skip(f"No test-data directory for {diagnostic.slug} (not a development checkout)")
        if not paths.catalog.exists():
            pytest.skip(
                f"No catalog file for {diagnostic.slug}/{test_case_name}. "
                f"Run `ref test-cases fetch --provider {provider.slug}` first."
            )
        if not paths.manifest.exists() or not paths.regression.exists():
            pytest.skip(f"No committed baseline for {diagnostic.slug}/{test_case_name}")

        assert_test_case_no_drift(config, diagnostic, test_case_name, paths, tmp_path)

    return test_run_test_cases


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
        clean: bool = False,
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
        clean
            If True, delete the output directory before running

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

        # Validate that all non-empty collections have the required 'path' column
        for src_type, collection in self.datasets.items():
            if len(collection.datasets) > 0 and "path" not in collection.datasets.columns:
                raise DatasetResolutionError(
                    f"Datasets for '{src_type}' are missing the required 'path' column. "
                    f"Run 'ref test-cases fetch' to generate the paths file."
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

        if clean and output_dir.exists():
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        definition = ExecutionDefinition(
            diagnostic=diagnostic,
            key=f"test-{test_case_name}",
            datasets=self.datasets,
            output_directory=output_dir,
            root_directory=output_dir.parent,
        )

        # Run the diagnostic with the regression-capture hook enabled, so the native output is
        # normalised (prepare_regression_output) before the bundle is built and captured.
        return diagnostic.run(definition, capture_regression=True)
