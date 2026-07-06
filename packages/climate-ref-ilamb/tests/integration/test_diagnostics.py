from pathlib import Path

import pytest
from climate_ref_ilamb import provider

from climate_ref.testing import assert_test_case_no_drift
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.testing import (
    TestCasePaths,
    collect_test_case_params,
)


@pytest.fixture(scope="session")
def provider_test_data_dir() -> Path:
    """Path to the package-local test data directory."""
    return Path(__file__).parent.parent / "test-data"


# Test case params for parameterized test_case tests
test_case_params = collect_test_case_params(provider)


@pytest.mark.slow
@pytest.mark.test_cases
@pytest.mark.parametrize("diagnostic,test_case_name", test_case_params)
def test_run_test_cases(
    diagnostic: Diagnostic,
    test_case_name: str,
    provider_test_data_dir: Path,
    config,
    tmp_path: Path,
):
    """
    Execute each diagnostic test case end-to-end and assert the committed bundle has not drifted.

    Runs the diagnostic against the fetched data using the same execute/build stages as
    ``ref test-cases run``, then compares the freshly rebuilt committed bundle to the tracked
    ``regression/`` baseline within tolerance. Unlike ``ref test-cases replay`` this re-executes
    the diagnostic rather than replaying stored native blobs, so it also proves the diagnostic
    still runs and emits a valid bundle.

    Ingesting the committed bundles into the database is covered separately by the executor
    result-handling tests (``packages/climate-ref/tests/unit/executor/test_result_handling.py``
    and friends) and is intentionally not re-checked here.

    Requires: `ref test-cases fetch --provider ilamb` to have been run first.
    """
    diagnostic.provider.configure(config)

    paths = TestCasePaths.from_test_data_dir(
        provider_test_data_dir,
        diagnostic.slug,
        test_case_name,
    )
    if not paths.catalog.exists():
        pytest.skip(f"No catalog file for {diagnostic.slug}/{test_case_name}")
    if not paths.manifest.exists() or not paths.regression.exists():
        pytest.skip(f"No committed baseline for {diagnostic.slug}/{test_case_name}")

    assert_test_case_no_drift(config, diagnostic, test_case_name, paths, tmp_path)
