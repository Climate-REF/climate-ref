from pathlib import Path

import pytest
from climate_ref_ilamb import provider

from climate_ref.testing import TestCaseRunner, validate_result
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.testing import (
    RegressionValidator,
    TestCasePaths,
    collect_test_case_params,
    load_datasets_from_yaml,
)


@pytest.fixture(scope="session")
def provider_test_data_dir() -> Path:
    """Path to the package-local test data directory."""
    return Path(__file__).parent.parent / "test-data"


xfail_diagnostics = [
    "ohc-noaa",  # Missing sample data
]
skipped_diagnostics = []


diagnostics = [
    pytest.param(
        diagnostic,
        id=diagnostic.slug,
        marks=[
            *([pytest.mark.xfail(reason="Expected failure")] if diagnostic.slug in xfail_diagnostics else []),
            *([pytest.mark.skip(reason="Problem test")] if diagnostic.slug in skipped_diagnostics else []),
        ],
    )
    for diagnostic in provider.diagnostics()
]

# Test case params for parameterized test_case tests
test_case_params = collect_test_case_params(provider)


@pytest.mark.slow
@pytest.mark.parametrize("diagnostic", diagnostics)
def test_diagnostics(diagnostic: Diagnostic, diagnostic_validation):
    validator = diagnostic_validation(diagnostic)

    definition = validator.get_definition()
    validator.execute(definition)


@pytest.mark.parametrize("diagnostic", diagnostics)
def test_build_results(diagnostic: Diagnostic, diagnostic_validation):
    validator = diagnostic_validation(diagnostic)

    definition = validator.get_regression_definition()
    validator.validate(definition)
    validator.execution_regression.check(definition.key, definition.output_directory)


@pytest.mark.parametrize("diagnostic,test_case_name", test_case_params)
def test_validate_test_case_regression(
    diagnostic: Diagnostic,
    test_case_name: str,
    provider_test_data_dir: Path,
    config,
    tmp_path: Path,
):
    """
    Validate pre-stored test case regression outputs as CMEC bundles.

    Each diagnostic/test_case is a separate parameterized test.
    """
    diagnostic.provider.configure(config)

    paths = TestCasePaths.from_test_data_dir(
        provider_test_data_dir,
        diagnostic.slug,
        test_case_name,
    )

    if not paths.catalog.exists():
        pytest.skip(f"No catalog file for {diagnostic.slug}/{test_case_name}")
    if not paths.regression.exists():
        pytest.skip(f"No regression data for {diagnostic.slug}/{test_case_name}")

    validator = RegressionValidator(
        diagnostic=diagnostic,
        test_case_name=test_case_name,
        test_data_dir=provider_test_data_dir,
    )

    definition = validator.load_regression_definition(tmp_path / diagnostic.slug / test_case_name)
    validator.validate(definition)


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
    Run diagnostic test cases end-to-end with ESGF data.

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

    datasets = load_datasets_from_yaml(paths.catalog)

    runner = TestCaseRunner(config=config, datasets=datasets)
    output_dir = tmp_path / diagnostic.slug / test_case_name

    result = runner.run(diagnostic, test_case_name, output_dir)

    assert result.successful, f"Diagnostic {diagnostic.slug} failed"
    validate_result(diagnostic, config, result)
