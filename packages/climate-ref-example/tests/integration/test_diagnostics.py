from pathlib import Path

import pytest
from climate_ref_example import provider

from climate_ref.testing import TestCaseRunner, validate_result
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.testing import (
    RegressionValidator,
    get_test_case_regression_path,
    load_datasets_from_yaml,
)

# Standard parametrized tests using sample data
diagnostics = [pytest.param(diagnostic, id=diagnostic.slug) for diagnostic in provider.diagnostics()]


@pytest.mark.slow
@pytest.mark.parametrize("diagnostic", diagnostics)
def test_diagnostics(diagnostic: Diagnostic, diagnostic_validation):
    """Run diagnostic end-to-end with sample data."""
    validator = diagnostic_validation(diagnostic)
    definition = validator.get_definition()
    validator.execute(definition)


@pytest.mark.parametrize("diagnostic", diagnostics)
def test_build_results(diagnostic: Diagnostic, diagnostic_validation):
    """Validate regression outputs can be built from sample data."""
    validator = diagnostic_validation(diagnostic)
    definition = validator.get_regression_definition()
    validator.validate(definition)
    validator.execution_regression.check(definition.key, definition.output_directory)


def test_validate_test_case_regression(
    subtests: pytest.Subtests,
    regression_dir: Path,
    test_data_dir: Path,
    config,
    tmp_path: Path,
):
    """
    Validate pre-stored test case regression outputs as CMEC bundles.

    Uses pytest 9 subtests to iterate over all diagnostics with test_data_spec.
    Each diagnostic/test_case is reported as a separate subtest.
    """
    for diagnostic in provider.diagnostics():
        if diagnostic.test_data_spec is None:
            continue

        diagnostic.provider.configure(config)

        for test_case in diagnostic.test_data_spec.test_cases:
            with subtests.test(msg=f"{diagnostic.slug}/{test_case.name}"):
                regression_path = get_test_case_regression_path(
                    regression_dir,
                    provider.slug,
                    diagnostic.slug,
                    test_case.name,
                )

                if not regression_path.exists():
                    pytest.skip(f"No regression data for {diagnostic.slug}/{test_case.name}")

                validator = RegressionValidator(
                    diagnostic=diagnostic,
                    test_case_name=test_case.name,
                    regression_data_dir=regression_dir,
                    test_data_dir=test_data_dir,
                )

                definition = validator.load_regression_definition(tmp_path / diagnostic.slug / test_case.name)
                validator.validate(definition)


@pytest.mark.slow
@pytest.mark.test_cases
def test_run_test_cases(
    subtests: pytest.Subtests,
    catalog_dir: Path,
    config,
    tmp_path: Path,
):
    """
    Run diagnostic test cases end-to-end with ESGF data.

    Uses pytest 9 subtests to iterate over all diagnostics with test_data_spec.
    Each diagnostic/test_case is reported as a separate subtest.

    Requires: `ref test-cases fetch --provider example` to have been run first.
    """
    for diagnostic in provider.diagnostics():
        if diagnostic.test_data_spec is None:
            continue

        diagnostic.provider.configure(config)

        for test_case in diagnostic.test_data_spec.test_cases:
            with subtests.test(msg=f"{diagnostic.slug}/{test_case.name}"):
                catalog_path = catalog_dir / diagnostic.slug / f"{test_case.name}.yaml"

                if not catalog_path.exists():
                    pytest.skip(f"No catalog file for {diagnostic.slug}/{test_case.name}")

                datasets = load_datasets_from_yaml(catalog_path)

                runner = TestCaseRunner(config=config, datasets=datasets)
                output_dir = tmp_path / diagnostic.slug / test_case.name

                result = runner.run(diagnostic, test_case.name, output_dir)

                assert result.successful, f"Diagnostic {diagnostic.slug} failed"
                validate_result(diagnostic, config, result)
