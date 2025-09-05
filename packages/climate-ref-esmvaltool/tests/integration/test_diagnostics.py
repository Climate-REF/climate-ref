import pytest
from climate_ref_esmvaltool import provider

from climate_ref_core.diagnostics import Diagnostic

SKIP = {
    "regional-historical-annual-cycle",
    "regional-historical-timeseries",
}

diagnostics = [
    pytest.param(
        diagnostic,
        id=diagnostic.slug,
        marks=pytest.mark.skipif(
            diagnostic.slug in SKIP,
            reason="Output data too large to store in git",
        ),
    )
    for diagnostic in provider.diagnostics()
]


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
