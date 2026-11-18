import pytest
from climate_ref_esmvaltool import provider

from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.testing import collect_test_case_params


@pytest.mark.parametrize(
    "slug,expected_cases",
    [
        # No CMIP7 model publishes fco2antt for esm-flat10 yet
        ("transient-climate-response-emissions", ["cmip6"]),
    ],
)
def test_flat10_cases_disabled_until_real_inputs_available(slug, expected_cases):
    """Exclude unsupported fetches and integration cases without removing CMIP7 support."""
    diagnostic = next(d for d in provider.diagnostics() if d.slug == slug)
    assert diagnostic.test_data_spec is not None
    assert [case.name for case in diagnostic.test_data_spec.test_cases] == expected_cases
    for case in diagnostic.test_data_spec.test_cases:
        assert case.requests
        assert all(isinstance(request, CMIP6Request) for request in case.requests)

    collected = [
        param.values[1] for param in collect_test_case_params(provider) if param.values[0] is diagnostic
    ]
    assert collected == expected_cases
    assert any(
        requirement.source_type == SourceDatasetType.CMIP7
        for alternative in diagnostic.data_requirements
        for requirement in alternative
    )


def test_zec_cmip7_case_uses_published_flat10_data():
    """esm-flat10-zec and its esm-flat10 parent are published for CanESM5-1."""
    diagnostic = next(d for d in provider.diagnostics() if d.slug == "zero-emission-commitment")
    case = diagnostic.test_data_spec.get_case("cmip7")

    (request,) = case.requests
    assert isinstance(request, CMIP7Request)
    assert request.facets["experiment_id"] == ["esm-flat10", "esm-flat10-zec"]
