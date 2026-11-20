from climate_ref_esmvaltool import provider

from climate_ref_core.esgf import CMIP7Request
from climate_ref_core.testing import collect_test_case_params


def test_tcre_cmip7_case_is_expected_to_fail_until_fco2antt_is_published():
    """CanESM5-1 publishes esm-flat10 tas, but no CMIP7 model publishes its fco2antt yet."""
    diagnostic = next(d for d in provider.diagnostics() if d.slug == "transient-climate-response-emissions")
    case = diagnostic.test_data_spec.get_case("cmip7")

    (request,) = case.requests
    assert isinstance(request, CMIP7Request)
    assert request.facets["experiment_id"] == ["esm-flat10", "esm-piControl"]
    assert "fco2antt" in request.facets["variable_id"]
    assert case.xfail

    (param,) = [
        param
        for param in collect_test_case_params(provider)
        if param.values[0] is diagnostic and param.values[1] == "cmip7"
    ]
    assert any(mark.name == "xfail" and mark.kwargs["strict"] for mark in param.marks)


def test_zec_cmip7_case_uses_published_flat10_data():
    """esm-flat10-zec and its esm-flat10 parent are published for CanESM5-1."""
    diagnostic = next(d for d in provider.diagnostics() if d.slug == "zero-emission-commitment")
    case = diagnostic.test_data_spec.get_case("cmip7")

    (request,) = case.requests
    assert isinstance(request, CMIP7Request)
    assert request.facets["experiment_id"] == ["esm-flat10", "esm-flat10-zec"]
