import importlib.resources

from climate_ref_esmvaltool.diagnostics.sea_ice_area_basic import (
    _REFERENCE_REQUEST,
    _REFERENCE_REQUIREMENT,
    REFERENCE_FACETS,
    SeaIceAreaBasic,
)
from climate_ref_esmvaltool.reference_registry import parse_registry_key

from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.esgf.registry import _matches_facets


def registry_keys() -> list[str]:
    """Every key of the ESMValTool reference registry."""
    data = importlib.resources.files("climate_ref_esmvaltool.dataset_registry").joinpath("data.txt")
    return [line.split()[0] for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_the_request_selects_what_the_recipe_names():
    """The fetched files are the OSI-450 data the recipe compares each model against.

    Both hemispheres, each with the monthly ``sic`` the recipe reads
    and the ``fx`` ``areacello`` it declares as a supplementary variable.
    """
    matched = [
        facets
        for key in registry_keys()
        if (facets := parse_registry_key(key)) and _matches_facets(facets, _REFERENCE_REQUEST.facets)
    ]

    assert {facets["source_id"] for facets in matched} == {"OSI-450-nh", "OSI-450-sh"}
    assert {facets["variable_id"] for facets in matched} == {"sic", "areacello"}
    # 36 years of monthly sea ice concentration per hemisphere, plus one cell area file each.
    assert len(matched) == 74


def test_the_request_and_the_requirement_agree():
    """A test case cannot fetch one set of files and the solver then select another."""
    assert _REFERENCE_REQUEST.facets is REFERENCE_FACETS
    # A `FacetFilter` wraps a lone value in a tuple, so the filter holds the same facets, not the same dict.
    assert _REFERENCE_REQUIREMENT.filters[0].facets == {
        facet: (value,) if isinstance(value, str) else tuple(value)
        for facet, value in REFERENCE_FACETS.items()
    }
    assert _REFERENCE_REQUEST.source_type == SourceDatasetType.ESMValToolReference.value


def test_every_requirement_collection_asks_for_the_reference_data():
    """Neither the CMIP6 nor the CMIP7 branch may run without the observations."""
    for collection in SeaIceAreaBasic.data_requirements:
        assert _REFERENCE_REQUIREMENT in collection
