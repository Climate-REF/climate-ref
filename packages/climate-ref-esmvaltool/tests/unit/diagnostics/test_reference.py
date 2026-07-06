"""Tests for the declarative reference-dataset spec."""

from climate_ref_esmvaltool.diagnostics.reference import ESMValToolReferenceSpec
from climate_ref_esmvaltool.diagnostics.sea_ice_area_basic import SeaIceAreaBasic

from climate_ref_core.datasets import SourceDatasetType


def test_to_recipe_dataset_full_alphabetical_order():
    spec = ESMValToolReferenceSpec(
        project="OBS",
        dataset="OSI-450-nh",
        mip="OImon",
        tier=2,
        obs_type="reanaly",
        version="v3",
        supplementary_variables=({"short_name": "areacello", "mip": "fx"},),
    )
    entry = spec.to_recipe_dataset()
    assert entry == {
        "dataset": "OSI-450-nh",
        "mip": "OImon",
        "project": "OBS",
        "supplementary_variables": [{"short_name": "areacello", "mip": "fx"}],
        "tier": 2,
        "type": "reanaly",
        "version": "v3",
    }
    # Key order is significant: recipes are dumped with sort_keys=False.
    assert list(entry.keys()) == [
        "dataset",
        "mip",
        "project",
        "supplementary_variables",
        "tier",
        "type",
        "version",
    ]


def test_to_recipe_dataset_omits_unset_optionals():
    spec = ESMValToolReferenceSpec(project="obs4MIPs", dataset="GPCP-V2.3")
    assert spec.to_recipe_dataset() == {"dataset": "GPCP-V2.3", "project": "obs4MIPs"}


def test_to_recipe_dataset_builds_independent_mutable_entries():
    spec = ESMValToolReferenceSpec(
        project="native6", dataset="ERA5", tier=3, obs_type="reanaly", version="v1"
    )
    first = spec.to_recipe_dataset()
    second = spec.to_recipe_dataset()
    # Distinct objects so callers can mutate one without affecting the other.
    assert first == second
    assert first is not second
    first["dataset"] = "OTHER"
    assert second["dataset"] == "ERA5"


def test_diagnostic_maps_specs_to_reference_selectors():
    selectors = SeaIceAreaBasic().reference_dataset_selectors()
    assert [s.source_type for s in selectors] == [SourceDatasetType.ESMValToolReference] * 2
    assert [dict(s.facets) for s in selectors] == [
        {"project": "OBS", "source_id": "OSI-450-nh", "table_id": "OImon"},
        {"project": "OBS", "source_id": "OSI-450-sh", "table_id": "OImon"},
    ]


def test_reference_selectors_omit_table_id_when_mip_unset():
    # A spec without a mip (e.g. an obs4MIPs reference) matches by project + dataset only.
    class _Diag(SeaIceAreaBasic):
        reference_datasets = (ESMValToolReferenceSpec(project="obs4MIPs", dataset="GPCP-V2.3"),)

    (selector,) = _Diag().reference_dataset_selectors()
    assert dict(selector.facets) == {"project": "obs4MIPs", "source_id": "GPCP-V2.3"}
