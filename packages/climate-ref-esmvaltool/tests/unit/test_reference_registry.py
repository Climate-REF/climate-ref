import pytest
from climate_ref_esmvaltool.reference_registry import parse_registry_key

from climate_ref_core.esgf import RegistryRequest


class TestParseRegistryKey:
    def test_parse_obs_key(self):
        key = "ESMValTool/OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_197901-197912.nc"
        result = parse_registry_key(key)

        assert result["project"] == "OBS"
        assert result["source_id"] == "OSI-450-nh"
        assert result["variable_id"] == "sic"
        assert result["frequency"] == "mon"
        assert result["version"] == "v3"
        assert result["tier"] == 2
        assert result["key"] == key

    def test_parse_native6_key(self):
        key = "ESMValTool/native6/Tier3/ERA5/v1/mon/tas/era5_tas_1980_monthly.nc"
        result = parse_registry_key(key)

        assert result["project"] == "native6"
        assert result["source_id"] == "ERA5"
        assert result["variable_id"] == "tas"
        assert result["frequency"] == "mon"

    def test_frequency_matches_what_the_catalog_records(self):
        """A request names the frequency ingest derives, not the MIP table in the filename."""
        key = "ESMValTool/OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_197901-197912.nc"

        assert parse_registry_key(key)["frequency"] == "mon"

    @pytest.mark.parametrize(
        "key",
        [
            # Not a reference file at all.
            "ESMValTool/recipes/recipe_sea_ice.yml",
            # Under a project, but not a shape that project's layout allows.
            "ESMValTool/OBS/odd.nc",
        ],
    )
    def test_parse_key_that_is_not_reference_data(self, key):
        assert parse_registry_key(key) == {}


def test_registry_request_accepts_the_parser():
    """The provider owns its registry, so it hands the reader to the request."""
    request = RegistryRequest(
        slug="osi-450",
        registry_name="esmvaltool-datasets",
        source_type="ESMValToolReference",
        facets={"source_id": "OSI-450-nh"},
        key_parser=parse_registry_key,
    )

    assert request._get_parser() is parse_registry_key
