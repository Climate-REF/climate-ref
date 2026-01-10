"""Tests for CMIP6 to CMIP7 format conversion."""

import numpy as np
import pytest
import xarray as xr

from climate_ref_core.cmip6_to_cmip7 import (
    VARIABLE_BRANDING,
    BrandingSuffix,
    convert_cmip6_dataset,
    convert_cmip6_to_cmip7_attrs,
    create_cmip7_instance_id,
    create_cmip7_path,
    get_branding_suffix,
    get_cmip7_variable_name,
    get_frequency_from_table,
    get_realm_from_table,
)


class TestBrandingSuffix:
    def test_str_representation(self):
        suffix = BrandingSuffix("tavg", "h2m", "hxy", "u")
        assert str(suffix) == "tavg-h2m-hxy-u"

    def test_default_values(self):
        suffix = BrandingSuffix()
        assert str(suffix) == "tavg-u-hxy-u"


class TestGetBrandingSuffix:
    def test_known_variable_tas(self):
        suffix = get_branding_suffix("tas")
        assert suffix.temporal_label == "tavg"
        assert suffix.vertical_label == "h2m"
        assert suffix.horizontal_label == "hxy"
        assert suffix.area_label == "u"

    def test_known_variable_tos(self):
        suffix = get_branding_suffix("tos")
        assert suffix.vertical_label == "d0m"
        assert suffix.area_label == "sea"

    def test_known_variable_areacella(self):
        suffix = get_branding_suffix("areacella")
        assert suffix.temporal_label == "ti"

    def test_unknown_variable_returns_defaults(self):
        suffix = get_branding_suffix("unknown_var")
        assert suffix.temporal_label == "tavg"


class TestGetCmip7VariableName:
    def test_tas(self):
        name = get_cmip7_variable_name("tas")
        assert name == "tas_tavg-h2m-hxy-u"

    def test_pr(self):
        name = get_cmip7_variable_name("pr")
        assert name == "pr_tavg-u-hxy-u"

    def test_tos(self):
        name = get_cmip7_variable_name("tos")
        assert name == "tos_tavg-d0m-hxy-sea"

    def test_explicit_branding(self):
        branding = BrandingSuffix("tmax", "h2m", "hxy", "u")
        name = get_cmip7_variable_name("tas", branding)
        assert name == "tas_tmax-h2m-hxy-u"


class TestGetFrequencyFromTable:
    @pytest.mark.parametrize(
        ("table_id", "expected"),
        [
            ("Amon", "mon"),
            ("Omon", "mon"),
            ("Lmon", "mon"),
            ("day", "day"),
            ("Aday", "day"),
            ("Oday", "day"),
            ("3hr", "3hr"),
            ("6hrLev", "6hr"),
            ("fx", "fx"),
            ("Ofx", "fx"),
        ],
    )
    def test_frequency_extraction(self, table_id: str, expected: str):
        assert get_frequency_from_table(table_id) == expected


class TestGetRealmFromTable:
    @pytest.mark.parametrize(
        ("table_id", "expected"),
        [
            ("Amon", "atmos"),
            ("Omon", "ocean"),
            ("Lmon", "land"),
            ("LImon", "landIce"),
            ("SImon", "seaIce"),
            ("AERmon", "aerosol"),
            ("fx", "atmos"),
            ("Ofx", "ocean"),
        ],
    )
    def test_realm_mapping(self, table_id: str, expected: str):
        assert get_realm_from_table(table_id) == expected


class TestConvertCmip6ToCmip7Attrs:
    def test_adds_new_attributes(self):
        cmip6_attrs = {
            "variable_id": "tas",
            "table_id": "Amon",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
        }
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["mip_era"] == "CMIP7"
        assert cmip7_attrs["region"] == "GLB"
        assert cmip7_attrs["archive_id"] == "WCRP"
        assert cmip7_attrs["host_collection"] == "CMIP7"
        assert cmip7_attrs["drs_specs"] == "MIP-DRS7"
        assert cmip7_attrs["branding_suffix"] == "tavg-h2m-hxy-u"

    def test_converts_table_id_to_realm(self):
        cmip6_attrs = {"table_id": "Amon", "variable_id": "tas"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["table_id"] == "atmos"
        assert cmip7_attrs["frequency"] == "mon"

    def test_updates_conventions(self):
        cmip6_attrs = {"Conventions": "CF-1.6", "variable_id": "tas"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["Conventions"] == "CF-1.12 CMIP-7.0"

    def test_preserves_existing_attributes(self):
        cmip6_attrs = {
            "variable_id": "tas",
            "source_id": "ACCESS-ESM1-5",
            "institution_id": "CSIRO",
            "custom_attr": "should_be_preserved",
        }
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["source_id"] == "ACCESS-ESM1-5"
        assert cmip7_attrs["institution_id"] == "CSIRO"
        assert cmip7_attrs["custom_attr"] == "should_be_preserved"


class TestConvertCmip6Dataset:
    @pytest.fixture
    def sample_cmip6_dataset(self) -> xr.Dataset:
        """Create a minimal CMIP6-style dataset for testing."""
        time = np.arange(12)
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(0, 360, 10)
        rng = np.random.default_rng(42)

        data = rng.random((len(time), len(lat), len(lon)))

        ds = xr.Dataset(
            {"tas": (["time", "lat", "lon"], data)},
            coords={
                "time": time,
                "lat": lat,
                "lon": lon,
            },
            attrs={
                "variable_id": "tas",
                "table_id": "Amon",
                "source_id": "ACCESS-ESM1-5",
                "experiment_id": "historical",
                "variant_label": "r1i1p1f1",
                "institution_id": "CSIRO",
                "grid_label": "gn",
                "Conventions": "CF-1.6",
            },
        )
        return ds

    def test_converts_global_attributes(self, sample_cmip6_dataset: xr.Dataset):
        ds_cmip7 = convert_cmip6_dataset(sample_cmip6_dataset)

        assert ds_cmip7.attrs["mip_era"] == "CMIP7"
        assert ds_cmip7.attrs["region"] == "GLB"
        assert ds_cmip7.attrs["branding_suffix"] == "tavg-h2m-hxy-u"

    def test_renames_variables(self, sample_cmip6_dataset: xr.Dataset):
        ds_cmip7 = convert_cmip6_dataset(sample_cmip6_dataset, rename_variables=True)

        assert "tas_tavg-h2m-hxy-u" in ds_cmip7.data_vars
        assert "tas" not in ds_cmip7.data_vars

    def test_preserves_variables_when_rename_disabled(self, sample_cmip6_dataset: xr.Dataset):
        ds_cmip7 = convert_cmip6_dataset(sample_cmip6_dataset, rename_variables=False)

        assert "tas" in ds_cmip7.data_vars
        assert "tas_tavg-h2m-hxy-u" not in ds_cmip7.data_vars

    def test_preserves_data_values(self, sample_cmip6_dataset: xr.Dataset):
        original_data = sample_cmip6_dataset["tas"].values.copy()
        ds_cmip7 = convert_cmip6_dataset(sample_cmip6_dataset, rename_variables=True)

        np.testing.assert_array_equal(ds_cmip7["tas_tavg-h2m-hxy-u"].values, original_data)

    def test_does_not_modify_original_by_default(self, sample_cmip6_dataset: xr.Dataset):
        original_mip_era = sample_cmip6_dataset.attrs.get("mip_era")
        convert_cmip6_dataset(sample_cmip6_dataset)

        # Original should be unchanged
        assert sample_cmip6_dataset.attrs.get("mip_era") == original_mip_era


class TestCreateCmip7InstanceId:
    def test_creates_valid_instance_id(self):
        attrs = {
            "mip_era": "CMIP7",
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "region": "GLB",
            "frequency": "mon",
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "grid_label": "gn",
            "version": "v20240101",
        }
        instance_id = create_cmip7_instance_id(attrs)

        expected = (
            "CMIP7.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.GLB.mon.tas.tavg-h2m-hxy-u.gn.v20240101"
        )
        assert instance_id == expected


class TestCreateCmip7Path:
    def test_creates_valid_path(self):
        attrs = {
            "drs_specs": "MIP-DRS7",
            "mip_era": "CMIP7",
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "region": "GLB",
            "frequency": "mon",
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "grid_label": "gn",
            "version": "v20240101",
        }
        path = create_cmip7_path(attrs)

        expected = "MIP-DRS7/CMIP7/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/GLB/mon/tas/tavg-h2m-hxy-u/gn/v20240101"  # noqa: E501
        assert path == expected


class TestVariableBrandingCoverage:
    """Test that common CMIP6 variables have branding defined."""

    @pytest.mark.parametrize(
        "variable_id",
        [
            "tas",
            "pr",
            "psl",
            "tos",
            "siconc",
            "areacella",
            "sftlf",
        ],
    )
    def test_common_variables_have_branding(self, variable_id: str):
        assert variable_id in VARIABLE_BRANDING
