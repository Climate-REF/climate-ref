"""Tests for CMIP6 to CMIP7 format conversion."""

import numpy as np
import pytest
import xarray as xr

from climate_ref_core.cmip6_to_cmip7 import (
    CMIP6_ONLY_ATTRIBUTES,
    VARIABLE_BRANDING,
    BrandingSuffix,
    convert_cmip6_dataset,
    convert_cmip6_to_cmip7_attrs,
    convert_variant_index,
    create_cmip7_filename,
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


class TestConvertVariantIndex:
    """Test CMIP6 integer index to CMIP7 string format conversion."""

    @pytest.mark.parametrize(
        ("value", "prefix", "expected"),
        [
            pytest.param(1, "r", "r1", id="int_r"),
            pytest.param(2, "i", "i2", id="int_i"),
            pytest.param(123, "p", "p123", id="int_p_large"),
            pytest.param(4, "f", "f4", id="int_f"),
            pytest.param("1", "r", "r1", id="str_numeric"),
            pytest.param("r1", "r", "r1", id="str_with_prefix_r"),
            pytest.param("i2", "i", "i2", id="str_with_prefix_i"),
        ],
    )
    def test_index_conversion(self, value, prefix, expected):
        assert convert_variant_index(value, prefix) == expected


class TestConvertCmip6ToCmip7Attrs:
    """Tests for CMIP6 to CMIP7 attribute conversion.

    Based on CMIP7 Global Attributes V1.0 (DOI: 10.5281/zenodo.17250297).
    """

    def test_adds_new_attributes(self):
        cmip6_attrs = {
            "variable_id": "tas",
            "table_id": "Amon",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
        }
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        # Core CMIP7 attributes
        assert cmip7_attrs["mip_era"] == "CMIP7"
        assert cmip7_attrs["region"] == "glb"  # lowercase per CMIP7 spec
        assert cmip7_attrs["drs_specs"] == "MIP-DRS7"
        assert cmip7_attrs["branding_suffix"] == "tavg-h2m-hxy-u"

        # New mandatory CMIP7 attributes (not in CMIP6)
        assert cmip7_attrs["data_specs_version"] == "MIP-DS7.1.0.0"
        assert cmip7_attrs["product"] == "model-output"
        assert cmip7_attrs["license_id"] == "CC-BY-4.0"
        assert cmip7_attrs["branded_variable"] == "tas_tavg-h2m-hxy-u"
        assert "tracking_id" in cmip7_attrs
        assert cmip7_attrs["tracking_id"].startswith("hdl:21.14107/")
        assert "creation_date" in cmip7_attrs

        # Removed CMIP6 attributes should not be present
        assert "archive_id" not in cmip7_attrs
        assert "host_collection" not in cmip7_attrs
        assert "cv_version" not in cmip7_attrs

    def test_converts_table_id_to_realm(self):
        cmip6_attrs = {"table_id": "Amon", "variable_id": "tas"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["table_id"] == "atmos"
        assert cmip7_attrs["realm"] == "atmos"
        assert cmip7_attrs["frequency"] == "mon"
        assert cmip7_attrs["cmip6_compound_name"] == "Amon.tas"

    def test_updates_conventions(self):
        cmip6_attrs = {"Conventions": "CF-1.6", "variable_id": "tas"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        # CMIP7 spec says just "CF-1.12", not "CF-1.12 CMIP-7.0"
        assert cmip7_attrs["Conventions"] == "CF-1.12"

    def test_converts_variant_indices(self):
        """Test that CMIP6 integer indices are converted to CMIP7 string format."""
        cmip6_attrs = {
            "variable_id": "tas",
            "realization_index": 1,
            "initialization_index": 2,
            "physics_index": 3,
            "forcing_index": 4,
        }
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["realization_index"] == "r1"
        assert cmip7_attrs["initialization_index"] == "i2"
        assert cmip7_attrs["physics_index"] == "p3"
        assert cmip7_attrs["forcing_index"] == "f4"
        assert cmip7_attrs["variant_label"] == "r1i2p3f4"

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

    def test_removes_cmip6_only_attributes(self):
        """Test that CMIP6-only attributes are removed during conversion."""
        cmip6_attrs = {
            "variable_id": "tas",
            "table_id": "Amon",
            # CMIP6-only attributes that should be removed
            "further_info_url": "https://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.historical.none.r1i1p1f1",
            "grid": "native atmosphere N96 grid (145x192 latxlon)",
            "member_id": "r1i1p1f1",
            "sub_experiment": "none",
            "sub_experiment_id": "none",
            # These should be preserved
            "grid_label": "gn",
            "variant_label": "r1i1p1f1",
        }
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        # CMIP6-only attributes should be removed
        for attr in CMIP6_ONLY_ATTRIBUTES:
            assert attr not in cmip7_attrs, f"{attr} should be removed"

        # grid_label and variant_label should still be present
        assert cmip7_attrs["grid_label"] == "gn"
        assert "variant_label" in cmip7_attrs


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
        assert ds_cmip7.attrs["region"] == "glb"  # lowercase per CMIP7 spec
        assert ds_cmip7.attrs["branding_suffix"] == "tavg-h2m-hxy-u"
        assert ds_cmip7.attrs["branded_variable"] == "tas_tavg-h2m-hxy-u"
        assert ds_cmip7.attrs["Conventions"] == "CF-1.12"

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


class TestCreateCmip7Filename:
    """Test CMIP7 filename generation per MIP-DRS7 spec."""

    def test_creates_valid_filename_with_time_range(self):
        """Test filename with time range for time-dependent variables."""
        attrs = {
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "frequency": "mon",
            "region": "glb",
            "grid_label": "g13s",
            "source_id": "CanESM6-MR",
            "experiment_id": "historical",
            "variant_label": "r2i1p1f1",
        }
        filename = create_cmip7_filename(attrs, "190001-190912")

        expected = "tas_tavg-h2m-hxy-u_mon_glb_g13s_CanESM6-MR_historical_r2i1p1f1_190001-190912.nc"
        assert filename == expected

    def test_creates_valid_filename_without_time_range(self):
        """Test filename without time range for fixed/time-independent variables."""
        attrs = {
            "variable_id": "areacella",
            "branding_suffix": "ti-u-hxy-u",
            "frequency": "fx",
            "region": "glb",
            "grid_label": "gn",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
        }
        filename = create_cmip7_filename(attrs)

        expected = "areacella_ti-u-hxy-u_fx_glb_gn_ACCESS-ESM1-5_historical_r1i1p1f1.nc"
        assert filename == expected

    def test_uses_defaults_for_missing_attributes(self):
        """Test that defaults are used for missing optional attributes."""
        attrs = {
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "source_id": "TestModel",
            "experiment_id": "piControl",
            "variant_label": "r1i1p1f1",
        }
        filename = create_cmip7_filename(attrs)

        # Should use default frequency=mon, region=glb, grid_label=gn
        assert filename == "tas_tavg-h2m-hxy-u_mon_glb_gn_TestModel_piControl_r1i1p1f1.nc"


class TestCreateCmip7Path:
    """Test CMIP7 directory path generation per MIP-DRS7 spec."""

    def test_creates_valid_path(self):
        attrs = {
            "drs_specs": "MIP-DRS7",
            "mip_era": "CMIP7",
            "activity_id": "CMIP",
            "institution_id": "CCCma",
            "source_id": "CanESM6-MR",
            "experiment_id": "historical",
            "variant_label": "r2i1p1f1",
            "region": "glb",
            "frequency": "mon",
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "grid_label": "g13s",
        }
        path = create_cmip7_path(attrs, "v20250622")

        expected = "MIP-DRS7/CMIP7/CMIP/CCCma/CanESM6-MR/historical/r2i1p1f1/glb/mon/tas/tavg-h2m-hxy-u/g13s/v20250622"  # noqa: E501
        assert path == expected

    def test_uses_version_from_attrs(self):
        """Test that version is taken from attrs if not provided as argument."""
        attrs = {
            "drs_specs": "MIP-DRS7",
            "mip_era": "CMIP7",
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "region": "glb",
            "frequency": "mon",
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "grid_label": "gn",
            "version": "v20240101",
        }
        path = create_cmip7_path(attrs)

        assert path.endswith("/v20240101")

    def test_uses_defaults_for_missing_attributes(self):
        """Test that defaults are used for missing attributes."""
        attrs = {
            "institution_id": "TestInst",
            "source_id": "TestModel",
            "experiment_id": "piControl",
            "variant_label": "r1i1p1f1",
            "variable_id": "pr",
            "branding_suffix": "tavg-u-hxy-u",
        }
        path = create_cmip7_path(attrs)

        # Should use defaults: MIP-DRS7, CMIP7, CMIP, glb, mon, gn, v1
        assert path.startswith("MIP-DRS7/CMIP7/CMIP/")
        assert "/glb/" in path
        assert "/mon/" in path
        assert path.endswith("/gn/v1")


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
