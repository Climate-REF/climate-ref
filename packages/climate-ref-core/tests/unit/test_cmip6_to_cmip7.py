"""Tests for CMIP6 to CMIP7 format conversion."""

import attrs
import numpy as np
import pytest
import xarray as xr

from climate_ref_core.cmip6_to_cmip7 import (
    _DREQ_VARIABLES,
    CMIP6_ONLY_ATTRIBUTES,
    BrandingSuffix,
    DReqVariableMapping,
    convert_cmip6_dataset,
    convert_cmip6_to_cmip7_attrs,
    convert_variant_index,
    create_cmip7_filename,
    create_cmip7_path,
    get_branding_suffix,
    get_cmip7_compound_name,
    get_frequency_from_table,
    get_realm,
)


class TestDReqDataLoading:
    """Test that DReq data is loaded correctly at import time."""

    def test_dreq_variables_loaded(self):
        assert len(_DREQ_VARIABLES) > 30, f"Expected >30 DReq variables, got {len(_DREQ_VARIABLES)}"

    def test_dreq_entry_is_dreq_variable_mapping(self):
        entry = _DREQ_VARIABLES.get("Amon.tas")
        assert entry is not None, "Amon.tas should be in DReq"
        assert isinstance(entry, DReqVariableMapping)

    def test_dreq_entry_has_required_fields(self):
        entry = _DREQ_VARIABLES["Amon.tas"]
        assert entry.table_id == "Amon"
        assert entry.variable_id == "tas"
        assert entry.cmip6_compound_name == "Amon.tas"
        assert entry.cmip7_compound_name != ""
        assert entry.branded_variable_name != ""
        assert entry.out_name == "tas"
        assert entry.branding_suffix != ""
        assert entry.temporal_label != ""
        assert entry.vertical_label != ""
        assert entry.horizontal_label != ""
        assert entry.area_label != ""
        assert entry.realm != ""
        assert entry.region != ""


class TestDReqVariableMapping:
    """Test DReqVariableMapping serialisation and deserialisation."""

    @pytest.fixture
    def sample_mapping(self) -> DReqVariableMapping:
        return DReqVariableMapping(
            table_id="Amon",
            variable_id="tas",
            cmip6_compound_name="Amon.tas",
            cmip7_compound_name="atmos.tas.tavg-h2m-hxy-u.mon.glb",
            branded_variable_name="tas_tavg-h2m-hxy-u",
            out_name="tas",
            branding_suffix="tavg-h2m-hxy-u",
            temporal_label="tavg",
            vertical_label="h2m",
            horizontal_label="hxy",
            area_label="u",
            realm="atmos",
            region="glb",
        )

    def test_to_dict(self, sample_mapping: DReqVariableMapping):
        d = sample_mapping.to_dict()
        assert isinstance(d, dict)
        assert d["table_id"] == "Amon"
        assert d["variable_id"] == "tas"
        assert d["realm"] == "atmos"
        assert d["branding_suffix"] == "tavg-h2m-hxy-u"

    def test_from_dict(self):
        data = {
            "table_id": "Omon",
            "variable_id": "tos",
            "cmip6_compound_name": "Omon.tos",
            "cmip7_compound_name": "ocean.tos.tavg-u-hxy-sea.mon.glb",
            "branded_variable_name": "tos_tavg-u-hxy-sea",
            "out_name": "tos",
            "branding_suffix": "tavg-u-hxy-sea",
            "temporal_label": "tavg",
            "vertical_label": "u",
            "horizontal_label": "hxy",
            "area_label": "sea",
            "realm": "ocean",
            "region": "glb",
        }
        mapping = DReqVariableMapping.from_dict(data)
        assert mapping.table_id == "Omon"
        assert mapping.variable_id == "tos"
        assert mapping.realm == "ocean"
        assert mapping.area_label == "sea"

    def test_round_trip(self, sample_mapping: DReqVariableMapping):
        restored = DReqVariableMapping.from_dict(sample_mapping.to_dict())
        assert restored == sample_mapping

    def test_is_frozen(self, sample_mapping: DReqVariableMapping):
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            sample_mapping.table_id = "Omon"  # type: ignore[misc]

    def test_from_dict_missing_field_raises(self):
        data = {"table_id": "Amon", "variable_id": "tas"}
        with pytest.raises(KeyError):
            DReqVariableMapping.from_dict(data)


class TestBrandingSuffix:
    def test_str_representation(self):
        suffix = BrandingSuffix("tavg", "h2m", "hxy", "u")
        assert str(suffix) == "tavg-h2m-hxy-u"

    def test_default_values(self):
        suffix = BrandingSuffix()
        assert str(suffix) == "tavg-u-hxy-u"


class TestGetBrandingSuffix:
    """Test branding suffix lookup using DReq compound names."""

    def test_known_variable_tas(self):
        suffix = get_branding_suffix("Amon", "tas")
        assert suffix.temporal_label == "tavg"
        assert suffix.vertical_label == "h2m"
        assert suffix.horizontal_label == "hxy"
        assert suffix.area_label == "u"

    def test_known_variable_tos(self):
        """DReq says Omon.tos has vertical_label=u, not d0m."""
        suffix = get_branding_suffix("Omon", "tos")
        assert suffix.vertical_label == "u"
        assert suffix.area_label == "sea"

    def test_unknown_variable_raises(self):
        with pytest.raises(KeyError, match="not found in Data Request"):
            get_branding_suffix("Amon", "completely_unknown_var_xyz")

    def test_unknown_table_raises(self):
        with pytest.raises(KeyError, match="not found in Data Request"):
            get_branding_suffix("Xmon", "tas")

    def test_hus_uses_p19(self):
        """DReq says Amon.hus uses p19 vertical, not al."""
        suffix = get_branding_suffix("Amon", "hus")
        assert suffix.vertical_label == "p19"
        assert suffix.area_label == "u"

    def test_ta_uses_p19_air(self):
        """DReq says Amon.ta uses p19 vertical and air area."""
        suffix = get_branding_suffix("Amon", "ta")
        assert suffix.vertical_label == "p19"
        assert suffix.area_label == "air"

    def test_ua_uses_p19_air(self):
        suffix = get_branding_suffix("Amon", "ua")
        assert suffix.vertical_label == "p19"
        assert suffix.area_label == "air"

    def test_va_uses_p19_air(self):
        suffix = get_branding_suffix("Amon", "va")
        assert suffix.vertical_label == "p19"
        assert suffix.area_label == "air"

    def test_zg_uses_p19_air(self):
        suffix = get_branding_suffix("Amon", "zg")
        assert suffix.vertical_label == "p19"
        assert suffix.area_label == "air"

    def test_tasmax_uses_tmaxavg(self):
        """DReq says Amon.tasmax uses tmaxavg temporal, not tmax."""
        suffix = get_branding_suffix("Amon", "tasmax")
        assert suffix.temporal_label == "tmaxavg"
        assert suffix.vertical_label == "h2m"

    def test_tasmin_uses_tminavg(self):
        suffix = get_branding_suffix("Amon", "tasmin")
        assert suffix.temporal_label == "tminavg"
        assert suffix.vertical_label == "h2m"

    def test_siconc(self):
        suffix = get_branding_suffix("SImon", "siconc")
        assert suffix.temporal_label == "tavg"
        assert suffix.area_label == "u"

    def test_vegfrac(self):
        suffix = get_branding_suffix("Emon", "vegFrac")
        assert suffix.temporal_label == "tavg"
        assert suffix.area_label == "u"

    def test_cveg(self):
        suffix = get_branding_suffix("Lmon", "cVeg")
        assert suffix.area_label == "lnd"

    def test_treefrac(self):
        suffix = get_branding_suffix("Lmon", "treeFrac")
        assert suffix.area_label == "u"


class TestGetBrandingSuffixREFVariables:
    """Test that ALL variables used by REF diagnostic providers resolve correctly."""

    @pytest.mark.parametrize(
        ("table_id", "variable_id", "expected_suffix"),
        [
            ("Amon", "cli", "tavg-al-hxy-u"),
            ("Amon", "clivi", "tavg-u-hxy-u"),
            ("Amon", "clt", "tavg-u-hxy-u"),
            ("Amon", "clwvi", "tavg-u-hxy-u"),
            ("Amon", "fco2antt", "tavg-u-hxy-u"),
            ("Amon", "hfls", "tavg-u-hxy-u"),
            ("Amon", "hfss", "tavg-u-hxy-u"),
            ("Amon", "hurs", "tavg-h2m-hxy-u"),
            ("Amon", "hus", "tavg-p19-hxy-u"),
            ("Amon", "pr", "tavg-u-hxy-u"),
            ("Amon", "psl", "tavg-u-hxy-u"),
            ("Amon", "rlds", "tavg-u-hxy-u"),
            ("Amon", "rlus", "tavg-u-hxy-u"),
            ("Amon", "rlut", "tavg-u-hxy-u"),
            ("Amon", "rlutcs", "tavg-u-hxy-u"),
            ("Amon", "rsds", "tavg-u-hxy-u"),
            ("Amon", "rsdt", "tavg-u-hxy-u"),
            ("Amon", "rsus", "tavg-u-hxy-u"),
            ("Amon", "rsut", "tavg-u-hxy-u"),
            ("Amon", "rsutcs", "tavg-u-hxy-u"),
            ("Amon", "ta", "tavg-p19-hxy-air"),
            ("Amon", "tas", "tavg-h2m-hxy-u"),
            ("Amon", "tasmax", "tmaxavg-h2m-hxy-u"),
            ("Amon", "tasmin", "tminavg-h2m-hxy-u"),
            ("Amon", "tauu", "tavg-u-hxy-u"),
            ("Amon", "ts", "tavg-u-hxy-u"),
            ("Amon", "ua", "tavg-p19-hxy-air"),
            ("Amon", "uas", "tavg-h10m-hxy-u"),
            ("Amon", "va", "tavg-p19-hxy-air"),
            ("Amon", "vas", "tavg-h10m-hxy-u"),
            ("Amon", "zg", "tavg-p19-hxy-air"),
            ("Emon", "vegFrac", "tavg-u-hxy-u"),
            ("Lmon", "cVeg", "tavg-u-hxy-lnd"),
            ("Lmon", "treeFrac", "tavg-u-hxy-u"),
            ("Omon", "tos", "tavg-u-hxy-sea"),
            ("SImon", "siconc", "tavg-u-hxy-u"),
        ],
        ids=lambda x: x if isinstance(x, str) and "." not in x else "",
    )
    def test_ref_variable_branding(self, table_id: str, variable_id: str, expected_suffix: str):
        suffix = get_branding_suffix(table_id, variable_id)
        assert str(suffix) == expected_suffix, (
            f"{table_id}.{variable_id}: expected {expected_suffix}, got {suffix}"
        )


class TestGetCmip7CompoundName:
    """Test CMIP7 compound name lookup."""

    def test_amon_tas(self):
        assert get_cmip7_compound_name("Amon", "tas") == "atmos.tas.tavg-h2m-hxy-u.mon.glb"

    def test_omon_tos(self):
        assert get_cmip7_compound_name("Omon", "tos") == "ocean.tos.tavg-u-hxy-sea.mon.glb"

    def test_simon_siconc(self):
        assert get_cmip7_compound_name("SImon", "siconc") == "seaIce.siconc.tavg-u-hxy-u.mon.glb"

    def test_amon_tasmax(self):
        cn = get_cmip7_compound_name("Amon", "tasmax")
        assert cn == "atmos.tas.tmaxavg-h2m-hxy-u.mon.glb"

    def test_lmon_cveg(self):
        cn = get_cmip7_compound_name("Lmon", "cVeg")
        assert cn == "land.cVeg.tavg-u-hxy-lnd.mon.glb"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="not found in Data Request"):
            get_cmip7_compound_name("Xmon", "unknown_xyz")


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


class TestGetRealm:
    """Test realm lookup from DReq entries."""

    def test_amon_tas(self):
        assert get_realm("Amon", "tas") == "atmos"

    def test_omon_tos(self):
        assert get_realm("Omon", "tos") == "ocean"

    def test_lmon_cveg(self):
        assert get_realm("Lmon", "cVeg") == "land"

    def test_simon_siconc(self):
        assert get_realm("SImon", "siconc") == "seaIce"

    def test_emon_vegfrac(self):
        assert get_realm("Emon", "vegFrac") == "land"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="not found in Data Request"):
            get_realm("Amon", "totally_unknown_xyz")


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
        assert cmip7_attrs["region"] == "glb"
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

    def test_converts_table_id_to_realm(self):
        cmip6_attrs = {"table_id": "Amon", "variable_id": "tas"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["realm"] == "atmos"
        assert cmip7_attrs["frequency"] == "mon"
        assert cmip7_attrs["cmip6_compound_name"] == "Amon.tas"

        assert "table_id" not in cmip7_attrs

    def test_uses_dreq_branding_when_table_id_present(self):
        """When table_id is in attrs, DReq compound lookup is used."""
        cmip6_attrs = {"table_id": "Amon", "variable_id": "hus"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        # DReq says Amon.hus -> p19 vertical, not al
        assert cmip7_attrs["vertical_label"] == "p19"
        assert cmip7_attrs["branding_suffix"] == "tavg-p19-hxy-u"

    def test_updates_conventions(self):
        cmip6_attrs = {"Conventions": "CF-1.6", "variable_id": "tas", "table_id": "Amon"}
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        assert cmip7_attrs["Conventions"] == "CF-1.12"

    def test_converts_variant_indices(self):
        """Test that CMIP6 integer indices are converted to CMIP7 string format."""
        cmip6_attrs = {
            "variable_id": "tas",
            "table_id": "Amon",
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
            "table_id": "Amon",
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
            "further_info_url": "https://furtherinfo.es-doc.org/CMIP6.CSIRO.ACCESS-ESM1-5.historical.none.r1i1p1f1",
            "grid": "native atmosphere N96 grid (145x192 latxlon)",
            "member_id": "r1i1p1f1",
            "sub_experiment": "none",
            "sub_experiment_id": "none",
            "grid_label": "gn",
            "variant_label": "r1i1p1f1",
        }
        cmip7_attrs = convert_cmip6_to_cmip7_attrs(cmip6_attrs)

        for attr in CMIP6_ONLY_ATTRIBUTES:
            assert attr not in cmip7_attrs, f"{attr} should be removed"

        assert cmip7_attrs["grid_label"] == "gn"
        assert "variant_label" in cmip7_attrs

    def test_missing_table_id_raises(self):
        """table_id is required for conversion."""
        with pytest.raises(KeyError):
            convert_cmip6_to_cmip7_attrs({"variable_id": "tas"})


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
        assert ds_cmip7.attrs["region"] == "glb"
        assert ds_cmip7.attrs["branding_suffix"] == "tavg-h2m-hxy-u"
        assert ds_cmip7.attrs["branded_variable"] == "tas_tavg-h2m-hxy-u"
        assert ds_cmip7.attrs["Conventions"] == "CF-1.12"

    def test_preserves_variable_name(self, sample_cmip6_dataset: xr.Dataset):
        ds_cmip7 = convert_cmip6_dataset(sample_cmip6_dataset)

        assert "tas" in ds_cmip7.data_vars
        assert "tas_tavg-h2m-hxy-u" not in ds_cmip7.data_vars

    def test_does_not_modify_original_by_default(self, sample_cmip6_dataset: xr.Dataset):
        original_mip_era = sample_cmip6_dataset.attrs.get("mip_era")
        convert_cmip6_dataset(sample_cmip6_dataset)

        assert sample_cmip6_dataset.attrs.get("mip_era") == original_mip_era

    def test_uses_dreq_for_branding(self):
        """Verify dataset conversion uses DReq lookup."""
        rng = np.random.default_rng(42)
        ds = xr.Dataset(
            {"hus": (["time", "plev", "lat", "lon"], rng.random((3, 2, 3, 4)))},
            attrs={
                "variable_id": "hus",
                "table_id": "Amon",
                "source_id": "TestModel",
                "experiment_id": "historical",
                "variant_label": "r1i1p1f1",
            },
        )
        ds_cmip7 = convert_cmip6_dataset(ds)
        # DReq says Amon.hus -> p19, not al
        assert ds_cmip7.attrs["vertical_label"] == "p19"
        assert ds_cmip7.attrs["branding_suffix"] == "tavg-p19-hxy-u"


class TestCreateCmip7Filename:
    """Test CMIP7 filename generation per MIP-DRS7 spec."""

    def test_creates_valid_filename_with_time_range(self):
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
        attrs = {
            "variable_id": "tas",
            "branding_suffix": "tavg-h2m-hxy-u",
            "source_id": "TestModel",
            "experiment_id": "piControl",
            "variant_label": "r1i1p1f1",
        }
        filename = create_cmip7_filename(attrs)

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
        attrs = {
            "institution_id": "TestInst",
            "source_id": "TestModel",
            "experiment_id": "piControl",
            "variant_label": "r1i1p1f1",
            "variable_id": "pr",
            "branding_suffix": "tavg-u-hxy-u",
        }
        path = create_cmip7_path(attrs)

        assert path.startswith("MIP-DRS7/CMIP7/CMIP/")
        assert "/glb/" in path
        assert "/mon/" in path
        assert path.endswith("/gn/v1")
