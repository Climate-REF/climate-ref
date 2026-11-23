"""Tests for the bundled CMIP7 Data Request mappings."""

import attrs
import pytest

from climate_ref_core.data_request import DReqVariableMapping, dreq_entries, get_dreq_entry


class TestDReqDataLoading:
    """Test that DReq data is loaded correctly at import time."""

    def test_dreq_entry_is_dreq_variable_mapping(self):
        entry = get_dreq_entry(table_id="Amon", variable_id="tas")
        assert entry is not None, "Amon.tas should be in DReq"
        assert isinstance(entry, DReqVariableMapping)

    def test_dreq_entry_has_required_fields(self):
        entry = get_dreq_entry(table_id="Amon", variable_id="tas")
        assert entry.table_id == "Amon"
        assert entry.variable_id == "tas"
        assert entry.cmip6_compound_name == "Amon.tas"
        assert entry.cmip7_compound_name != ""
        assert entry.branded_variable != ""
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
            branded_variable="tas_tavg-h2m-hxy-u",
            branding_suffix="tavg-h2m-hxy-u",
            temporal_label="tavg",
            vertical_label="h2m",
            horizontal_label="hxy",
            area_label="u",
            realm="atmos",
            region="glb",
            frequency="mon",
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
            "branded_variable": "tos_tavg-u-hxy-sea",
            "branding_suffix": "tavg-u-hxy-sea",
            "temporal_label": "tavg",
            "vertical_label": "u",
            "horizontal_label": "hxy",
            "area_label": "sea",
            "realm": "ocean",
            "region": "glb",
            "frequency": "mon",
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


class TestREFVariables:
    """Test that ALL variables used by REF diagnostic providers resolve correctly."""

    @pytest.mark.parametrize(
        ("table_id", "variable_id", "expected_suffix"),
        [
            pytest.param("Amon", "cli", "tavg-al-hxy-u", id="Amon_cli"),
            pytest.param("Amon", "clivi", "tavg-u-hxy-u", id="Amon_clivi"),
            pytest.param("Amon", "clt", "tavg-u-hxy-u", id="Amon_clt"),
            pytest.param("Amon", "clwvi", "tavg-u-hxy-u", id="Amon_clwvi"),
            pytest.param("Amon", "fco2antt", "tavg-u-hxy-u", id="Amon_fco2antt"),
            pytest.param("Amon", "hfls", "tavg-u-hxy-u", id="Amon_hfls"),
            pytest.param("Amon", "hfss", "tavg-u-hxy-u", id="Amon_hfss"),
            pytest.param("Amon", "hurs", "tavg-h2m-hxy-u", id="Amon_hurs"),
            pytest.param("Amon", "hus", "tavg-p19-hxy-u", id="Amon_hus"),
            pytest.param("Amon", "pr", "tavg-u-hxy-u", id="Amon_pr"),
            pytest.param("Amon", "psl", "tavg-u-hxy-u", id="Amon_psl"),
            pytest.param("Amon", "rlds", "tavg-u-hxy-u", id="Amon_rlds"),
            pytest.param("Amon", "rlus", "tavg-u-hxy-u", id="Amon_rlus"),
            pytest.param("Amon", "rlut", "tavg-u-hxy-u", id="Amon_rlut"),
            pytest.param("Amon", "rlutcs", "tavg-u-hxy-u", id="Amon_rlutcs"),
            pytest.param("Amon", "rsds", "tavg-u-hxy-u", id="Amon_rsds"),
            pytest.param("Amon", "rsdt", "tavg-u-hxy-u", id="Amon_rsdt"),
            pytest.param("Amon", "rsus", "tavg-u-hxy-u", id="Amon_rsus"),
            pytest.param("Amon", "rsut", "tavg-u-hxy-u", id="Amon_rsut"),
            pytest.param("Amon", "rsutcs", "tavg-u-hxy-u", id="Amon_rsutcs"),
            pytest.param("Amon", "ta", "tavg-p19-hxy-air", id="Amon_ta"),
            pytest.param("Amon", "tas", "tavg-h2m-hxy-u", id="Amon_tas"),
            pytest.param("Amon", "tasmax", "tmaxavg-h2m-hxy-u", id="Amon_tasmax"),
            pytest.param("Amon", "tasmin", "tminavg-h2m-hxy-u", id="Amon_tasmin"),
            pytest.param("Amon", "tauu", "tavg-u-hxy-u", id="Amon_tauu"),
            pytest.param("Amon", "ts", "tavg-u-hxy-u", id="Amon_ts"),
            pytest.param("Amon", "ua", "tavg-p19-hxy-air", id="Amon_ua"),
            pytest.param("Amon", "uas", "tavg-h10m-hxy-u", id="Amon_uas"),
            pytest.param("Amon", "va", "tavg-p19-hxy-air", id="Amon_va"),
            pytest.param("Amon", "vas", "tavg-h10m-hxy-u", id="Amon_vas"),
            pytest.param("Amon", "zg", "tavg-p19-hxy-air", id="Amon_zg"),
            pytest.param("Emon", "vegFrac", "tavg-u-hxy-u", id="Emon_vegFrac"),
            pytest.param("Lmon", "cVeg", "tavg-u-hxy-lnd", id="Lmon_cVeg"),
            pytest.param("Lmon", "treeFrac", "tavg-u-hxy-u", id="Lmon_treeFrac"),
            pytest.param("Omon", "tos", "tavg-u-hxy-sea", id="Omon_tos"),
            pytest.param("SImon", "siconc", "tavg-u-hxy-u", id="SImon_siconc"),
        ],
    )
    def test_ref_variable_branding(self, table_id: str, variable_id: str, expected_suffix: str):
        suffix = get_dreq_entry(table_id, variable_id).branding_suffix
        assert suffix == expected_suffix, (
            f"{table_id}.{variable_id}: expected {expected_suffix}, got {suffix}"
        )


class TestGetDreqEntry:
    def test_cmip7_names(self):
        entry = get_dreq_entry("Amon", "tasmax")
        assert entry.variable_id == "tas"
        assert entry.branded_variable == "tas_tmaxavg-h2m-hxy-u"
        assert entry.cmip7_compound_name == "atmos.tas.tmaxavg-h2m-hxy-u.mon.glb"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="not found in Data Request"):
            get_dreq_entry("Xmon", "unknown_xyz")


def test_dreq_entries_are_unique_per_compound_name():
    entries = dreq_entries()
    assert entries
    assert len({entry.cmip6_compound_name for entry in entries}) == len(entries)
