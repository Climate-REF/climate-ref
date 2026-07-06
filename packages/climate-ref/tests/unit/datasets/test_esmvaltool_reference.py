"""Tests for the ESMValTool reference dataset adapter and its custom-format parser."""

from pathlib import Path

import pytest

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.datasets import ingest_datasets
from climate_ref.datasets.esmvaltool_reference import (
    ESMValToolReferenceDatasetAdapter,
    parse_esmvaltool_reference,
)


@pytest.mark.parametrize(
    "path, expected",
    [
        pytest.param(
            "/root/ESMValTool/OBS/Tier2/CERES-EBAF/OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc",
            {
                "project": "OBS",
                "source_id": "CERES-EBAF",
                "variable_id": "rlut",
                "table_id": "Amon",
                "version": "Ed4.2",
                "data_type": "sat",
                "tier": 2,
                "start_time": "2000-03-01",
                "end_time": "2023-11-30",
            },
            id="obs",
        ),
        pytest.param(
            "/root/ESMValTool/OBS/Tier2/TROPFLUX/OBS6_TROPFLUX_reanaly_v1_Omon_tos_197901-201812.nc",
            {
                "project": "OBS6",
                "source_id": "TROPFLUX",
                "variable_id": "tos",
                "table_id": "Omon",
                "version": "v1",
                "data_type": "reanaly",
                "tier": 2,
            },
            id="obs6",
        ),
        pytest.param(
            "/root/ESMValTool/native6/Tier3/ERA5/v1/mon/hus/era5_specific_humidity_1980_monthly.nc",
            {
                "project": "native6",
                "source_id": "ERA5",
                "variable_id": "hus",
                "table_id": "mon",
                "version": "v1",
                "data_type": None,
                "tier": 3,
                "start_time": None,
                "end_time": None,
            },
            id="native6",
        ),
        pytest.param(
            "/root/ESMValTool/obs4MIPs/GPCP-V2.3/v20180519/pr_GPCP-SG_L3_v2.3_197901-201710.nc",
            {
                "project": "obs4MIPs",
                "source_id": "GPCP-V2.3",
                "variable_id": "pr",
                "table_id": "mon",
                "version": "v20180519",
                "tier": None,
            },
            id="obs4mips",
        ),
    ],
)
def test_parse_layouts(path, expected):
    result = parse_esmvaltool_reference(path)
    assert "INVALID_ASSET" not in result
    for key, value in expected.items():
        assert result[key] == value, key
    assert result["path"] == path


def test_parse_rejects_unknown_layout():
    result = parse_esmvaltool_reference("/root/somewhere/random.nc")
    assert result["INVALID_ASSET"] == "/root/somewhere/random.nc"


def test_parse_rejects_malformed_obs_filename():
    result = parse_esmvaltool_reference("/root/ESMValTool/OBS/Tier2/FOO/OBS_FOO_sat.nc")
    assert "INVALID_ASSET" in result


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.fixture
def reference_tree(tmp_path) -> Path:
    """A minimal ESMValTool reference tree of empty .nc files (parser is path-based)."""
    root = tmp_path / "ESMValTool"
    files = [
        # single-file OBS dataset
        "OBS/Tier2/CERES-EBAF/OBS_CERES-EBAF_sat_Ed4.2_Amon_rlut_200003-202311.nc",
        # two-file OBS dataset (same slug -> one dataset, two files)
        "OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_197901-197912.nc",
        "OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_198001-198012.nc",
        # two-file native6 dataset
        "native6/Tier3/ERA5/v1/mon/hus/era5_specific_humidity_1980_monthly.nc",
        "native6/Tier3/ERA5/v1/mon/hus/era5_specific_humidity_1981_monthly.nc",
        # single-file obs4MIPs dataset
        "obs4MIPs/GPCP-V2.3/v20180519/pr_GPCP-SG_L3_v2.3_197901-201710.nc",
    ]
    for rel in files:
        _touch(root / rel)
    return root


@pytest.fixture
def db() -> Database:
    config = Config.default()
    database = Database("sqlite:///:memory:")
    database.migrate(config)
    yield database
    database.close()


def test_ingest_end_to_end(reference_tree, db):
    adapter = ESMValToolReferenceDatasetAdapter()

    stats = ingest_datasets(adapter, reference_tree, db, skip_invalid=True)

    # 4 distinct datasets across the three layouts, 6 files total
    assert stats.datasets_created == 4
    assert stats.files_added == 6

    catalog = adapter.load_catalog(db)
    assert set(catalog["instance_id"]) == {
        "esmvaltool-reference.OBS.CERES-EBAF.Amon.rlut.Ed4.2",
        "esmvaltool-reference.OBS.OSI-450-nh.OImon.sic.v3",
        "esmvaltool-reference.native6.ERA5.mon.hus.v1",
        "esmvaltool-reference.obs4MIPs.GPCP-V2.3.mon.pr.v20180519",
    }
    # the OSI-450 and ERA5 datasets each own two files
    counts = catalog["instance_id"].value_counts()
    assert counts["esmvaltool-reference.OBS.OSI-450-nh.OImon.sic.v3"] == 2
    assert counts["esmvaltool-reference.native6.ERA5.mon.hus.v1"] == 2


def test_ingest_is_idempotent(reference_tree, db):
    adapter = ESMValToolReferenceDatasetAdapter()

    ingest_datasets(adapter, reference_tree, db, skip_invalid=True)
    stats = ingest_datasets(adapter, reference_tree, db, skip_invalid=True)

    assert stats.datasets_created == 0
    assert stats.datasets_unchanged == 4
    assert stats.files_added == 0
