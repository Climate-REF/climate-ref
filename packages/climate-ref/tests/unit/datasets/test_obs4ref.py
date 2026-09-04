"""Tests for the obs4REF adapter."""

import netCDF4
import numpy as np
import pytest

from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter, Obs4REFDatasetAdapter


def _write_obs4_style_file(path, *, activity_id: str) -> None:
    """Write a minimal netCDF file with the global/variable attrs ``parse_obs4mips`` needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.activity_id = activity_id
        ds.frequency = "mon"
        ds.grid = "native"
        ds.grid_label = "gn"
        ds.institution_id = "TESTORG"
        ds.nominal_resolution = "100 km"
        ds.realm = "atmos"
        ds.product = "observations"
        ds.source_id = "TEST-SRC"
        ds.source_type = "satellite"
        ds.variable_id = "ts"
        ds.variant_label = "v1"
        ds.source_version_number = "1"

        ds.createDimension("time", 3)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1850-01-01"
        time_var.calendar = "standard"
        time_var[:] = [0, 30, 60]

        ts_var = ds.createVariable("ts", "f4", ("time",))
        ts_var.units = "K"
        ts_var.long_name = "Surface Temperature"
        ts_var[:] = np.array([280.0, 281.0, 282.0])


@pytest.fixture
def obs4ref_style_dir(tmp_path):
    """An obs4REF-registry-shaped file that still claims ``activity_id="obs4MIPs"`` inside."""
    fixture_dir = tmp_path / "obs4ref_style"
    _write_obs4_style_file(
        fixture_dir / "obs4REF" / "TESTORG" / "TEST-SRC" / "mon" / "ts" / "gn" / "v1" / "ts_mon.nc",
        activity_id="obs4MIPs",
    )
    return fixture_dir


class TestObs4REFDatasetAdapter:
    def test_activity_id(self):
        assert Obs4REFDatasetAdapter.activity_id == "obs4REF"
        assert Obs4MIPsDatasetAdapter.activity_id == "obs4MIPs"

    def test_stamps_collection_regardless_of_file(self, obs4ref_style_dir):
        """The registry republishes obs4MIPs files unchanged, so the adapter decides the collection."""
        data_catalog = Obs4REFDatasetAdapter().find_local_datasets(obs4ref_style_dir)

        assert len(data_catalog) == 1
        assert data_catalog["activity_id"].iloc[0] == "obs4REF"
        assert data_catalog["instance_id"].iloc[0] == "obs4REF.obs4REF.TESTORG.TEST-SRC.mon.ts.100km.gn.v1"

    def test_same_file_gets_distinct_ids(self, obs4ref_style_dir):
        obs4mips_catalog = Obs4MIPsDatasetAdapter().find_local_datasets(obs4ref_style_dir)
        ref_catalog = Obs4REFDatasetAdapter().find_local_datasets(obs4ref_style_dir)

        obs4mips_instance_id = obs4mips_catalog["instance_id"].iloc[0]
        ref_instance_id = ref_catalog["instance_id"].iloc[0]
        assert obs4mips_instance_id.startswith("obs4MIPs.obs4MIPs.")
        assert ref_instance_id.startswith("obs4REF.obs4REF.")
        assert obs4mips_instance_id.split(".", 2)[2] == ref_instance_id.split(".", 2)[2]

    def test_obs4mips_adapter_warns_on_obs4ref_layout(self, obs4ref_style_dir, caplog):
        Obs4MIPsDatasetAdapter().find_local_datasets(obs4ref_style_dir)

        warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("look like obs4REF data" in msg and "--source-type obs4ref" in msg for msg in warnings)

    def test_obs4ref_adapter_does_not_warn(self, obs4ref_style_dir, caplog):
        Obs4REFDatasetAdapter().find_local_datasets(obs4ref_style_dir)

        assert not [r for r in caplog.records if r.levelname == "WARNING"]
