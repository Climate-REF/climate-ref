"""Tests for the obs4REF adapter (A3): warn-only cross-adapter enforcement.

``Obs4REFDatasetAdapter`` and ``Obs4MIPsDatasetAdapter`` share the same
:func:`~climate_ref.datasets.obs4mips.parse_obs4mips` parser, but each restricts its own
``accepted_activity_ids``. A file whose ``activity_id`` doesn't match the adapter that
parses it is still ingested (warn-only, not rejected) -- see ``obs4mips.py`` docstrings.
"""

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
    """A single obs4REF-DRS-shaped file with ``activity_id="obs4REF"``."""
    fixture_dir = tmp_path / "obs4ref_style"
    _write_obs4_style_file(
        fixture_dir / "obs4REF" / "TESTORG" / "TEST-SRC" / "mon" / "ts" / "gn" / "v1" / "ts_mon.nc",
        activity_id="obs4REF",
    )
    return fixture_dir


class TestObs4REFDatasetAdapter:
    def test_instance_id_prefix(self):
        assert Obs4REFDatasetAdapter.instance_id_prefix == "obs4REF"
        assert Obs4REFDatasetAdapter.accepted_activity_ids == ("obs4REF",)

    def test_load_local_datasets_prefixes_instance_id(self, obs4ref_style_dir):
        adapter = Obs4REFDatasetAdapter()
        data_catalog = adapter.find_local_datasets(obs4ref_style_dir)

        assert len(data_catalog) == 1
        assert data_catalog["instance_id"].iloc[0].startswith("obs4REF.")

    def test_cross_parsed_by_obs4mips_adapter_warns_and_ingests(self, obs4ref_style_dir, caplog):
        """An obs4REF file parsed by the obs4MIPs adapter is still ingested, with a warning."""
        obs4mips_adapter = Obs4MIPsDatasetAdapter()
        ref_adapter = Obs4REFDatasetAdapter()

        obs4mips_catalog = obs4mips_adapter.find_local_datasets(obs4ref_style_dir)
        ref_catalog = ref_adapter.find_local_datasets(obs4ref_style_dir)

        assert len(obs4mips_catalog) == 1
        obs4mips_instance_id = obs4mips_catalog["instance_id"].iloc[0]
        ref_instance_id = ref_catalog["instance_id"].iloc[0]

        assert obs4mips_instance_id.startswith("obs4MIPs.")
        assert ref_instance_id.startswith("obs4REF.")
        # Non-colliding slugs -- the cross-parsed file never masquerades as the same dataset.
        assert obs4mips_instance_id != ref_instance_id

        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("outside the expected" in msg for msg in warning_messages)
