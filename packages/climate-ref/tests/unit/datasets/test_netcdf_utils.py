"""Tests for netcdf_utils module."""

import netCDF4
import numpy as np
import pytest
import xarray as xr

from climate_ref.datasets.netcdf_utils import (
    read_global_attrs,
    read_time_bounds,
    read_variable_attrs,
    read_vertical_levels,
)


@pytest.fixture
def sample_nc_file(tmp_path):
    """Create a sample netCDF file with time, vertical levels, and variable attrs."""
    filepath = tmp_path / "test.nc"
    with netCDF4.Dataset(filepath, "w") as ds:
        # Global attributes
        ds.activity_id = "CMIP"
        ds.variable_id = "tas"
        ds.frequency = "mon"
        ds.source_id = "ACCESS-ESM1-5"

        # Dimensions
        ds.createDimension("time", 12)
        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)
        ds.createDimension("plev", 5)

        # Time variable with CF-compliant attributes
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1850-01-01"
        time_var.calendar = "standard"
        time_var[:] = np.arange(15, 360, 30)  # mid-month values

        # Data variable with metadata
        tas = ds.createVariable("tas", "f4", ("time", "lat", "lon"))
        tas.standard_name = "air_temperature"
        tas.long_name = "Near-Surface Air Temperature"
        tas.units = "K"
        tas[:] = np.random.default_rng(42).random((12, 10, 20))

    return str(filepath)


@pytest.fixture
def no_time_nc_file(tmp_path):
    """Create a netCDF file without a time dimension (fx frequency)."""
    filepath = tmp_path / "no_time.nc"
    with netCDF4.Dataset(filepath, "w") as ds:
        ds.activity_id = "CMIP"
        ds.variable_id = "areacella"
        ds.frequency = "fx"

        ds.createDimension("lat", 10)
        ds.createDimension("lon", 20)

        areacella = ds.createVariable("areacella", "f4", ("lat", "lon"))
        areacella.standard_name = "cell_area"
        areacella.long_name = "Grid-Cell Area"
        areacella.units = "m2"
        areacella[:] = np.ones((10, 20))

    return str(filepath)


@pytest.fixture
def empty_time_nc_file(tmp_path):
    """Create a netCDF file with a zero-length time dimension."""
    filepath = tmp_path / "empty_time.nc"
    with netCDF4.Dataset(filepath, "w") as ds:
        ds.activity_id = "CMIP"
        ds.variable_id = "tas"

        ds.createDimension("time", 0)
        ds.createDimension("lat", 5)
        ds.createDimension("lon", 5)

        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1850-01-01"
        time_var.calendar = "standard"

    return str(filepath)


@pytest.fixture
def no_time_units_nc_file(tmp_path):
    """Create a netCDF file with a time variable but no units attribute."""
    filepath = tmp_path / "no_units.nc"
    with netCDF4.Dataset(filepath, "w") as ds:
        ds.activity_id = "CMIP"
        ds.variable_id = "tas"

        ds.createDimension("time", 3)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var[:] = [0, 1, 2]

    return str(filepath)


class TestReadGlobalAttrs:
    def test_reads_existing_attrs(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_global_attrs(ds, ["activity_id", "variable_id", "frequency"])

        assert result == {
            "activity_id": "CMIP",
            "variable_id": "tas",
            "frequency": "mon",
        }

    def test_missing_attrs_return_none(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_global_attrs(ds, ["nonexistent_attr", "also_missing"])

        assert result == {"nonexistent_attr": None, "also_missing": None}

    def test_mixed_existing_and_missing(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_global_attrs(ds, ["activity_id", "nonexistent"])

        assert result["activity_id"] == "CMIP"
        assert result["nonexistent"] is None

    def test_empty_keys_list(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_global_attrs(ds, [])

        assert result == {}


class TestReadVariableAttrs:
    def test_reads_variable_attrs(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_variable_attrs(ds, "tas", ["standard_name", "long_name", "units"])

        assert result == {
            "standard_name": "air_temperature",
            "long_name": "Near-Surface Air Temperature",
            "units": "K",
        }

    def test_missing_variable_returns_none_values(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_variable_attrs(ds, "nonexistent_var", ["standard_name", "units"])

        assert result == {"standard_name": None, "units": None}

    def test_empty_variable_id_returns_none_values(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_variable_attrs(ds, "", ["standard_name"])

        assert result == {"standard_name": None}

    def test_missing_attr_on_existing_variable(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_variable_attrs(ds, "tas", ["standard_name", "nonexistent_attr"])

        assert result["standard_name"] == "air_temperature"
        assert result["nonexistent_attr"] is None


class TestReadVerticalLevels:
    def test_finds_plev(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            result = read_vertical_levels(ds)

        assert result == 5

    def test_no_vertical_dim_returns_one(self, no_time_nc_file):
        with netCDF4.Dataset(no_time_nc_file, "r") as ds:
            result = read_vertical_levels(ds)

        assert result == 1

    def test_finds_first_matching_dim(self, tmp_path):
        """When multiple vertical dim names exist, returns the first match."""
        filepath = tmp_path / "multi_vert.nc"
        with netCDF4.Dataset(filepath, "w") as ds:
            # 'lev' comes before 'plev' in VERTICAL_DIM_NAMES
            ds.createDimension("lev", 3)
            ds.createDimension("plev", 7)

        with netCDF4.Dataset(filepath, "r") as ds:
            result = read_vertical_levels(ds)

        assert result == 3


class TestReadTimeBounds:
    def test_reads_time_bounds(self, sample_nc_file):
        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            start, end = read_time_bounds(ds)

        assert start is not None
        assert end is not None
        assert "1850" in start
        assert start != end

    def test_no_time_variable(self, no_time_nc_file):
        with netCDF4.Dataset(no_time_nc_file, "r") as ds:
            start, end = read_time_bounds(ds)

        assert start is None
        assert end is None

    def test_zero_length_time(self, empty_time_nc_file):
        with netCDF4.Dataset(empty_time_nc_file, "r") as ds:
            start, end = read_time_bounds(ds)

        assert start is None
        assert end is None

    def test_no_time_units(self, no_time_units_nc_file):
        with netCDF4.Dataset(no_time_units_nc_file, "r") as ds:
            start, end = read_time_bounds(ds)

        assert start is None
        assert end is None

    def test_single_timestep(self, tmp_path):
        """File with exactly one time step should return same start and end."""
        filepath = tmp_path / "single_time.nc"
        with netCDF4.Dataset(filepath, "w") as ds:
            ds.createDimension("time", 1)
            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.units = "days since 1850-01-01"
            time_var.calendar = "standard"
            time_var[:] = [15.0]

        with netCDF4.Dataset(filepath, "r") as ds:
            start, end = read_time_bounds(ds)

        assert start is not None
        assert start == end

    def test_matches_xarray_output(self, sample_nc_file):
        """Verify netCDF4+cftime produces the same time strings as xarray."""
        with xr.open_dataset(sample_nc_file, use_cftime=True) as xr_ds:
            xr_start = str(xr_ds["time"].values[0])
            xr_end = str(xr_ds["time"].values[-1])

        with netCDF4.Dataset(sample_nc_file, "r") as ds:
            nc_start, nc_end = read_time_bounds(ds)

        assert nc_start == xr_start, f"Start mismatch: netCDF4={nc_start!r}, xarray={xr_start!r}"
        assert nc_end == xr_end, f"End mismatch: netCDF4={nc_end!r}, xarray={xr_end!r}"

    def test_matches_xarray_with_noleap_calendar(self, tmp_path):
        """Verify matching output with a non-standard calendar."""
        filepath = tmp_path / "noleap.nc"
        with netCDF4.Dataset(filepath, "w") as ds:
            ds.createDimension("time", 3)
            time_var = ds.createVariable("time", "f8", ("time",))
            time_var.units = "days since 0001-01-01"
            time_var.calendar = "noleap"
            time_var[:] = [0.5, 365.5, 730.5]

        with xr.open_dataset(filepath, use_cftime=True) as xr_ds:
            xr_start = str(xr_ds["time"].values[0])
            xr_end = str(xr_ds["time"].values[-1])

        with netCDF4.Dataset(filepath, "r") as ds:
            nc_start, nc_end = read_time_bounds(ds)

        assert nc_start == xr_start, f"Start mismatch: netCDF4={nc_start!r}, xarray={xr_start!r}"
        assert nc_end == xr_end, f"End mismatch: netCDF4={nc_end!r}, xarray={xr_end!r}"
