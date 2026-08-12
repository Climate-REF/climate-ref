"""
Unit tests for the ENSO driver helpers.

`enso_driver` runs inside the PMP conda environment,
so the xcdat / pcmdi_metrics / EnsoMetrics imports it makes at module scope are stubbed..
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from climate_ref_pmp.pmp_driver import _get_resource

# Modules that only exist inside the PMP conda environment.
_CONDA_ONLY_MODULES = (
    "xcdat",
    "pcmdi_metrics",
    "pcmdi_metrics.enso",
    "pcmdi_metrics.enso.lib",
    "pcmdi_metrics.io",
    "pcmdi_metrics.utils",
    "EnsoMetrics",
    "EnsoMetrics.EnsoCollectionsLib",
    "EnsoMetrics.EnsoComputeMetricsLib",
    "EnsoPlots",
    "EnsoPlots.EnsoMetricPlot",
)


@pytest.fixture(scope="module")
def enso_driver():
    """Import the driver script with its conda-only dependencies stubbed."""
    with patch.dict(sys.modules, {name: MagicMock() for name in _CONDA_ONLY_MODULES}):
        path = Path(_get_resource("climate_ref_pmp.drivers", "enso_driver.py", use_resources=True))
        spec = importlib.util.spec_from_file_location("enso_driver_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # xcdat wraps xarray, so the reader and writer used here are the xarray ones.
        module.xc = xr
        yield module


@pytest.fixture
def landmask_calls(enso_driver, monkeypatch):
    """Record the file each land-sea mask is derived from, without touching create_land_sea_mask."""
    calls = []

    def _fake(file_path, var_name, output_dir=".", output_filename=None):
        calls.append((file_path, var_name))
        return f"{output_dir}/landmask_{var_name}.nc"

    monkeypatch.setattr(enso_driver, "generate_landmask_path", _fake)
    return calls


def _write_slice(tmp_path, name, start, periods):
    """Write a small monthly `ts` field so the files can actually be concatenated."""
    time = pd.date_range(start, periods=periods, freq="MS")
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([120.0, 130.0])
    values = np.arange(periods * lat.size * lon.size, dtype=float).reshape(periods, lat.size, lon.size)
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), values)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    path = tmp_path / name
    ds.to_netcdf(path)
    return str(path)


def _dataset_dict(paths, *, data_type="model", dataset="ACCESS-CM2_r1i1p1f1", variable="ts"):
    return {
        data_type: {dataset: {variable: {"path + filename": paths, "varname": variable}}},
        "metricsCollection": "ENSO_proc",
    }


EARLY = "ts_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185001-185012.nc"
LATE = "ts_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_185101-185112.nc"


class TestConcatenateTimeseries:
    def test_files_are_joined_in_time(self, enso_driver, tmp_path):
        first = _write_slice(tmp_path, EARLY, "1850-01-01", 12)
        second = _write_slice(tmp_path, LATE, "1851-01-01", 12)

        joined = enso_driver.concatenate_timeseries([first, second], output_dir=str(tmp_path))

        with xr.open_dataset(joined) as ds:
            assert ds.sizes["time"] == 24
            assert str(ds["time"].values[0])[:7] == "1850-01"
            assert str(ds["time"].values[-1])[:7] == "1851-12"

    def test_joined_values_match_the_inputs(self, enso_driver, tmp_path):
        """The join must not reorder or resample the data."""
        first = _write_slice(tmp_path, EARLY, "1850-01-01", 12)
        second = _write_slice(tmp_path, LATE, "1851-01-01", 12)

        joined = enso_driver.concatenate_timeseries([first, second], output_dir=str(tmp_path))

        with xr.open_dataset(joined) as ds, xr.open_dataset(first) as a, xr.open_dataset(second) as b:
            expected = xr.concat([a["ts"], b["ts"]], dim="time")
            xr.testing.assert_allclose(ds["ts"], expected)

    def test_existing_join_is_reused(self, enso_driver, tmp_path):
        """Reruns and sibling variables must not pay to rewrite the same file."""
        first = _write_slice(tmp_path, EARLY, "1850-01-01", 12)
        second = _write_slice(tmp_path, LATE, "1851-01-01", 12)
        out = tmp_path / "out"

        joined = enso_driver.concatenate_timeseries([first, second], output_dir=str(out))
        Path(joined).write_text("sentinel")
        again = enso_driver.concatenate_timeseries([first, second], output_dir=str(out))

        assert again == joined
        assert Path(again).read_text() == "sentinel"

    def test_overlapping_packagings_are_deduplicated(self, enso_driver, tmp_path):
        """
        One period can be registered twice in different packagings: GPCP-Monthly-3-2 arrives as a
        single multi-year file from the obs4REF cache and as annual files from the ESGF mirror.
        The join must cover the period exactly once and stay monotonic in time.
        """
        whole = _write_slice(tmp_path, "pr_mon_GPCP_gn_185001-185112.nc", "1850-01-01", 24)
        year_one = _write_slice(tmp_path, "pr_mon_GPCP_gn_185001-185012.nc", "1850-01-01", 12)
        year_two = _write_slice(tmp_path, "pr_mon_GPCP_gn_185101-185112.nc", "1851-01-01", 12)

        joined = enso_driver.concatenate_timeseries(
            [whole, year_one, year_two], output_dir=str(tmp_path / "out")
        )

        with xr.open_dataset(joined) as ds:
            assert ds.sizes["time"] == 24
            assert ds.indexes["time"].is_monotonic_increasing
            assert not ds.indexes["time"].has_duplicates


class TestUpdateDictDatasets:
    def test_split_timeseries_is_joined_into_one_file(self, enso_driver, tmp_path, landmask_calls):
        """EnsoMetrics reads one file per variable, so a split series must arrive joined."""
        files = [
            _write_slice(tmp_path, LATE, "1851-01-01", 12),
            _write_slice(tmp_path, EARLY, "1850-01-01", 12),
        ]

        result = enso_driver.update_dict_datasets(_dataset_dict(files), str(tmp_path))

        path = result["model"]["ACCESS-CM2_r1i1p1f1"]["sst"]["path + filename"]
        assert isinstance(path, str)
        assert path not in files
        with xr.open_dataset(path) as ds:
            assert ds.sizes["time"] == 24

    def test_join_is_chronological_regardless_of_input_order(self, enso_driver, tmp_path, landmask_calls):
        """The catalog does not order files, but CMIP filenames sort chronologically."""
        files = [
            _write_slice(tmp_path, LATE, "1851-01-01", 12),
            _write_slice(tmp_path, EARLY, "1850-01-01", 12),
        ]

        result = enso_driver.update_dict_datasets(_dataset_dict(files), str(tmp_path))

        with xr.open_dataset(result["model"]["ACCESS-CM2_r1i1p1f1"]["sst"]["path + filename"]) as ds:
            assert str(ds["time"].values[0])[:7] == "1850-01"

    def test_landmask_is_derived_from_the_joined_file(self, enso_driver, tmp_path, landmask_calls):
        files = [
            _write_slice(tmp_path, LATE, "1851-01-01", 12),
            _write_slice(tmp_path, EARLY, "1850-01-01", 12),
        ]

        enso_driver.update_dict_datasets(_dataset_dict(files), str(tmp_path))

        assert len(landmask_calls) == 1
        assert landmask_calls[0][0].endswith("_concatenated.nc")

    def test_single_file_list_is_unwrapped_and_not_joined(self, enso_driver, tmp_path, landmask_calls):
        """A lone file is passed straight through, as it always has been."""
        only = _write_slice(tmp_path, EARLY, "1850-01-01", 12)

        result = enso_driver.update_dict_datasets(_dataset_dict([only]), str(tmp_path))

        assert result["model"]["ACCESS-CM2_r1i1p1f1"]["sst"]["path + filename"] == only

    def test_plain_string_is_kept(self, enso_driver, tmp_path, landmask_calls):
        only = _write_slice(tmp_path, EARLY, "1850-01-01", 12)

        result = enso_driver.update_dict_datasets(_dataset_dict(only), str(tmp_path))

        assert result["model"]["ACCESS-CM2_r1i1p1f1"]["sst"]["path + filename"] == only

    def test_multi_file_observations_are_renamed(self, enso_driver, tmp_path, landmask_calls):
        """The HadISST-1-1 reference is itself split over files, and the rename must still apply."""
        files = [
            _write_slice(tmp_path, "ts_HadISST-1-1_185101-185112.nc", "1851-01-01", 12),
            _write_slice(tmp_path, "ts_HadISST-1-1_185001-185012.nc", "1850-01-01", 12),
        ]

        result = enso_driver.update_dict_datasets(
            _dataset_dict(files, data_type="observations", dataset="HadISST-1-1"), str(tmp_path)
        )

        assert isinstance(result["observations"]["HadISST"]["sst"]["path + filename"], str)

    def test_duplicate_copies_are_not_concatenated(self, enso_driver, tmp_path, landmask_calls):
        """
        HadISST-1-1 may be ingested from both the obs4REF cache and the ESGF obs4MIPs mirror,
        so the same file arrives twice.
        The copies cover one time range and must not be joined.
        """
        name = "ts_mon_HadISST-1-1_PCMDI_gn_185001-185012.nc"
        cache = tmp_path / "obs4ref"
        mirror = tmp_path / "obs4MIPs"
        cache.mkdir()
        mirror.mkdir()
        copies = [
            _write_slice(cache, name, "1850-01-01", 12),
            _write_slice(mirror, name, "1850-01-01", 12),
        ]

        result = enso_driver.update_dict_datasets(
            _dataset_dict(copies, data_type="observations", dataset="HadISST-1-1"), str(tmp_path)
        )

        path = result["observations"]["HadISST"]["sst"]["path + filename"]
        assert path in copies  # one copy passed straight through, not a joined file
        with xr.open_dataset(path) as ds:
            assert ds.sizes["time"] == 12

    def test_duplicates_are_removed_before_joining_real_slices(self, enso_driver, tmp_path, landmask_calls):
        """A mirrored copy alongside genuine time slices must not disturb the join."""
        mirror = tmp_path / "mirror"
        mirror.mkdir()
        early = _write_slice(tmp_path, EARLY, "1850-01-01", 12)
        late = _write_slice(tmp_path, LATE, "1851-01-01", 12)
        duplicate_of_early = _write_slice(mirror, EARLY, "1850-01-01", 12)

        result = enso_driver.update_dict_datasets(
            _dataset_dict([late, duplicate_of_early, early]), str(tmp_path)
        )

        with xr.open_dataset(result["model"]["ACCESS-CM2_r1i1p1f1"]["sst"]["path + filename"]) as ds:
            assert ds.sizes["time"] == 24

    def test_missing_file_in_a_list_is_reported(self, enso_driver, tmp_path, landmask_calls):
        present = _write_slice(tmp_path, EARLY, "1850-01-01", 12)
        missing = str(tmp_path / "ts_Amon_ACCESS-CM2_historical_r1i1p1f1_gn_999901-999912.nc")

        with pytest.raises(FileNotFoundError, match="999901-999912"):
            enso_driver.update_dict_datasets(_dataset_dict([present, missing]), str(tmp_path))

    def test_empty_list_is_rejected(self, enso_driver, tmp_path, landmask_calls):
        with pytest.raises(ValueError, match="No paths found for model"):
            enso_driver.update_dict_datasets(_dataset_dict([]), str(tmp_path))

    def test_unsupported_path_type_is_rejected(self, enso_driver, tmp_path, landmask_calls):
        with pytest.raises(NotImplementedError, match="not a string or list of strings"):
            enso_driver.update_dict_datasets(_dataset_dict(42), str(tmp_path))
