"""Tests for climate_ref_core.esgf.cmip7 module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cftime
import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_ref_core.cmip6_to_cmip7 import create_cmip7_filename, create_cmip7_path
from climate_ref_core.esgf import CMIP7Request
from climate_ref_core.esgf.cmip7 import (
    _MANDATORY_CONVERTED_ATTRS,
    _bump_version,
    _convert_file_to_cmip7,
    _get_cmip7_cache_dir,
    _invalid_conversion_reason,
    _latest_file,
)


class TestCMIP7Request:
    """Tests for CMIP7Request class."""

    def test_init_basic(self):
        """Test basic initialization."""
        request = CMIP7Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        assert request.slug == "test-request"
        assert request.facets == {"source_id": "ACCESS-ESM1-5", "variable_id": "tas"}
        assert request.remove_ensembles is False
        assert request.time_span is None
        assert request.source_type == "CMIP7"

    def test_init_with_time_span(self):
        """Test initialization with time span."""
        request = CMIP7Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
            time_span=("2000-01", "2010-12"),
        )
        assert request.time_span == ("2000-01", "2010-12")

    def test_init_with_remove_ensembles(self):
        """Test initialization with remove_ensembles flag."""
        request = CMIP7Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
            remove_ensembles=True,
        )
        assert request.remove_ensembles is True

    def test_repr(self):
        """Test string representation."""
        request = CMIP7Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
        )
        repr_str = repr(request)
        assert "CMIP7Request" in repr_str
        assert "test-request" in repr_str
        assert "ACCESS-ESM1-5" in repr_str

    def test_available_facets(self):
        """Test that available facets are defined."""
        assert "activity_id" in CMIP7Request.available_facets
        assert "source_id" in CMIP7Request.available_facets
        assert "variable_id" in CMIP7Request.available_facets
        assert "variant_label" in CMIP7Request.available_facets  # CMIP7 name
        assert "frequency" in CMIP7Request.available_facets

    def test_facet_mapping(self):
        """Test CMIP7 to CMIP6 facet mapping."""
        assert CMIP7Request.facet_mapping["variant_label"] == "member_id"

    def test_convert_to_cmip6_facets(self):
        """Test conversion of CMIP7 facets to CMIP6."""
        request = CMIP7Request(
            slug="test",
            facets={
                "source_id": "ACCESS-ESM1-5",
                "variant_label": "r1i1p1f1",
                "variable_id": "tas",
            },
        )
        # Access private method for testing
        cmip6_facets = request._cmip6_facets
        assert cmip6_facets["source_id"] == "ACCESS-ESM1-5"
        assert cmip6_facets["member_id"] == "r1i1p1f1"  # Mapped from variant_label
        assert cmip6_facets["variable_id"] == "tas"
        assert "variant_label" not in cmip6_facets

    def test_convert_to_cmip7_metadata(self):
        """Test conversion of CMIP6 metadata to CMIP7 format."""
        request = CMIP7Request(slug="test", facets={})
        cmip6_row = {
            "source_id": "ACCESS-ESM1-5",
            "member_id": "r1i1p1f1",
            "variable_id": "tas",
            "table_id": "Amon",
        }
        cmip7_row = request._convert_to_cmip7_metadata(cmip6_row)

        assert cmip7_row["source_id"] == "ACCESS-ESM1-5"
        assert cmip7_row["variant_label"] == "r1i1p1f1"  # Renamed from member_id
        assert "member_id" not in cmip7_row
        assert cmip7_row["mip_era"] == "CMIP7"
        assert cmip7_row["frequency"] == "mon"  # Derived from table_id

    @pytest.mark.parametrize(
        "table_id,expected_frequency",
        [
            ("Amon", "mon"),
            ("day", "day"),
            ("fx", "fx"),
            ("Oyr", "yr"),
            ("Omon", "mon"),
            ("unknown", "mon"),  # Default
        ],
    )
    def test_table_id_to_frequency_mapping(self, table_id, expected_frequency):
        """Test table_id to frequency mapping."""
        request = CMIP7Request(slug="test", facets={})
        cmip6_row = {"table_id": table_id}
        cmip7_row = request._convert_to_cmip7_metadata(cmip6_row)
        assert cmip7_row["frequency"] == expected_frequency

    def test_convert_to_cmip7_metadata_preserves_existing_frequency(self):
        """Test that existing frequency is not overwritten."""
        request = CMIP7Request(slug="test", facets={})
        cmip6_row = {"table_id": "Amon", "frequency": "day"}
        cmip7_row = request._convert_to_cmip7_metadata(cmip6_row)
        assert cmip7_row["frequency"] == "day"  # Preserved, not overwritten

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    def test_fetch_datasets_empty(self, mock_cmip6_request_class):
        """Test fetch_datasets when CMIP6 returns empty."""
        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame()
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        result = request.fetch_datasets()

        assert result.empty

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_fetch_datasets_converts_files(self, mock_convert, mock_cmip6_request_class, tmp_path):
        """Test fetch_datasets converts CMIP6 files to CMIP7."""
        # Create a mock CMIP6 file
        cmip6_file = tmp_path / "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200012.nc"
        cmip6_file.touch()

        # Set up mock CMIP6 request
        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5"],
                "member_id": ["r1i1p1f1"],
                "variable_id": ["tas"],
                "table_id": ["Amon"],
                "files": [[str(cmip6_file)]],
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        # Set up mock conversion
        cmip7_file = tmp_path / "cmip7" / "tas.nc"
        cmip7_file.parent.mkdir(parents=True, exist_ok=True)
        cmip7_file.touch()
        mock_convert.return_value = cmip7_file

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        result = request.fetch_datasets()

        assert not result.empty
        assert len(result) == 1
        assert result.iloc[0]["variant_label"] == "r1i1p1f1"
        assert result.iloc[0]["mip_era"] == "CMIP7"
        assert str(cmip7_file) in result.iloc[0]["files"]
        mock_convert.assert_called_once()

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    def test_fetch_datasets_missing_file(self, mock_cmip6_request_class, caplog):
        """Test fetch_datasets handles missing files gracefully."""
        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5"],
                "member_id": ["r1i1p1f1"],
                "variable_id": ["tas"],
                "table_id": ["Amon"],
                "files": [["/nonexistent/file.nc"]],
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        result = request.fetch_datasets()

        assert result.empty


class TestGetCmip7CacheDir:
    """Tests for _get_cmip7_cache_dir function."""

    def test_returns_path(self, tmp_path, monkeypatch):
        """Test that function returns a Path object."""
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
        result = _get_cmip7_cache_dir()
        assert isinstance(result, Path)

    def test_honours_dataset_cache_dir(self, tmp_path, monkeypatch):
        """Converted files land under the configured dataset cache, not the OS default."""
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
        result = _get_cmip7_cache_dir()
        assert result == tmp_path / "cmip7-converted"
        assert result.is_dir()

    def test_falls_back_to_os_cache(self, monkeypatch):
        """With no override the path stays under the platform cache for climate_ref."""
        monkeypatch.delenv("REF_DATASET_CACHE_DIR", raising=False)
        result = _get_cmip7_cache_dir()
        assert "climate_ref" in str(result)
        assert "cmip7-converted" in str(result)


class TestConvertedCacheIntegrity:
    """Exercise conversion-cache checks against real NetCDF files."""

    def _write_file(
        self,
        path: Path,
        *,
        missing_attr: str | None = None,
        variable_id: str = "cli",
        data_variable: str = "cli",
        time_units: str | None = "days since 2000-01-01",
        values: np.ndarray | None = None,
        least_significant_digit: int | None = None,
    ) -> None:
        attrs = {name: name for name in _MANDATORY_CONVERTED_ATTRS}
        attrs["variable_id"] = variable_id
        if missing_attr is not None:
            attrs.pop(missing_attr)
        time_attrs = {} if time_units is None else {"units": time_units, "calendar": "noleap"}
        data = np.ones((2, 1)) if values is None else values
        ds = xr.Dataset(
            {data_variable: (("time", "lat"), data)},
            coords={
                "time": xr.DataArray([0.0, 31.0], dims="time", attrs=time_attrs),
                "lat": [0.0],
            },
            attrs=attrs,
        )
        encoding = (
            {data_variable: {"least_significant_digit": least_significant_digit}}
            if least_significant_digit is not None
            else None
        )
        ds.to_netcdf(path, encoding=encoding)

    def test_accepts_valid_legacy_file_without_marker(self, tmp_path):
        path = tmp_path / "legacy.nc"
        self._write_file(path)

        assert _invalid_conversion_reason(path) is None

    def test_reports_missing_mandatory_attribute(self, tmp_path):
        path = tmp_path / "missing-attribute.nc"
        self._write_file(path, missing_attr="branding_suffix")

        assert _invalid_conversion_reason(path) == "missing mandatory attributes: branding_suffix"

    def test_reports_missing_declared_variable(self, tmp_path):
        path = tmp_path / "missing-variable.nc"
        self._write_file(path, variable_id="msftm")

        assert _invalid_conversion_reason(path) == "missing data variable: msftm"

    def test_reports_missing_time_units(self, tmp_path):
        path = tmp_path / "missing-time-units.nc"
        self._write_file(path, time_units=None)

        assert _invalid_conversion_reason(path) == "time coordinate has no units"

    def test_reports_unreadable_file(self, tmp_path):
        path = tmp_path / "truncated.nc"
        path.write_bytes(b"not a netcdf file")

        reason = _invalid_conversion_reason(path)

        assert reason is not None
        assert reason.startswith("cannot read metadata:")

    def test_rejects_destructively_quantized_ozone_cache(self, tmp_path):
        path = tmp_path / "quantized-o3.nc"
        self._write_file(path, variable_id="o3", data_variable="o3", least_significant_digit=3)

        assert _invalid_conversion_reason(path) == ("ozone dataset uses destructive decimal quantisation: o3")

    def test_rejects_ozone_cache_with_quantized_formula_term(self, tmp_path):
        path = tmp_path / "quantized-o3-coefficients.nc"
        self._write_file(path, variable_id="o3", data_variable="o3")
        with xr.open_dataset(path) as original:
            ds = original.load()
        ds["b"] = ("lev", np.array([0.0011, 0.0012]))
        ds.to_netcdf(path, encoding={"b": {"least_significant_digit": 3}})

        assert _invalid_conversion_reason(path) == ("ozone dataset uses destructive decimal quantisation: b")

    def test_accepts_scientifically_valid_zero_ozone_cache(self, tmp_path):
        path = tmp_path / "zero-o3.nc"
        self._write_file(path, variable_id="o3", data_variable="o3", values=np.zeros((2, 1)))

        assert _invalid_conversion_reason(path) is None


class TestConvertFileToCmip7:
    """Tests for _convert_file_to_cmip7 function."""

    def test_chunked_conversion_round_trips_real_netcdf(self, tmp_path, monkeypatch):
        """The bounded Dask write preserves data and publishes a readable NetCDF file."""
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path / "cache"))
        source = tmp_path / "o3.nc"
        values = np.linspace(1e-9, 16e-8, 16, dtype=np.float32).reshape(2, 2, 2, 2)
        hybrid_b = np.array([0.0011, 0.0012], dtype=np.float64)
        hybrid_ap = np.zeros(2, dtype=np.float64)
        times = [cftime.DatetimeNoLeap(2000, 1, 16), cftime.DatetimeNoLeap(2000, 2, 16)]
        time_bounds = np.array(
            [
                [cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 2, 1)],
                [cftime.DatetimeNoLeap(2000, 2, 1), cftime.DatetimeNoLeap(2000, 3, 1)],
            ]
        )
        xr.Dataset(
            {
                "o3": (("time", "lev", "lat", "lon"), values),
                "time_bnds": (("time", "bnds"), time_bounds),
                "ap": ("lev", hybrid_ap),
                "b": ("lev", hybrid_b),
            },
            coords={
                "time": xr.DataArray(times, dims="time", attrs={"bounds": "time_bnds"}),
                "lat": [-45.0, 45.0],
                "lon": [0.0, 180.0],
                "lev": [1, 2],
            },
            attrs={
                "table_id": "AERmon",
                "variable_id": "o3",
                "activity_id": "CMIP",
                "institution_id": "CSIRO",
                "source_id": "ACCESS-ESM1-5",
                "experiment_id": "historical",
                "variant_label": "r1i1p1f1",
                "grid_label": "gn",
            },
        ).to_netcdf(source)
        facets = {
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "o3",
            "table_id": "AERmon",
            "grid_label": "gn",
            "version": "v1",
            "branding_suffix": "tavg-al-hxy-u",
            "region": "glb",
        }

        output = _convert_file_to_cmip7(source, facets)

        with xr.open_dataset(output) as converted:
            np.testing.assert_array_equal(converted["o3"].values, values)
            np.testing.assert_array_equal(converted["b"].values, hybrid_b)
            pressure = converted["ap"].values + converted["b"].values * 100_000.0
            assert np.all(np.diff(pressure) > 0)
            np.testing.assert_array_equal(converted["time_bnds"].values, time_bounds)
            assert converted.attrs["mip_era"] == "CMIP7"

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_uses_cached_file(self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path):
        """Test that cached files are reused without converting."""
        # Set up cache directory with existing file
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Set up mock for open_dataset context manager
        mock_ds = MagicMock()
        mock_ds.attrs = {name: name for name in _MANDATORY_CONVERTED_ATTRS}
        mock_ds.variables = {"variable_id": MagicMock()}
        mock_ds.attrs["variable_id"] = "variable_id"
        mock_ds.sizes = {}
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)

        # Facets must include DReq-enriched fields (enrichment happens upstream
        # in _convert_to_cmip7_metadata before calling _convert_file_to_cmip7)
        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "table_id": "Amon",
            "grid_label": "gn",
            "version": "v1",
            "branding_suffix": "tavg-h2m-hxy-u",
            "region": "glb",
        }
        # MIP-DRS7 path: drs_specs/mip_era/activity/institution/source/experiment/
        #                variant/region/frequency/variable/branding/grid/version
        drs_path = cache_dir / Path(
            "MIP-DRS7",
            "CMIP7",
            "CMIP",
            "CSIRO",
            "ACCESS-ESM1-5",
            "historical",
            "r1i1p1f1",
            "glb",
            "mon",
            "tas",
            "tavg-h2m-hxy-u",
            "gn",
            "v1",
        )
        drs_path.mkdir(parents=True)

        # Real filename from create_cmip7_filename with DReq-enriched facets
        expected_filename = "tas_tavg-h2m-hxy-u_mon_glb_gn_ACCESS-ESM1-5_historical_r1i1p1f1.nc"
        cached_file = drs_path / expected_filename
        cached_file.touch()

        cmip6_path = tmp_path / "test_input.nc"
        cmip6_path.touch()
        result = _convert_file_to_cmip7(cmip6_path, cmip7_facets)

        assert result == cached_file
        assert mock_open.call_count == 2  # Source time range and cached-file integrity
        mock_convert.assert_not_called()  # Should not convert (cached)

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_converts_new_file(self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path):
        """Test that new files are converted with correct DReq-derived filename."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Set up mocks - open_dataset is used as a context manager
        mock_ds = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_converted_ds = MagicMock()
        mock_converted_ds.attrs = {}
        mock_converted_ds.to_netcdf.side_effect = lambda path, **kwargs: Path(path).touch()

        def convert_with_bounded_scheduler(ds):
            assert dask.config.get("array.chunk-size") == "64MiB"
            assert dask.config.get("scheduler") == "single-threaded"
            return mock_converted_ds

        mock_convert.side_effect = convert_with_bounded_scheduler

        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "table_id": "Amon",
            "grid_label": "gn",
            "version": "v1",
            "branding_suffix": "tavg-h2m-hxy-u",
            "region": "glb",
        }

        # Create input file
        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        result = _convert_file_to_cmip7(input_file, cmip7_facets)

        # Check that conversion happened
        mock_open.assert_called_once()
        assert mock_open.call_args.kwargs["chunks"] == "auto"
        mock_convert.assert_called_once_with(mock_ds)
        mock_converted_ds.to_netcdf.assert_called_once()

        # Check output path structure and filename includes DReq-derived branding
        assert "CMIP" in str(result)
        assert "ACCESS-ESM1-5" in str(result)
        assert result.name == "tas_tavg-h2m-hxy-u_mon_glb_gn_ACCESS-ESM1-5_historical_r1i1p1f1.nc"

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_rebuilds_stale_cached_conversion(
        self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path
    ):
        """A cached file from an older converter cannot bypass regeneration."""
        cache_dir = tmp_path / "cache"
        mock_cache_dir.return_value = cache_dir
        source = MagicMock()
        stale = MagicMock()
        stale.attrs = {}
        mock_open.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=source), __exit__=MagicMock(return_value=False)),
            MagicMock(__enter__=MagicMock(return_value=stale), __exit__=MagicMock(return_value=False)),
        ]
        converted = MagicMock(attrs={})
        converted.to_netcdf.side_effect = lambda path, **kwargs: Path(path).touch()
        mock_convert.return_value = converted
        facets = {
            "activity_id": "CMIP",
            "institution_id": "NCAR",
            "source_id": "CESM2",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "cli",
            "grid_label": "gn",
            "version": "20190308",
            "branding_suffix": "tavg-al-hxy-u",
            "region": "glb",
        }
        expected = (
            cache_dir
            / create_cmip7_path({"drs_specs": "MIP-DRS7", "mip_era": "CMIP7", **facets}, "20190308")
            / create_cmip7_filename(facets)
        )
        expected.parent.mkdir(parents=True)
        expected.touch()

        result = _convert_file_to_cmip7(tmp_path / "cli.nc", facets)

        assert result == expected
        mock_convert.assert_called_once_with(source)

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7._invalid_conversion_reason", return_value="truncated")
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_interrupted_rebuild_preserves_existing_cache(
        self,
        mock_cache_dir,
        mock_convert,
        mock_open,
        mock_invalid_reason,
        mock_time_range,
        tmp_path,
    ):
        """A failed write cannot replace the existing cache entry with a partial file."""
        cache_dir = tmp_path / "cache"
        mock_cache_dir.return_value = cache_dir
        source = MagicMock()
        mock_open.return_value.__enter__.return_value = source
        converted = MagicMock(attrs={})

        def interrupted_write(path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("interrupted")

        converted.to_netcdf.side_effect = interrupted_write
        mock_convert.return_value = converted
        facets = {
            "activity_id": "CMIP",
            "institution_id": "NCAR",
            "source_id": "CESM2",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "cli",
            "grid_label": "gn",
            "version": "20190308",
            "branding_suffix": "tavg-al-hxy-u",
            "region": "glb",
        }
        output = (
            cache_dir
            / create_cmip7_path({"drs_specs": "MIP-DRS7", "mip_era": "CMIP7", **facets}, "20190308")
            / create_cmip7_filename(facets)
        )
        output.parent.mkdir(parents=True)
        output.write_bytes(b"previous")

        with pytest.raises(OSError, match="interrupted"):
            _convert_file_to_cmip7(tmp_path / "cli.nc", facets)

        assert output.read_bytes() == b"previous"
        assert list(output.parent.glob("*.tmp.nc")) == []

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_handles_integer_facet_values(
        self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path
    ):
        """Test that integer facet values are converted to strings in DRS path."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Set up mocks
        mock_ds = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_converted_ds = MagicMock()
        mock_converted_ds.attrs = {}
        mock_converted_ds.to_netcdf.side_effect = lambda path, **kwargs: Path(path).touch()
        mock_convert.return_value = mock_converted_ds

        # Use integer values for some facets; include all required fields
        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": 123,  # Integer value
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "table_id": "Amon",
            "grid_label": "gn",
            "version": 1,  # Integer value
            "branding_suffix": "tavg-h2m-hxy-u",
            "region": "glb",
        }

        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        # Should not raise an error
        result = _convert_file_to_cmip7(input_file, cmip7_facets)
        assert isinstance(result, Path)

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_handles_permission_error_with_existing_file(
        self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path
    ):
        """Test that permission errors are handled when file already exists."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Set up mocks
        mock_ds = MagicMock()
        mock_ds.attrs = {name: name for name in _MANDATORY_CONVERTED_ATTRS}
        mock_ds.variables = {"variable_id": MagicMock()}
        mock_ds.attrs["variable_id"] = "variable_id"
        mock_ds.sizes = {}
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_converted_ds = MagicMock()
        mock_converted_ds.to_netcdf.side_effect = PermissionError("Permission denied")
        mock_convert.return_value = mock_converted_ds

        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "table_id": "Amon",
            "grid_label": "gn",
            "version": "v1",
            "branding_suffix": "tavg-h2m-hxy-u",
            "region": "glb",
        }

        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        # Pre-create the output directory and file (simulating race condition)
        cmip7_fn = "tas_tavg-h2m-hxy-u_mon_glb_gn_ACCESS-ESM1-5_historical_r1i1p1f1.nc"
        drs_path = cache_dir / Path(
            "MIP-DRS7",
            "CMIP7",
            "CMIP",
            "CSIRO",
            "ACCESS-ESM1-5",
            "historical",
            "r1i1p1f1",
            "glb",
            "mon",
            "tas",
            "tavg-h2m-hxy-u",
            "gn",
            "v1",
        )
        drs_path.mkdir(parents=True)
        existing_file = drs_path / cmip7_fn
        existing_file.touch()

        # Should not raise, should return existing file
        result = _convert_file_to_cmip7(input_file, cmip7_facets)
        assert result == existing_file

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_handles_permission_error_without_existing_file(
        self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path
    ):
        """Test that permission errors are re-raised when file does not exist."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        mock_ds = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_converted_ds = MagicMock()
        mock_converted_ds.to_netcdf.side_effect = PermissionError("Permission denied")
        mock_convert.return_value = mock_converted_ds

        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "table_id": "Amon",
            "grid_label": "gn",
            "version": "v1",
            "branding_suffix": "tavg-h2m-hxy-u",
            "region": "glb",
        }

        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        # No pre-existing output file, so PermissionError should be re-raised
        with pytest.raises(PermissionError, match="Permission denied"):
            _convert_file_to_cmip7(input_file, cmip7_facets)

    @patch("climate_ref_core.esgf.cmip7.format_cmip7_time_range", return_value=None)
    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_raises_if_missing(self, mock_cache_dir, mock_convert, mock_open, mock_time_range, tmp_path):
        """Test that empty facets raises KeyError from create_cmip7_path."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        mock_ds = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_converted_ds = MagicMock()
        mock_convert.return_value = mock_converted_ds

        # Empty facets - create_cmip7_path will fail on missing required keys
        cmip7_facets: dict[str, str] = {}

        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        with pytest.raises(KeyError):
            _convert_file_to_cmip7(input_file, cmip7_facets)


class TestCMIP7RequestMetadataEdgeCases:
    """Tests for edge cases in CMIP7 metadata conversion."""

    def test_convert_to_cmip7_metadata_no_member_id(self):
        """Test conversion when member_id is not in the row."""
        request = CMIP7Request(slug="test", facets={})
        cmip6_row = {
            "source_id": "ACCESS-ESM1-5",
            "variable_id": "tas",
        }
        cmip7_row = request._convert_to_cmip7_metadata(cmip6_row)

        assert cmip7_row["source_id"] == "ACCESS-ESM1-5"
        assert "member_id" not in cmip7_row
        assert "variant_label" not in cmip7_row  # No member_id to map
        assert cmip7_row["mip_era"] == "CMIP7"

    def test_convert_to_cmip7_metadata_no_table_id_no_frequency(self):
        """Test conversion when neither table_id nor frequency is present."""
        request = CMIP7Request(slug="test", facets={})
        cmip6_row = {
            "source_id": "ACCESS-ESM1-5",
            "variable_id": "tas",
        }
        cmip7_row = request._convert_to_cmip7_metadata(cmip6_row)

        # No frequency should be added since there's no table_id
        assert "frequency" not in cmip7_row

    def test_convert_to_cmip6_facets_unmapped_keys(self):
        """Test that unmapped facet keys pass through unchanged."""
        request = CMIP7Request(
            slug="test",
            facets={
                "activity_id": "CMIP",
                "frequency": "mon",
                "grid_label": "gn",
            },
        )
        cmip6_facets = request._cmip6_facets
        # Unmapped keys should pass through unchanged
        assert cmip6_facets["activity_id"] == "CMIP"
        assert cmip6_facets["frequency"] == "mon"
        assert cmip6_facets["grid_label"] == "gn"


class TestCMIP7RequestFetchEdgeCases:
    """Tests for edge cases in CMIP7Request.fetch_datasets."""

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_fetch_datasets_conversion_failure(self, mock_convert, mock_cmip6_request_class, tmp_path):
        """Test fetch_datasets when file conversion fails."""
        cmip6_file = tmp_path / "tas.nc"
        cmip6_file.touch()

        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5"],
                "member_id": ["r1i1p1f1"],
                "variable_id": ["tas"],
                "table_id": ["Amon"],
                "files": [[str(cmip6_file)]],
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        # Conversion raises an exception
        mock_convert.side_effect = RuntimeError("Conversion failed")

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        result = request.fetch_datasets()

        # No converted files means empty result
        assert result.empty

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_fetch_datasets_multiple_files_per_row(self, mock_convert, mock_cmip6_request_class, tmp_path):
        """Test fetch_datasets with multiple files in a single row."""
        cmip6_file1 = tmp_path / "tas_200001.nc"
        cmip6_file1.touch()
        cmip6_file2 = tmp_path / "tas_200101.nc"
        cmip6_file2.touch()

        cmip7_file1 = tmp_path / "cmip7" / "tas_200001.nc"
        cmip7_file2 = tmp_path / "cmip7" / "tas_200101.nc"
        cmip7_file1.parent.mkdir(parents=True, exist_ok=True)
        cmip7_file1.touch()
        cmip7_file2.touch()

        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5"],
                "member_id": ["r1i1p1f1"],
                "variable_id": ["tas"],
                "table_id": ["Amon"],
                "files": [[str(cmip6_file1), str(cmip6_file2)]],
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        mock_convert.side_effect = [cmip7_file1, cmip7_file2]

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        result = request.fetch_datasets()

        assert not result.empty
        assert len(result) == 1
        assert len(result.iloc[0]["files"]) == 2
        assert mock_convert.call_count == 2

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_fetch_datasets_partial_conversion_failure(
        self, mock_convert, mock_cmip6_request_class, tmp_path
    ):
        """Test fetch_datasets when some files fail to convert but others succeed."""
        cmip6_file1 = tmp_path / "tas_200001.nc"
        cmip6_file1.touch()
        cmip6_file2 = tmp_path / "tas_200101.nc"
        cmip6_file2.touch()

        cmip7_file2 = tmp_path / "cmip7" / "tas_200101.nc"
        cmip7_file2.parent.mkdir(parents=True, exist_ok=True)
        cmip7_file2.touch()

        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5"],
                "member_id": ["r1i1p1f1"],
                "variable_id": ["tas"],
                "table_id": ["Amon"],
                "files": [[str(cmip6_file1), str(cmip6_file2)]],
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        # First file fails, second succeeds
        mock_convert.side_effect = [RuntimeError("Conversion failed"), cmip7_file2]

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        result = request.fetch_datasets()

        assert not result.empty
        assert len(result.iloc[0]["files"]) == 1  # Only second file converted

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    def test_fetch_datasets_passes_remove_ensembles(self, mock_cmip6_request_class):
        """Test that remove_ensembles is passed to CMIP6Request."""
        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame()
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5"},
            remove_ensembles=True,
            time_span=("2000-01", "2010-12"),
        )
        request.fetch_datasets()

        # Verify CMIP6Request was created with correct params
        mock_cmip6_request_class.assert_called_once_with(
            slug="test-cmip6-source",
            facets={"source_id": "ACCESS-ESM1-5"},
            remove_ensembles=True,
            time_span=("2000-01", "2010-12"),
        )

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_fetch_datasets_multiple_rows(self, mock_convert, mock_cmip6_request_class, tmp_path):
        """Test fetch_datasets with multiple rows in the CMIP6 result."""
        cmip6_file1 = tmp_path / "tas.nc"
        cmip6_file1.touch()
        cmip6_file2 = tmp_path / "pr.nc"
        cmip6_file2.touch()

        cmip7_file1 = tmp_path / "cmip7" / "tas.nc"
        cmip7_file2 = tmp_path / "cmip7" / "pr.nc"
        cmip7_file1.parent.mkdir(parents=True, exist_ok=True)
        cmip7_file1.touch()
        cmip7_file2.touch()

        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5", "ACCESS-ESM1-5"],
                "member_id": ["r1i1p1f1", "r1i1p1f1"],
                "variable_id": ["tas", "pr"],
                "table_id": ["Amon", "Amon"],
                "files": [[str(cmip6_file1)], [str(cmip6_file2)]],
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

        mock_convert.side_effect = [cmip7_file1, cmip7_file2]

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5"},
        )
        result = request.fetch_datasets()

        assert not result.empty
        assert len(result) == 2
        assert result.iloc[0]["mip_era"] == "CMIP7"
        assert result.iloc[1]["mip_era"] == "CMIP7"


class TestLatestFile:
    """Test picking the file that holds a dataset's final timestep."""

    def _write_monthly_file(self, path: Path, start_year: int, end_year: int) -> Path:
        n = (end_year - start_year + 1) * 12
        time = [cftime.DatetimeNoLeap(start_year + index // 12, index % 12 + 1, 16) for index in range(n)]
        xr.Dataset({"toz": ("time", np.zeros(n))}, coords={"time": time}).to_netcdf(path)
        return path

    def test_picks_the_chunk_ending_last(self, tmp_path):
        """Only the final chunk is extended, so the earlier ones stay where they are."""
        early = self._write_monthly_file(tmp_path / "early.nc", 1850, 1949)
        late = self._write_monthly_file(tmp_path / "late.nc", 1950, 2014)

        assert _latest_file([early, late]).path == late
        assert _latest_file([late, early]).path == late

    def test_returns_the_final_timestep(self, tmp_path):
        """The end is needed to tell whether the dataset falls short of the requested end."""
        late = self._write_monthly_file(tmp_path / "late.nc", 1950, 2014)

        latest = _latest_file([late])
        assert (latest.end.year, latest.end.month) == (2014, 12)

    def test_skips_files_without_a_time_axis(self, tmp_path):
        """Fixed-frequency files carry no time axis, so they never win."""
        fixed = tmp_path / "areacella.nc"
        xr.Dataset({"areacella": ("lat", np.zeros(2))}, coords={"lat": [0.0, 1.0]}).to_netcdf(fixed)
        monthly = self._write_monthly_file(tmp_path / "a.nc", 1950, 2014)

        assert _latest_file([fixed, monthly]).path == monthly

    def test_no_time_axes_at_all(self, tmp_path):
        """A dataset with nothing to extend has no latest file."""
        fixed = tmp_path / "areacella.nc"
        xr.Dataset({"areacella": ("lat", np.zeros(2))}, coords={"lat": [0.0, 1.0]}).to_netcdf(fixed)

        assert _latest_file([fixed]) is None


class TestFetchDatasetsExtendHistorical:
    """Test that only the file ending last is extended."""

    def _write_monthly_file(self, path: Path, start_year: int, end_year: int) -> Path:
        n = (end_year - start_year + 1) * 12
        time = [cftime.DatetimeNoLeap(start_year + index // 12, index % 12 + 1, 16) for index in range(n)]
        xr.Dataset({"toz": ("time", np.zeros(n))}, coords={"time": time}).to_netcdf(path)
        return path

    def _stub_cmip6(self, mock_cmip6_request_class, files: list[Path], **extra: str) -> None:
        """Stand in for the ESGF fetch with a single CMIP6 row covering ``files``."""
        mock_cmip6_instance = MagicMock()
        mock_cmip6_instance.fetch_datasets.return_value = pd.DataFrame(
            {
                "source_id": ["GFDL-ESM4"],
                "member_id": ["r1i1p1f1"],
                "variable_id": ["toz"],
                "table_id": ["AERmon"],
                "files": [[str(path) for path in files]],
                **{key: [value] for key, value in extra.items()},
            }
        )
        mock_cmip6_request_class.return_value = mock_cmip6_instance

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_only_the_last_chunk_is_extended(self, mock_convert, mock_cmip6_request_class, tmp_path):
        """The earlier chunks stay where they are, so the converted files do not overlap."""
        early = self._write_monthly_file(tmp_path / "early.nc", 1850, 1949)
        late = self._write_monthly_file(tmp_path / "late.nc", 1950, 2014)

        self._stub_cmip6(mock_cmip6_request_class, [early, late])
        mock_convert.side_effect = [tmp_path / "a.nc", tmp_path / "b.nc"]

        request = CMIP7Request(
            slug="test",
            facets={"source_id": "GFDL-ESM4", "variable_id": "toz"},
            extend_historical_to=(2021, 12),
        )
        request.fetch_datasets()

        extended = {call.args[0]: call.kwargs["extend_historical_to"] for call in mock_convert.call_args_list}
        assert extended == {early: None, late: (2021, 12)}

        # Every file of the dataset moves to the bumped version, so the fabricated years
        # do not share an instance_id with the real data.
        assert {call.args[1]["version"] for call in mock_convert.call_args_list} == {"v1"}

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_no_extension_requested(self, mock_convert, mock_cmip6_request_class, tmp_path):
        """Without the opt-in no file is extended, and the files are not opened to find out."""
        only = self._write_monthly_file(tmp_path / "only.nc", 1950, 2014)

        self._stub_cmip6(mock_cmip6_request_class, [only])
        mock_convert.return_value = tmp_path / "a.nc"

        CMIP7Request(slug="test", facets={"source_id": "GFDL-ESM4"}).fetch_datasets()

        assert mock_convert.call_args_list[0].kwargs["extend_historical_to"] is None

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    @patch("climate_ref_core.esgf.cmip7._convert_file_to_cmip7")
    def test_version_untouched_when_nothing_is_fabricated(
        self, mock_convert, mock_cmip6_request_class, tmp_path
    ):
        """A dataset already reaching the requested end is real data, so it keeps its version."""
        only = self._write_monthly_file(tmp_path / "only.nc", 1950, 2021)

        self._stub_cmip6(mock_cmip6_request_class, [only], version="20190429")
        mock_convert.return_value = tmp_path / "a.nc"

        CMIP7Request(
            slug="test",
            facets={"source_id": "GFDL-ESM4"},
            extend_historical_to=(2021, 12),
        ).fetch_datasets()

        assert mock_convert.call_args_list[0].kwargs["extend_historical_to"] is None
        assert mock_convert.call_args_list[0].args[1]["version"] == "20190429"


class TestBumpVersion:
    """Test the version bump that keeps fabricated data out of the real instance_id."""

    @pytest.mark.parametrize(
        "version, expected",
        [
            ("v0", "v1"),
            ("v1", "v2"),
            ("v9", "v10"),
            # The ingest cannot parse a version out of these, so it sees v0.
            ("20190429", "v1"),
            ("", "v1"),
        ],
    )
    def test_bump(self, version, expected):
        assert _bump_version(version) == expected
