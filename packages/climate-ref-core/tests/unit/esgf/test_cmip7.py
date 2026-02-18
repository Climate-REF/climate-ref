"""Tests for climate_ref_core.esgf.cmip7 module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climate_ref_core.esgf import CMIP7Request
from climate_ref_core.esgf.cmip7 import _convert_file_to_cmip7, _get_cmip7_cache_dir


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

    def test_returns_path(self):
        """Test that function returns a Path object."""
        result = _get_cmip7_cache_dir()
        assert isinstance(result, Path)

    def test_path_contains_climate_ref(self):
        """Test that path contains climate-ref identifier."""
        result = _get_cmip7_cache_dir()
        assert "climate_ref" in str(result)
        assert "cmip7-converted" in str(result)


class TestConvertFileToCmip7:
    """Tests for _convert_file_to_cmip7 function."""

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
        mock_open.assert_called_once()  # Dataset is opened to derive time range
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

        # Create input file
        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        result = _convert_file_to_cmip7(input_file, cmip7_facets)

        # Check that conversion happened
        mock_open.assert_called_once()
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
