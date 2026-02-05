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
        assert "climate-ref" in str(result)
        assert "cmip7-converted" in str(result)


class TestConvertFileToCmip7:
    """Tests for _convert_file_to_cmip7 function."""

    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_uses_cached_file(self, mock_cache_dir, mock_convert, mock_open, tmp_path):
        """Test that cached files are reused."""
        # Set up cache directory with existing file
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Create the expected output path structure
        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": "CSIRO",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "grid_label": "gn",
            "version": "v1",
        }
        drs_path = cache_dir / Path(
            "CMIP",
            "CSIRO",
            "ACCESS-ESM1-5",
            "historical",
            "r1i1p1f1",
            "mon",
            "tas",
            "gn",
            "v1",
        )
        drs_path.mkdir(parents=True)

        # Create a cached file
        cached_file = drs_path / "test_input.nc"
        cached_file.touch()

        cmip6_path = Path("/some/path/test_input.nc")
        result = _convert_file_to_cmip7(cmip6_path, cmip7_facets)

        assert result == cached_file
        mock_open.assert_not_called()  # Should not try to open the file
        mock_convert.assert_not_called()  # Should not try to convert

    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_converts_new_file(self, mock_cache_dir, mock_convert, mock_open, tmp_path):
        """Test that new files are converted."""
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
            "grid_label": "gn",
            "version": "v1",
        }

        # Create input file
        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        result = _convert_file_to_cmip7(input_file, cmip7_facets)

        # Check that conversion happened
        mock_open.assert_called_once()
        mock_convert.assert_called_once_with(mock_ds)
        mock_converted_ds.to_netcdf.assert_called_once()

        # Check output path structure
        assert "CMIP" in str(result)
        assert "ACCESS-ESM1-5" in str(result)

    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_handles_integer_facet_values(self, mock_cache_dir, mock_convert, mock_open, tmp_path):
        """Test that integer facet values are converted to strings."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Set up mocks
        mock_ds = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(return_value=mock_ds)
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_converted_ds = MagicMock()
        mock_convert.return_value = mock_converted_ds

        # Use integer values for some facets
        cmip7_facets = {
            "activity_id": "CMIP",
            "institution_id": 123,  # Integer value
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "frequency": "mon",
            "variable_id": "tas",
            "grid_label": "gn",
            "version": 1,  # Integer value
        }

        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        # Should not raise an error
        result = _convert_file_to_cmip7(input_file, cmip7_facets)
        assert isinstance(result, Path)

    @patch("climate_ref_core.esgf.cmip7.xr.open_dataset")
    @patch("climate_ref_core.esgf.cmip7.convert_cmip6_dataset")
    @patch("climate_ref_core.esgf.cmip7._get_cmip7_cache_dir")
    def test_handles_permission_error_with_existing_file(
        self, mock_cache_dir, mock_convert, mock_open, tmp_path
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
            "grid_label": "gn",
            "version": "v1",
        }

        input_file = tmp_path / "test_input.nc"
        input_file.touch()

        # Pre-create the output directory and file (simulating race condition)
        drs_path = cache_dir / Path(
            "CMIP",
            "CSIRO",
            "ACCESS-ESM1-5",
            "historical",
            "r1i1p1f1",
            "mon",
            "tas",
            "gn",
            "v1",
        )
        drs_path.mkdir(parents=True)
        existing_file = drs_path / "test_input.nc"
        existing_file.touch()

        # Should not raise, should return existing file
        result = _convert_file_to_cmip7(input_file, cmip7_facets)
        assert result == existing_file
