"""Tests for climate_ref_core.esgf.fetcher module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from climate_ref_core.esgf import ESGFFetcher


class TestESGFFetcher:
    """Tests for ESGFFetcher class."""

    def test_init(self, tmp_path):
        """Test basic initialization."""
        output_dir = tmp_path / "output"
        fetcher = ESGFFetcher(output_dir=output_dir)
        assert fetcher.output_dir == output_dir
        assert fetcher.cache_dir is None
        assert output_dir.exists()

    def test_init_with_cache(self, tmp_path):
        """Test initialization with cache directory."""
        output_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"
        fetcher = ESGFFetcher(output_dir=output_dir, cache_dir=cache_dir)
        assert fetcher.output_dir == output_dir
        assert fetcher.cache_dir == cache_dir

    def test_init_creates_output_dir(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new" / "nested" / "output"
        assert not output_dir.exists()
        ESGFFetcher(output_dir=output_dir)
        assert output_dir.exists()

    def test_list_requests_for_diagnostic_no_spec(self, tmp_path):
        """Test listing requests when diagnostic has no test_data_spec."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = None

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert result == []

    def test_list_requests_for_diagnostic_with_requests(self, tmp_path):
        """Test listing requests when diagnostic has test cases with requests."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        # Create mock requests
        mock_request1 = MagicMock()
        mock_request2 = MagicMock()

        # Create mock test cases
        mock_test_case1 = MagicMock()
        mock_test_case1.name = "default"
        mock_test_case1.requests = (mock_request1,)

        mock_test_case2 = MagicMock()
        mock_test_case2.name = "edge-case"
        mock_test_case2.requests = (mock_request2,)

        # Create mock test_data_spec
        mock_spec = MagicMock()
        mock_spec.test_cases = (mock_test_case1, mock_test_case2)

        # Create mock diagnostic
        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = mock_spec

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert len(result) == 2
        assert result[0] == ("default", mock_request1)
        assert result[1] == ("edge-case", mock_request2)

    def test_list_requests_for_diagnostic_no_requests(self, tmp_path):
        """Test listing requests when test case has no requests."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_test_case = MagicMock()
        mock_test_case.name = "default"
        mock_test_case.requests = None

        mock_spec = MagicMock()
        mock_spec.test_cases = (mock_test_case,)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = mock_spec

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert result == []

    def test_fetch_request_empty_datasets(self, tmp_path):
        """Test fetch_request returns empty list when no datasets found."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame()

        result = fetcher.fetch_request(mock_request)
        assert result == []

    def test_fetch_request_no_files(self, tmp_path):
        """Test fetch_request handles datasets with no files."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [[]],
            }
        )

        result = fetcher.fetch_request(mock_request)
        assert result == []

    def test_fetch_request_file_not_exists(self, tmp_path):
        """Test fetch_request handles non-existent files gracefully."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [["/nonexistent/path/file.nc"]],
            }
        )

        result = fetcher.fetch_request(mock_request)
        assert result == []

    def test_fetch_request_with_valid_files(self, tmp_path):
        """Test fetch_request copies files correctly."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        # Create a mock source file
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test_data.nc"
        source_file.write_text("mock netcdf content")

        mock_ds = MagicMock()
        mock_ds.close = MagicMock()

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [[str(source_file)]],
            }
        )
        mock_request.generate_output_path.return_value = Path("output/test_data.nc")

        with patch("climate_ref_core.esgf.fetcher.xr.open_dataset", return_value=mock_ds):
            result = fetcher.fetch_request(mock_request)

        assert len(result) == 1
        assert result[0].exists()
        assert result[0].name == "test_data.nc"

    def test_fetch_request_with_symlink(self, tmp_path):
        """Test fetch_request creates symlinks when requested."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test_data.nc"
        source_file.write_text("mock netcdf content")

        mock_ds = MagicMock()
        mock_ds.close = MagicMock()

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [[str(source_file)]],
            }
        )
        mock_request.generate_output_path.return_value = Path("output/test_data.nc")

        with patch("climate_ref_core.esgf.fetcher.xr.open_dataset", return_value=mock_ds):
            result = fetcher.fetch_request(mock_request, symlink=True)

        assert len(result) == 1
        assert result[0].is_symlink()

    def test_fetch_request_skips_existing_files(self, tmp_path):
        """Test fetch_request skips files that already exist."""
        output_dir = tmp_path / "output"
        fetcher = ESGFFetcher(output_dir=output_dir)

        # Create source and destination files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test_data.nc"
        source_file.write_text("mock netcdf content")

        existing_output = output_dir / "existing" / "test_data.nc"
        existing_output.parent.mkdir(parents=True)
        existing_output.write_text("already exists")

        mock_ds = MagicMock()
        mock_ds.close = MagicMock()

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [[str(source_file)]],
            }
        )
        mock_request.generate_output_path.return_value = Path("existing/test_data.nc")

        with patch("climate_ref_core.esgf.fetcher.xr.open_dataset", return_value=mock_ds):
            result = fetcher.fetch_request(mock_request)

        assert len(result) == 1
        # Content should not have changed
        assert existing_output.read_text() == "already exists"

    def test_fetch_request_handles_xarray_error(self, tmp_path):
        """Test fetch_request handles errors when loading dataset."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "bad_data.nc"
        source_file.write_text("invalid content")

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [[str(source_file)]],
            }
        )

        with patch(
            "climate_ref_core.esgf.fetcher.xr.open_dataset",
            side_effect=Exception("Invalid NetCDF file"),
        ):
            result = fetcher.fetch_request(mock_request)

        assert result == []

    def test_fetch_for_diagnostic_no_spec(self, tmp_path):
        """Test fetch_for_diagnostic with no test_data_spec."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diagnostic"
        mock_diagnostic.test_data_spec = None

        result = fetcher.fetch_for_diagnostic(mock_diagnostic)
        assert result == {}

    def test_fetch_for_diagnostic_no_requests(self, tmp_path):
        """Test fetch_for_diagnostic with test case having no requests."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        mock_test_case = MagicMock()
        mock_test_case.requests = None

        mock_spec = MagicMock()
        mock_spec.get_case.return_value = mock_test_case

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diagnostic"
        mock_diagnostic.test_data_spec = mock_spec

        result = fetcher.fetch_for_diagnostic(mock_diagnostic)
        assert result == {}

    def test_fetch_for_diagnostic_with_requests(self, tmp_path):
        """Test fetch_for_diagnostic with valid requests."""
        fetcher = ESGFFetcher(output_dir=tmp_path)

        # Create mock source files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test_data.nc"
        source_file.write_text("mock content")

        mock_ds = MagicMock()
        mock_ds.close = MagicMock()

        mock_request = MagicMock()
        mock_request.slug = "request-1"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["ds1"],
                "files": [[str(source_file)]],
            }
        )
        mock_request.generate_output_path.return_value = Path("out/test.nc")

        mock_test_case = MagicMock()
        mock_test_case.requests = [mock_request]

        mock_spec = MagicMock()
        mock_spec.get_case.return_value = mock_test_case

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diagnostic"
        mock_diagnostic.test_data_spec = mock_spec

        with patch("climate_ref_core.esgf.fetcher.xr.open_dataset", return_value=mock_ds):
            result = fetcher.fetch_for_diagnostic(mock_diagnostic)

        assert "request-1" in result
        assert len(result["request-1"]) == 1
