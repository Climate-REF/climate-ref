"""Tests for climate_ref_core.esgf.fetcher module."""

from unittest.mock import MagicMock

import pandas as pd

from climate_ref_core.esgf import ESGFFetcher


class TestESGFFetcher:
    """Tests for ESGFFetcher class."""

    def test_init(self):
        """Test basic initialization."""
        fetcher = ESGFFetcher()
        assert fetcher is not None

    def test_list_requests_for_diagnostic_no_spec(self):
        """Test listing requests when diagnostic has no test_data_spec."""
        fetcher = ESGFFetcher()

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = None

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert result == []

    def test_list_requests_for_diagnostic_with_requests(self):
        """Test listing requests when diagnostic has test cases with requests."""
        fetcher = ESGFFetcher()

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

    def test_list_requests_for_diagnostic_no_requests(self):
        """Test listing requests when test case has no requests."""
        fetcher = ESGFFetcher()

        mock_test_case = MagicMock()
        mock_test_case.name = "default"
        mock_test_case.requests = None

        mock_spec = MagicMock()
        mock_spec.test_cases = (mock_test_case,)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = mock_spec

        result = fetcher.list_requests_for_diagnostic(mock_diagnostic)
        assert result == []


class TestFetchRequest:
    """Tests for ESGFFetcher.fetch_request method."""

    def test_empty_datasets_returns_empty_dataframe(self):
        """Test fetch_request returns empty DataFrame when no datasets found."""
        fetcher = ESGFFetcher()

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame()

        result = fetcher.fetch_request(mock_request)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_no_files_returns_empty_dataframe(self):
        """Test fetch_request returns empty DataFrame when datasets have no files."""
        fetcher = ESGFFetcher()

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [[]],
            }
        )

        result = fetcher.fetch_request(mock_request)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_file_not_exists_returns_empty_dataframe(self):
        """Test fetch_request skips non-existent files."""
        fetcher = ESGFFetcher()

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.source_type = "CMIP6"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "files": [["/nonexistent/path/file.nc"]],
            }
        )

        result = fetcher.fetch_request(mock_request)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_with_valid_files(self, tmp_path):
        """Test fetch_request returns DataFrame with path column for valid files."""
        fetcher = ESGFFetcher()

        # Create a mock source file
        source_file = tmp_path / "test_data.nc"
        source_file.write_text("mock netcdf content")

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.source_type = "CMIP6"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "files": [[str(source_file)]],
            }
        )

        result = fetcher.fetch_request(mock_request)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "path" in result.columns
        assert "source_type" in result.columns
        assert result.iloc[0]["path"] == str(source_file)
        assert result.iloc[0]["source_type"] == "CMIP6"
        assert result.iloc[0]["variable_id"] == "tas"
        assert result.iloc[0]["experiment_id"] == "historical"

    def test_expands_multiple_files(self, tmp_path):
        """Test fetch_request expands multiple files into separate rows."""
        fetcher = ESGFFetcher()

        # Create mock source files
        file1 = tmp_path / "file1.nc"
        file2 = tmp_path / "file2.nc"
        file1.write_text("content1")
        file2.write_text("content2")

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.source_type = "obs4MIPs"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "variable_id": ["pr"],
                "files": [[str(file1), str(file2)]],
            }
        )

        result = fetcher.fetch_request(mock_request)

        assert len(result) == 2
        paths = set(result["path"].tolist())
        assert str(file1) in paths
        assert str(file2) in paths
        # Both rows should have the same metadata
        assert all(result["variable_id"] == "pr")
        assert all(result["source_type"] == "obs4MIPs")

    def test_multiple_datasets(self, tmp_path):
        """Test fetch_request handles multiple datasets."""
        fetcher = ESGFFetcher()

        file1 = tmp_path / "file1.nc"
        file2 = tmp_path / "file2.nc"
        file1.write_text("content1")
        file2.write_text("content2")

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.source_type = "CMIP6"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1", "dataset2"],
                "variable_id": ["tas", "pr"],
                "files": [[str(file1)], [str(file2)]],
            }
        )

        result = fetcher.fetch_request(mock_request)

        assert len(result) == 2
        assert set(result["variable_id"]) == {"tas", "pr"}

    def test_skips_missing_files_keeps_existing(self, tmp_path):
        """Test fetch_request skips missing files but keeps existing ones."""
        fetcher = ESGFFetcher()

        existing_file = tmp_path / "exists.nc"
        existing_file.write_text("content")

        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.source_type = "CMIP6"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1", "dataset2"],
                "variable_id": ["tas", "pr"],
                "files": [[str(existing_file)], ["/nonexistent/file.nc"]],
            }
        )

        result = fetcher.fetch_request(mock_request)

        assert len(result) == 1
        assert result.iloc[0]["variable_id"] == "tas"


class TestFetchForTestCase:
    """Tests for ESGFFetcher.fetch_for_test_case method."""

    def test_no_requests_returns_empty_dataframe(self):
        """Test fetch_for_test_case with no requests."""
        fetcher = ESGFFetcher()

        result = fetcher.fetch_for_test_case(None)
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_requests_returns_empty_dataframe(self):
        """Test fetch_for_test_case with empty requests tuple."""
        fetcher = ESGFFetcher()

        result = fetcher.fetch_for_test_case(())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_request(self, tmp_path):
        """Test fetch_for_test_case with single request."""
        fetcher = ESGFFetcher()

        source_file = tmp_path / "test.nc"
        source_file.write_text("content")

        mock_request = MagicMock()
        mock_request.slug = "request-1"
        mock_request.source_type = "CMIP6"
        mock_request.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "variable_id": ["tas"],
                "files": [[str(source_file)]],
            }
        )

        result = fetcher.fetch_for_test_case((mock_request,))

        assert len(result) == 1
        assert result.iloc[0]["variable_id"] == "tas"
        assert result.iloc[0]["source_type"] == "CMIP6"

    def test_multiple_requests_concatenated(self, tmp_path):
        """Test fetch_for_test_case concatenates multiple requests."""
        fetcher = ESGFFetcher()

        file1 = tmp_path / "file1.nc"
        file2 = tmp_path / "file2.nc"
        file1.write_text("content1")
        file2.write_text("content2")

        mock_request1 = MagicMock()
        mock_request1.slug = "request-1"
        mock_request1.source_type = "CMIP6"
        mock_request1.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "variable_id": ["tas"],
                "files": [[str(file1)]],
            }
        )

        mock_request2 = MagicMock()
        mock_request2.slug = "request-2"
        mock_request2.source_type = "obs4MIPs"
        mock_request2.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["obs_dataset"],
                "variable_id": ["pr"],
                "files": [[str(file2)]],
            }
        )

        result = fetcher.fetch_for_test_case((mock_request1, mock_request2))

        assert len(result) == 2
        assert set(result["variable_id"]) == {"tas", "pr"}
        assert set(result["source_type"]) == {"CMIP6", "obs4MIPs"}

    def test_skips_empty_results(self, tmp_path):
        """Test fetch_for_test_case skips requests that return empty results."""
        fetcher = ESGFFetcher()

        source_file = tmp_path / "test.nc"
        source_file.write_text("content")

        mock_request1 = MagicMock()
        mock_request1.slug = "empty-request"
        mock_request1.fetch_datasets.return_value = pd.DataFrame()

        mock_request2 = MagicMock()
        mock_request2.slug = "valid-request"
        mock_request2.source_type = "CMIP6"
        mock_request2.fetch_datasets.return_value = pd.DataFrame(
            {
                "key": ["dataset1"],
                "variable_id": ["tas"],
                "files": [[str(source_file)]],
            }
        )

        result = fetcher.fetch_for_test_case((mock_request1, mock_request2))

        assert len(result) == 1
        assert result.iloc[0]["variable_id"] == "tas"

    def test_all_empty_returns_empty_dataframe(self):
        """Test fetch_for_test_case returns empty when all requests are empty."""
        fetcher = ESGFFetcher()

        mock_request1 = MagicMock()
        mock_request1.slug = "empty-1"
        mock_request1.fetch_datasets.return_value = pd.DataFrame()

        mock_request2 = MagicMock()
        mock_request2.slug = "empty-2"
        mock_request2.fetch_datasets.return_value = pd.DataFrame()

        result = fetcher.fetch_for_test_case((mock_request1, mock_request2))

        assert isinstance(result, pd.DataFrame)
        assert result.empty
