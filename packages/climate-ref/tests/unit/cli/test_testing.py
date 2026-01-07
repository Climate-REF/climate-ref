from unittest.mock import MagicMock, patch

import pandas as pd

from climate_ref.cli.testing import (
    _build_esgf_data_catalog,
    _filter_catalog_by_requests,
    _find_diagnostic,
)
from climate_ref_core.datasets import SourceDatasetType


class TestFilterCatalogByRequests:
    """Tests for _filter_catalog_by_requests function."""

    def test_empty_catalog(self):
        """Test filtering an empty catalog returns empty."""
        catalog = pd.DataFrame()
        requests = [MagicMock(facets={"source_id": "test"})]
        result = _filter_catalog_by_requests(catalog, requests)
        assert result.empty

    def test_no_requests(self):
        """Test filtering with no requests returns original catalog."""
        catalog = pd.DataFrame({"source_id": ["A", "B"], "variable_id": ["tas", "pr"]})
        result = _filter_catalog_by_requests(catalog, [])
        assert len(result) == 2

    def test_single_facet_match(self):
        """Test filtering with single facet match."""
        catalog = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5", "CESM2", "ACCESS-ESM1-5"],
                "variable_id": ["tas", "tas", "pr"],
            }
        )
        request = MagicMock(facets={"source_id": "ACCESS-ESM1-5"})
        result = _filter_catalog_by_requests(catalog, [request])
        assert len(result) == 2
        assert all(result["source_id"] == "ACCESS-ESM1-5")

    def test_multiple_facets_match(self):
        """Test filtering with multiple facets (AND condition)."""
        catalog = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5", "ACCESS-ESM1-5", "CESM2"],
                "variable_id": ["tas", "pr", "tas"],
            }
        )
        request = MagicMock(facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"})
        result = _filter_catalog_by_requests(catalog, [request])
        assert len(result) == 1
        assert result.iloc[0]["source_id"] == "ACCESS-ESM1-5"
        assert result.iloc[0]["variable_id"] == "tas"

    def test_multiple_requests_or_condition(self):
        """Test filtering with multiple requests (OR condition)."""
        catalog = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5", "CESM2", "MPI-ESM"],
                "variable_id": ["tas", "tas", "tas"],
            }
        )
        request1 = MagicMock(facets={"source_id": "ACCESS-ESM1-5"})
        request2 = MagicMock(facets={"source_id": "CESM2"})
        result = _filter_catalog_by_requests(catalog, [request1, request2])
        assert len(result) == 2
        assert set(result["source_id"]) == {"ACCESS-ESM1-5", "CESM2"}

    def test_tuple_facet_values(self):
        """Test filtering with tuple facet values (isin condition)."""
        catalog = pd.DataFrame(
            {
                "source_id": ["ACCESS-ESM1-5", "CESM2", "MPI-ESM"],
                "variable_id": ["tas", "pr", "tas"],
            }
        )
        request = MagicMock(facets={"source_id": ("ACCESS-ESM1-5", "MPI-ESM")})
        result = _filter_catalog_by_requests(catalog, [request])
        assert len(result) == 2
        assert set(result["source_id"]) == {"ACCESS-ESM1-5", "MPI-ESM"}

    def test_missing_facet_column(self):
        """Test filtering when facet column doesn't exist in catalog."""
        catalog = pd.DataFrame({"source_id": ["ACCESS-ESM1-5", "CESM2"]})
        request = MagicMock(facets={"nonexistent_facet": "value"})
        result = _filter_catalog_by_requests(catalog, [request])
        # When facet column doesn't exist, it should be skipped
        assert len(result) == 2


class TestBuildEsgfDataCatalog:
    def test_no_requests(self):
        """Test with no requests returns empty catalog."""
        result = _build_esgf_data_catalog(None)
        assert result == {}

    def test_no_test_data_dir(self):
        with patch("climate_ref.cli.testing.TEST_DATA_DIR", None):
            result = _build_esgf_data_catalog(tuple())
        assert result == {}

    def test_cmip6_requests_with_data(self, tmp_path):
        """Test loading CMIP6 data with matching requests."""
        # Setup mock TEST_DATA_DIR
        test_data_dir = tmp_path / "test-data"
        esgf_data_dir = test_data_dir / "esgf-data"
        cmip6_dir = esgf_data_dir / "CMIP6"
        cmip6_dir.mkdir(parents=True)

        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {"source_id": ["ACCESS-ESM1-5"], "variable_id": ["tas"]}
        )

        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {"source_id": "ACCESS-ESM1-5"}

        with (
            patch("climate_ref.cli.testing.TEST_DATA_DIR", test_data_dir),
            patch("climate_ref.cli.testing.CMIP6DatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request,))

        assert SourceDatasetType.CMIP6 in result
        assert len(result[SourceDatasetType.CMIP6]) == 1

    def test_obs4mips_requests_with_data(self, tmp_path):
        """Test loading obs4MIPs data with matching requests."""
        test_data_dir = tmp_path / "test-data"
        esgf_data_dir = test_data_dir / "esgf-data"
        obs_dir = esgf_data_dir / "obs4MIPs"
        obs_dir.mkdir(parents=True)

        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {"source_id": ["GPCP-SG"], "variable_id": ["pr"]}
        )

        obs_request = MagicMock()
        obs_request.source_type = "obs4MIPs"
        obs_request.facets = {"source_id": "GPCP-SG"}

        with (
            patch("climate_ref.cli.testing.TEST_DATA_DIR", test_data_dir),
            patch("climate_ref.cli.testing.Obs4MIPsDatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((obs_request,))

        assert SourceDatasetType.obs4MIPs in result

    def test_cmip6_dir_not_exists(self, tmp_path):
        """Test when CMIP6 directory doesn't exist."""
        test_data_dir = tmp_path / "test-data"
        esgf_data_dir = test_data_dir / "esgf-data"
        esgf_data_dir.mkdir(parents=True)

        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {}

        with patch("climate_ref.cli.testing.TEST_DATA_DIR", test_data_dir):
            result = _build_esgf_data_catalog((cmip6_request,))
        assert SourceDatasetType.CMIP6 not in result

    def test_adapter_exception_handled(self, tmp_path):
        """Test that exceptions from adapter are handled gracefully."""
        test_data_dir = tmp_path / "test-data"
        esgf_data_dir = test_data_dir / "esgf-data"
        cmip6_dir = esgf_data_dir / "CMIP6"
        cmip6_dir.mkdir(parents=True)

        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = Exception("Failed to load")

        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {}

        with (
            patch("climate_ref.cli.testing.TEST_DATA_DIR", test_data_dir),
            patch("climate_ref.cli.testing.CMIP6DatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request,))

        assert SourceDatasetType.CMIP6 not in result

    def test_empty_filtered_result_not_added(self, tmp_path):
        """Test that empty filtered results are not added to catalog."""
        test_data_dir = tmp_path / "test-data"
        esgf_data_dir = test_data_dir / "esgf-data"
        cmip6_dir = esgf_data_dir / "CMIP6"
        cmip6_dir.mkdir(parents=True)

        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {"source_id": ["ACCESS-ESM1-5"], "variable_id": ["tas"]}
        )

        # Request that won't match any data
        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {"source_id": "NONEXISTENT"}

        with (
            patch("climate_ref.cli.testing.TEST_DATA_DIR", test_data_dir),
            patch("climate_ref.cli.testing.CMIP6DatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request,))

        assert SourceDatasetType.CMIP6 not in result


class TestFindDiagnostic:
    def test_find_existing_diagnostic(self):
        """Test finding an existing diagnostic."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"

        mock_provider = MagicMock()
        mock_provider.slug = "test-provider"
        mock_provider.diagnostics.return_value = [mock_diagnostic]

        mock_registry = MagicMock()
        mock_registry.providers = [mock_provider]

        result = _find_diagnostic(mock_registry, "test-provider", "test-diag")
        assert result is mock_diagnostic

    def test_find_nonexistent_provider(self):
        """Test finding diagnostic with non-existent provider."""
        mock_provider = MagicMock()
        mock_provider.slug = "other-provider"

        mock_registry = MagicMock()
        mock_registry.providers = [mock_provider]

        result = _find_diagnostic(mock_registry, "nonexistent-provider", "test-diag")
        assert result is None

    def test_find_nonexistent_diagnostic(self):
        """Test finding non-existent diagnostic in existing provider."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "other-diag"

        mock_provider = MagicMock()
        mock_provider.slug = "test-provider"
        mock_provider.diagnostics.return_value = [mock_diagnostic]

        mock_registry = MagicMock()
        mock_registry.providers = [mock_provider]

        result = _find_diagnostic(mock_registry, "test-provider", "nonexistent-diag")
        assert result is None

    def test_find_with_multiple_providers(self):
        """Test finding diagnostic across multiple providers."""
        mock_diag1 = MagicMock()
        mock_diag1.slug = "diag-1"

        mock_diag2 = MagicMock()
        mock_diag2.slug = "diag-2"

        mock_provider1 = MagicMock()
        mock_provider1.slug = "provider-1"
        mock_provider1.diagnostics.return_value = [mock_diag1]

        mock_provider2 = MagicMock()
        mock_provider2.slug = "provider-2"
        mock_provider2.diagnostics.return_value = [mock_diag2]

        mock_registry = MagicMock()
        mock_registry.providers = [mock_provider1, mock_provider2]

        result = _find_diagnostic(mock_registry, "provider-2", "diag-2")
        assert result is mock_diag2


class TestFetchTestDataCommand:
    def test_fetch_help(self, invoke_cli):
        """Test fetch command help."""
        result = invoke_cli(["testing", "fetch", "--help"])
        assert "Fetch test data from ESGF" in result.stdout

    def test_fetch_dry_run(self, invoke_cli):
        """Test fetch command with dry run."""
        result = invoke_cli(["testing", "fetch", "--dry-run"])
        # Should complete successfully even if no diagnostics have test_data_spec
        assert result.exit_code == 0

    def test_fetch_no_test_data_dir(self, invoke_cli, mocker):
        """Test fetch command when TEST_DATA_DIR is not available."""
        mocker.patch("climate_ref.testing.TEST_DATA_DIR", None)
        invoke_cli(["testing", "fetch"], expected_exit_code=1)

    def test_fetch_with_provider_filter(self, invoke_cli):
        """Test fetch command with provider filter."""
        result = invoke_cli(["testing", "fetch", "--provider", "example", "--dry-run"])
        assert result.exit_code == 0


class TestListCasesCommand:
    def test_list_help(self, invoke_cli):
        """Test list command help."""
        result = invoke_cli(["testing", "list", "--help"])
        assert "List test cases" in result.stdout

    def test_list_all(self, invoke_cli):
        """Test listing all test cases."""
        result = invoke_cli(["testing", "list"])
        assert result.exit_code == 0
        assert "Provider" in result.stdout
        assert "Diagnostic" in result.stdout

    def test_list_with_provider_filter(self, invoke_cli):
        """Test listing test cases with provider filter."""
        result = invoke_cli(["testing", "list", "--provider", "example"])
        assert result.exit_code == 0


class TestRunTestCaseCommand:
    def test_run_help(self, invoke_cli):
        """Test run command help."""
        result = invoke_cli(["testing", "run", "--help"])
        assert "Run a specific test case" in result.stdout

    def test_run_nonexistent_diagnostic(self, invoke_cli):
        """Test running non-existent diagnostic."""
        invoke_cli(
            ["testing", "run", "--provider", "nonexistent", "--diagnostic", "fake"],
            expected_exit_code=1,
        )

    def test_run_diagnostic_no_test_data_spec(self, invoke_cli):
        """Test running diagnostic without test_data_spec."""
        # The example provider's diagnostics may not have test_data_spec
        invoke_cli(
            ["testing", "run", "--provider", "example", "--diagnostic", "global-mean-timeseries"],
            expected_exit_code=1,
        )

    def test_run_nonexistent_test_case(self, invoke_cli, mocker):
        """Test running non-existent test case."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = False
        mock_diag.test_data_spec.case_names = ["default"]

        mocker.patch("climate_ref.cli.testing._find_diagnostic", return_value=mock_diag)

        invoke_cli(
            [
                "testing",
                "run",
                "--provider",
                "example",
                "--diagnostic",
                "test",
                "--test-case",
                "nonexistent",
            ],
            expected_exit_code=1,
        )
