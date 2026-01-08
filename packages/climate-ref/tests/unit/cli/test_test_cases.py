from unittest.mock import MagicMock, patch

import pandas as pd

from climate_ref.cli.test_cases import (
    _build_esgf_data_catalog,
    _filter_catalog_by_requests,
    _find_diagnostic,
)
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)


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
        with patch("climate_ref.testing.ESGF_DATA_DIR", None):
            result = _build_esgf_data_catalog(tuple())
        assert result == {}

    def test_cmip6_requests_with_data(self, tmp_path):
        """Test loading CMIP6 data with matching requests."""
        # Setup mock ESGF_DATA_DIR
        esgf_data_dir = tmp_path / "esgf-data"
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
            patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir),
            patch("climate_ref.datasets.cmip6.CMIP6DatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request,))

        assert SourceDatasetType.CMIP6 in result
        assert len(result[SourceDatasetType.CMIP6]) == 1

    def test_obs4mips_requests_with_data(self, tmp_path):
        """Test loading obs4MIPs data with matching requests."""
        esgf_data_dir = tmp_path / "esgf-data"
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
            patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir),
            patch("climate_ref.datasets.obs4mips.Obs4MIPsDatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((obs_request,))

        assert SourceDatasetType.obs4MIPs in result

    def test_cmip6_dir_not_exists(self, tmp_path):
        """Test when CMIP6 directory doesn't exist."""
        esgf_data_dir = tmp_path / "esgf-data"
        esgf_data_dir.mkdir(parents=True)

        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {}

        with patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir):
            result = _build_esgf_data_catalog((cmip6_request,))
        assert SourceDatasetType.CMIP6 not in result

    def test_adapter_exception_handled(self, tmp_path):
        """Test that exceptions from adapter are handled gracefully."""
        esgf_data_dir = tmp_path / "esgf-data"
        cmip6_dir = esgf_data_dir / "CMIP6"
        cmip6_dir.mkdir(parents=True)

        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = Exception("Failed to load")

        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {}

        with (
            patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir),
            patch("climate_ref.datasets.cmip6.CMIP6DatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request,))

        assert SourceDatasetType.CMIP6 not in result

    def test_empty_filtered_result_not_added(self, tmp_path):
        """Test that empty filtered results are not added to catalog."""
        esgf_data_dir = tmp_path / "esgf-data"
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
            patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir),
            patch("climate_ref.datasets.cmip6.CMIP6DatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request,))

        assert SourceDatasetType.CMIP6 not in result

    def test_obs4mips_dir_not_exists(self, tmp_path):
        """Test when obs4MIPs directory doesn't exist."""
        esgf_data_dir = tmp_path / "esgf-data"
        esgf_data_dir.mkdir(parents=True)

        obs_request = MagicMock()
        obs_request.source_type = "obs4MIPs"
        obs_request.facets = {}

        with patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir):
            result = _build_esgf_data_catalog((obs_request,))
        assert SourceDatasetType.obs4MIPs not in result

    def test_obs4mips_adapter_exception_handled(self, tmp_path):
        """Test that exceptions from obs4MIPs adapter are handled gracefully."""
        esgf_data_dir = tmp_path / "esgf-data"
        obs_dir = esgf_data_dir / "obs4MIPs"
        obs_dir.mkdir(parents=True)

        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = Exception("Failed to load obs4MIPs")

        obs_request = MagicMock()
        obs_request.source_type = "obs4MIPs"
        obs_request.facets = {}

        with (
            patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir),
            patch("climate_ref.datasets.obs4mips.Obs4MIPsDatasetAdapter", return_value=mock_adapter),
        ):
            result = _build_esgf_data_catalog((obs_request,))

        assert SourceDatasetType.obs4MIPs not in result

    def test_mixed_requests_both_types(self, tmp_path):
        """Test loading both CMIP6 and obs4MIPs data with matching requests."""
        esgf_data_dir = tmp_path / "esgf-data"
        cmip6_dir = esgf_data_dir / "CMIP6"
        obs_dir = esgf_data_dir / "obs4MIPs"
        cmip6_dir.mkdir(parents=True)
        obs_dir.mkdir(parents=True)

        mock_cmip6_adapter = MagicMock()
        mock_cmip6_adapter.find_local_datasets.return_value = pd.DataFrame(
            {"source_id": ["ACCESS-ESM1-5"], "variable_id": ["tas"]}
        )

        mock_obs_adapter = MagicMock()
        mock_obs_adapter.find_local_datasets.return_value = pd.DataFrame(
            {"source_id": ["GPCP-SG"], "variable_id": ["pr"]}
        )

        cmip6_request = MagicMock()
        cmip6_request.source_type = "CMIP6"
        cmip6_request.facets = {"source_id": "ACCESS-ESM1-5"}

        obs_request = MagicMock()
        obs_request.source_type = "obs4MIPs"
        obs_request.facets = {"source_id": "GPCP-SG"}

        with (
            patch("climate_ref.testing.ESGF_DATA_DIR", esgf_data_dir),
            patch("climate_ref.datasets.cmip6.CMIP6DatasetAdapter", return_value=mock_cmip6_adapter),
            patch("climate_ref.datasets.obs4mips.Obs4MIPsDatasetAdapter", return_value=mock_obs_adapter),
        ):
            result = _build_esgf_data_catalog((cmip6_request, obs_request))

        assert SourceDatasetType.CMIP6 in result
        assert SourceDatasetType.obs4MIPs in result
        assert len(result[SourceDatasetType.CMIP6]) == 1
        assert len(result[SourceDatasetType.obs4MIPs]) == 1


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
        result = invoke_cli(["test-cases", "fetch", "--help"])
        assert "Fetch test data from ESGF" in result.stdout

    def test_fetch_dry_run(self, invoke_cli):
        """Test fetch command with dry run."""
        result = invoke_cli(["test-cases", "fetch", "--dry-run"])
        # Should complete successfully even if no diagnostics have test_data_spec
        assert result.exit_code == 0

    def test_fetch_no_test_data_dir(self, invoke_cli, mocker):
        """Test fetch command when ESGF_DATA_DIR is not available."""
        mocker.patch("climate_ref.testing.ESGF_DATA_DIR", None)
        invoke_cli(["test-cases", "fetch"], expected_exit_code=1)

    def test_fetch_with_provider_filter(self, invoke_cli):
        """Test fetch command with provider filter."""
        result = invoke_cli(["test-cases", "fetch", "--provider", "example", "--dry-run"])
        assert result.exit_code == 0

    def test_fetch_with_diagnostic_filter(self, invoke_cli):
        """Test fetch command with diagnostic filter."""
        result = invoke_cli(["test-cases", "fetch", "--diagnostic", "nonexistent-diagnostic", "--dry-run"])
        # Should complete since no matching diagnostics will be found
        assert result.exit_code == 0

    def test_fetch_with_test_case_filter(self, invoke_cli, mocker, tmp_path):
        """Test fetch command with test case filter in dry run."""
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()

        # Create a mock diagnostic with test_data_spec
        mock_request = MagicMock()
        mock_request.slug = "test-request"
        mock_request.source_type = "CMIP6"

        mock_test_case = MagicMock()
        mock_test_case.name = "specific-case"
        mock_test_case.description = "A specific test case"
        mock_test_case.requests = [mock_request]

        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_test_case]

        mock_diag = MagicMock()
        mock_diag.slug = "test-diag"
        mock_diag.test_data_spec = mock_spec
        mock_diag.provider = MagicMock(slug="test-provider")

        mock_provider = MagicMock()
        mock_provider.slug = "test-provider"
        mock_provider.diagnostics.return_value = [mock_diag]

        mock_registry = MagicMock()
        mock_registry.providers = [mock_provider]

        mocker.patch("climate_ref.testing.ESGF_DATA_DIR", test_data_dir / "esgf-data")
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(["test-cases", "fetch", "--test-case", "specific-case", "--dry-run"])
        assert result.exit_code == 0

    def test_fetch_with_custom_output_directory(self, invoke_cli, tmp_path, mocker):
        """Test fetch command with custom output directory."""
        output_dir = tmp_path / "custom-output"

        mock_fetcher = MagicMock()

        mock_request = MagicMock()
        mock_request.slug = "test-request"

        mock_test_case = MagicMock()
        mock_test_case.name = "default"
        mock_test_case.requests = [mock_request]

        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_test_case]

        mock_diag = MagicMock()
        mock_diag.slug = "test-diag"
        mock_diag.test_data_spec = mock_spec
        mock_diag.provider = MagicMock(slug="test-provider")

        mock_provider = MagicMock()
        mock_provider.slug = "test-provider"
        mock_provider.diagnostics.return_value = [mock_diag]

        mock_registry = MagicMock()
        mock_registry.providers = [mock_provider]

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher)

        result = invoke_cli(["test-cases", "fetch", f"--output-directory={output_dir}"])
        assert result.exit_code == 0
        mock_fetcher.fetch_request.assert_called()


class TestListCasesCommand:
    def test_list_help(self, invoke_cli):
        """Test list command help."""
        result = invoke_cli(["test-cases", "list", "--help"])
        assert "List test cases" in result.stdout

    def test_list_all(self, invoke_cli):
        """Test listing all test cases."""
        result = invoke_cli(["test-cases", "list"])
        assert result.exit_code == 0
        assert "Provider" in result.stdout
        assert "Diagnostic" in result.stdout

    def test_list_with_provider_filter(self, invoke_cli):
        """Test listing test cases with provider filter."""
        result = invoke_cli(["test-cases", "list", "--provider", "example"])
        assert result.exit_code == 0


class TestRunTestCaseCommand:
    def test_run_help(self, invoke_cli):
        """Test run command help."""
        result = invoke_cli(["test-cases", "run", "--help"])
        assert "Run a specific test case" in result.stdout

    def test_run_nonexistent_diagnostic(self, invoke_cli):
        """Test running non-existent diagnostic."""
        invoke_cli(
            ["test-cases", "run", "--provider", "nonexistent", "--diagnostic", "fake"],
            expected_exit_code=1,
        )

    def test_run_diagnostic_no_test_data_spec(self, invoke_cli, mocker):
        """Test running diagnostic without test_data_spec."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = None

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_nonexistent_test_case(self, invoke_cli, mocker):
        """Test running non-existent test case."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = False
        mock_diag.test_data_spec.case_names = ["default"]

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)

        invoke_cli(
            [
                "test-cases",
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

    def test_run_dataset_resolution_error(self, invoke_cli, mocker):
        """Test run command handles DatasetResolutionError."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_runner = MagicMock()
        mock_runner.run.side_effect = DatasetResolutionError("No datasets found for requirement")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_no_test_data_spec_error(self, invoke_cli, mocker):
        """Test run command handles NoTestDataSpecError from runner."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_runner = MagicMock()
        mock_runner.run.side_effect = NoTestDataSpecError("No test data spec")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_test_case_not_found_error(self, invoke_cli, mocker):
        """Test run command handles TestCaseNotFoundError from runner."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)
        mock_diag.test_data_spec.case_names = ["default", "other"]

        mock_runner = MagicMock()
        mock_runner.run.side_effect = TestCaseNotFoundError("Test case not found")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_general_exception(self, invoke_cli, mocker):
        """Test run command handles general Exception from runner."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_runner = MagicMock()
        mock_runner.run.side_effect = Exception("Unexpected error")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_successful_execution(self, invoke_cli, mocker, tmp_path):
        """Test run command with successful execution."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = "metrics.json"
        mock_result.output_bundle_filename = "output.json"
        mock_result.to_output_path.side_effect = lambda x: tmp_path / x

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.testing.TEST_DATA_DIR", None)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
        )
        assert result.exit_code == 0

    def test_run_unsuccessful_execution(self, invoke_cli, mocker):
        """Test run command with unsuccessful execution result."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_result = MagicMock()
        mock_result.successful = False

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_with_force_regen(self, invoke_cli, mocker, tmp_path):
        """Test run command with force_regen regenerates baseline."""
        test_data_dir = tmp_path / "test-data"
        test_data_dir.mkdir()

        # Create a mock metric file
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text('{"test": "data"}')

        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = "metrics.json"
        mock_result.output_bundle_filename = None
        mock_result.to_output_path.return_value = metrics_file

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.testing.TEST_DATA_DIR", test_data_dir)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test", "--force-regen"],
        )
        assert result.exit_code == 0

        # Check that regression baseline was created
        baseline_file = test_data_dir / "regression" / "example" / "test" / "default_metric.json"
        assert baseline_file.exists()

    def test_run_with_existing_baseline(self, invoke_cli, mocker, tmp_path):
        """Test run command logs when baseline exists."""
        test_data_dir = tmp_path / "test-data"
        regression_dir = test_data_dir / "regression" / "example" / "test"
        regression_dir.mkdir(parents=True)
        baseline_file = regression_dir / "default_metric.json"
        baseline_file.write_text('{"existing": "baseline"}')

        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = "metrics.json"
        mock_result.output_bundle_filename = None
        mock_result.to_output_path.return_value = tmp_path / "metrics.json"

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.testing.TEST_DATA_DIR", test_data_dir)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
        )
        assert result.exit_code == 0
        # Baseline should not be modified
        assert baseline_file.read_text() == '{"existing": "baseline"}'
