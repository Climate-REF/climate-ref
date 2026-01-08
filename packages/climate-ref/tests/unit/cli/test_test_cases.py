from unittest.mock import MagicMock, patch

import pandas as pd

from climate_ref.cli.test_cases import (
    _fetch_and_build_catalog,
    _find_diagnostic,
)
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)


class TestFetchAndBuildCatalog:
    """Tests for _fetch_and_build_catalog function."""

    def test_no_requests_raises_error(self):
        """Test with no requests raises DatasetResolutionError."""
        mock_diag = MagicMock()
        mock_diag.provider.slug = "test-provider"
        mock_diag.slug = "test-diag"

        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.requests = None

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame()

        with patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher):
            try:
                _fetch_and_build_catalog(mock_diag, mock_tc)
                assert False, "Should have raised DatasetResolutionError"
            except DatasetResolutionError:
                pass

    def test_cmip6_data_returned(self):
        """Test with CMIP6 data returns properly grouped catalog."""
        mock_diag = MagicMock()
        mock_diag.provider.slug = "test-provider"
        mock_diag.slug = "test-diag"

        mock_tc = MagicMock()
        mock_tc.name = "default"

        mock_datasets = MagicMock()

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["CMIP6", "CMIP6"],
                "source_id": ["ACCESS-ESM1-5", "ACCESS-ESM1-5"],
                "variable_id": ["tas", "pr"],
                "path": ["/path/to/tas.nc", "/path/to/pr.nc"],
            }
        )

        # Mock the adapter to return a proper catalog DataFrame
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["CMIP6.test.tas"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "frequency": ["mon"],
                "path": ["/path/to/tas.nc"],
            }
        )

        with (
            patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.cli.test_cases.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases.solve_test_case", return_value=mock_datasets),
            patch("climate_ref.cli.test_cases.get_catalog_path", return_value=None),
        ):
            result = _fetch_and_build_catalog(mock_diag, mock_tc)

        assert result is mock_datasets

    def test_saves_catalog_yaml(self, tmp_path):
        """Test that catalog YAML is saved when path is available."""
        mock_diag = MagicMock()
        mock_diag.provider.slug = "test-provider"
        mock_diag.slug = "test-diag"

        mock_tc = MagicMock()
        mock_tc.name = "default"

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["CMIP6"],
                "source_id": ["ACCESS-ESM1-5"],
                "path": ["/path/to/file.nc"],
            }
        )

        # Mock the adapter to return a proper catalog DataFrame
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["CMIP6.test.tas"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "frequency": ["mon"],
                "path": ["/path/to/file.nc"],
            }
        )

        with (
            patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.cli.test_cases.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases.solve_test_case", return_value=mock_datasets),
            patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path),
            patch("climate_ref.cli.test_cases.save_datasets_to_yaml") as mock_save,
        ):
            _fetch_and_build_catalog(mock_diag, mock_tc)

        mock_save.assert_called_once_with(mock_datasets, catalog_path)


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

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(["test-cases", "fetch", "--test-case", "specific-case", "--dry-run"])
        assert result.exit_code == 0


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

    def test_run_no_catalog_file(self, invoke_cli, mocker):
        """Test run command when catalog file doesn't exist."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=None)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_with_fetch_flag(self, invoke_cli, mocker, tmp_path):
        """Test run command with --fetch flag fetches data first."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_datasets = MagicMock()
        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = None
        mock_result.output_bundle_filename = None

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases._fetch_and_build_catalog", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.cli.test_cases.TEST_DATA_DIR", None)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test", "--fetch"],
        )
        assert result.exit_code == 0

    def test_run_dataset_resolution_error(self, invoke_cli, mocker, tmp_path):
        """Test run command handles DatasetResolutionError."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_runner = MagicMock()
        mock_runner.run.side_effect = DatasetResolutionError("No datasets found for requirement")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_no_test_data_spec_error(self, invoke_cli, mocker, tmp_path):
        """Test run command handles NoTestDataSpecError from runner."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_runner = MagicMock()
        mock_runner.run.side_effect = NoTestDataSpecError("No test data spec")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_test_case_not_found_error(self, invoke_cli, mocker, tmp_path):
        """Test run command handles TestCaseNotFoundError from runner."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)
        mock_diag.test_data_spec.case_names = ["default", "other"]

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_runner = MagicMock()
        mock_runner.run.side_effect = TestCaseNotFoundError("Test case not found")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_general_exception(self, invoke_cli, mocker, tmp_path):
        """Test run command handles general Exception from runner."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_runner = MagicMock()
        mock_runner.run.side_effect = Exception("Unexpected error")

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

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

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = "metrics.json"
        mock_result.output_bundle_filename = "output.json"
        mock_result.to_output_path.side_effect = lambda x: tmp_path / x

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.cli.test_cases.TEST_DATA_DIR", None)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
        )
        assert result.exit_code == 0

    def test_run_unsuccessful_execution(self, invoke_cli, mocker, tmp_path):
        """Test run command with unsuccessful execution result."""
        mock_diag = MagicMock()
        mock_diag.test_data_spec = MagicMock()
        mock_diag.test_data_spec.has_case.return_value = True
        mock_diag.test_data_spec.get_case.return_value = MagicMock(requests=None)

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_result = MagicMock()
        mock_result.successful = False

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

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

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = "metrics.json"
        mock_result.output_bundle_filename = None
        mock_result.to_output_path.return_value = metrics_file

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.cli.test_cases.TEST_DATA_DIR", test_data_dir)

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

        mock_datasets = MagicMock()
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()

        mock_result = MagicMock()
        mock_result.successful = True
        mock_result.metric_bundle_filename = "metrics.json"
        mock_result.output_bundle_filename = None
        mock_result.to_output_path.return_value = tmp_path / "metrics.json"

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)
        mocker.patch("climate_ref.cli.test_cases.get_catalog_path", return_value=catalog_path)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=mock_datasets)
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.cli.test_cases.TEST_DATA_DIR", test_data_dir)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
        )
        assert result.exit_code == 0
        # Baseline should not be modified
        assert baseline_file.read_text() == '{"existing": "baseline"}'
