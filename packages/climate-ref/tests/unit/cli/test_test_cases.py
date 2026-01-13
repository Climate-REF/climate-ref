from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climate_ref.cli.test_cases import (
    _build_catalog,
    _fetch_and_build_catalog,
    _find_diagnostic,
)
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)


@pytest.fixture
def mock_fetcher():
    """Create a mock ESGFFetcher with default CMIP6 data."""
    fetcher = MagicMock()
    fetcher.fetch_for_test_case.return_value = pd.DataFrame(
        {
            "source_type": ["CMIP6"],
            "source_id": ["ACCESS-ESM1-5"],
            "variable_id": ["tas"],
            "path": ["/path/to/tas.nc"],
        }
    )
    return fetcher


@pytest.fixture
def mock_adapter():
    """Create a mock CMIP6DatasetAdapter."""
    adapter = MagicMock()
    adapter.find_local_datasets.return_value = pd.DataFrame(
        {
            "instance_id": ["CMIP6.test.tas"],
            "source_id": ["ACCESS-ESM1-5"],
            "variable_id": ["tas"],
            "frequency": ["mon"],
            "path": ["/path/to/tas.nc"],
        }
    )
    return adapter


@pytest.fixture
def mock_diagnostic():
    """Create a mock diagnostic with test_data_spec."""
    diag = MagicMock()
    diag.provider.slug = "test-provider"
    diag.slug = "test-diag"
    diag.test_data_spec = MagicMock()
    diag.test_data_spec.has_case.return_value = True
    diag.test_data_spec.get_case.return_value = MagicMock(requests=None)
    diag.test_data_spec.case_names = ["default"]
    return diag


@pytest.fixture
def mock_test_case():
    """Create a mock test case."""
    tc = MagicMock()
    tc.name = "default"
    return tc


class TestBuildCatalog:
    """Tests for _build_catalog function."""

    def test_empty_df_after_filtering(self, caplog):
        """Test warning when no matching files found after filtering."""
        mock_adapter = MagicMock()
        # Adapter returns df but none match the fetched files
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {"path": ["/other/file.nc"], "variable_id": ["tas"]}
        )

        file_paths = [Path("/path/to/fetched.nc")]
        result = _build_catalog(mock_adapter, file_paths)

        assert len(result) == 0
        assert "No matching files found" in caplog.text

    def test_adapter_exception_logged(self, caplog):
        """Test exception from adapter is logged as warning and concat fails."""
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = Exception("Parse error")

        file_paths = [Path("/path/to/file.nc")]
        # When all directories fail, pd.concat([]) raises ValueError
        with pytest.raises(ValueError, match="No objects to concatenate"):
            _build_catalog(mock_adapter, file_paths)

        assert "Failed to parse" in caplog.text


class TestFetchAndBuildCatalog:
    """Tests for _fetch_and_build_catalog function."""

    def test_no_requests_raises_error(self, mock_diagnostic, mock_test_case):
        """Test with no requests raises DatasetResolutionError."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame()

        with patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher):
            with pytest.raises(DatasetResolutionError):
                _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

    def test_cmip6_data_returned(self, mock_diagnostic, mock_test_case, mock_fetcher, mock_adapter):
        """Test with CMIP6 data returns properly grouped catalog."""
        mock_datasets = MagicMock()

        with (
            patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.cli.test_cases.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets

    def test_saves_catalog_yaml(self, tmp_path, mock_diagnostic, mock_test_case, mock_fetcher, mock_adapter):
        """Test that catalog YAML is saved when path is available."""
        mock_datasets = MagicMock()
        test_case_dir = tmp_path / "test-diag" / "default"

        mock_paths = MagicMock()
        mock_paths.catalog = test_case_dir / "catalog.yaml"

        with (
            patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.cli.test_cases.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=mock_paths),
            patch("climate_ref.cli.test_cases.save_datasets_to_yaml") as mock_save,
        ):
            _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        # Catalog is saved to paths.catalog
        mock_save.assert_called_once_with(mock_datasets, test_case_dir / "catalog.yaml")

    def test_obs4mips_data_returned(self, mock_diagnostic, mock_test_case):
        """Test with obs4MIPs data returns properly grouped catalog."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["obs4MIPs"],
                "source_id": ["GPCP-SG"],
                "variable_id": ["pr"],
                "path": ["/path/to/pr.nc"],
            }
        )
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["obs4MIPs.test.pr"],
                "source_id": ["GPCP-SG"],
                "variable_id": ["pr"],
                "path": ["/path/to/pr.nc"],
            }
        )
        mock_datasets = MagicMock()

        with (
            patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.cli.test_cases.Obs4MIPsDatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets

    def test_empty_data_catalog_raises_error(self, mock_diagnostic, mock_test_case):
        """Test error when data catalog is empty after building."""
        mock_fetcher = MagicMock()
        # Return data with unknown source type that won't be processed
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["unknown"],
                "path": ["/path/to/file.nc"],
            }
        )

        with patch("climate_ref.cli.test_cases.ESGFFetcher", return_value=mock_fetcher):
            with pytest.raises(DatasetResolutionError, match="No datasets found"):
                _fetch_and_build_catalog(mock_diagnostic, mock_test_case)


class TestFindDiagnostic:
    """Tests for _find_diagnostic function."""

    def test_find_existing_diagnostic(self):
        """Test finding an existing diagnostic."""
        mock_diagnostic = MagicMock(slug="test-diag")
        mock_provider = MagicMock(slug="test-provider")
        mock_provider.diagnostics.return_value = [mock_diagnostic]
        mock_registry = MagicMock(providers=[mock_provider])

        result = _find_diagnostic(mock_registry, "test-provider", "test-diag")
        assert result is mock_diagnostic

    def test_find_nonexistent_provider(self):
        """Test finding diagnostic with non-existent provider."""
        mock_provider = MagicMock(slug="other-provider")
        mock_registry = MagicMock(providers=[mock_provider])

        result = _find_diagnostic(mock_registry, "nonexistent-provider", "test-diag")
        assert result is None

    def test_find_nonexistent_diagnostic(self):
        """Test finding non-existent diagnostic in existing provider."""
        mock_diagnostic = MagicMock(slug="other-diag")
        mock_provider = MagicMock(slug="test-provider")
        mock_provider.diagnostics.return_value = [mock_diagnostic]
        mock_registry = MagicMock(providers=[mock_provider])

        result = _find_diagnostic(mock_registry, "test-provider", "nonexistent-diag")
        assert result is None


class TestFetchTestDataCommand:
    """Tests for fetch test data CLI command."""

    def test_fetch_help(self, invoke_cli):
        """Test fetch command help."""
        result = invoke_cli(["test-cases", "fetch", "--help"])
        assert "Fetch test data from ESGF" in result.stdout

    def test_fetch_dry_run(self, invoke_cli):
        """Test fetch command with dry run."""
        result = invoke_cli(["test-cases", "fetch", "--dry-run"])
        assert result.exit_code == 0

    def test_fetch_with_provider_filter(self, invoke_cli):
        """Test fetch command with provider filter."""
        result = invoke_cli(["test-cases", "fetch", "--provider", "example", "--dry-run"])
        assert result.exit_code == 0

    def test_fetch_with_diagnostic_filter(self, invoke_cli):
        """Test fetch command with diagnostic filter."""
        result = invoke_cli(["test-cases", "fetch", "--diagnostic", "nonexistent", "--dry-run"])
        assert result.exit_code == 0

    def test_fetch_with_test_case_filter(self, invoke_cli, mocker):
        """Test fetch command with test case filter in dry run."""
        mock_request = MagicMock(slug="test-request", source_type="CMIP6")
        mock_test_case = MagicMock(name="specific-case", description="test", requests=[mock_request])
        mock_spec = MagicMock(test_cases=[mock_test_case])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="test-provider")

        mock_provider = MagicMock(slug="test-provider")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(["test-cases", "fetch", "--test-case", "specific-case", "--dry-run"])
        assert result.exit_code == 0


class TestListCasesCommand:
    """Tests for list test cases CLI command."""

    def test_list_help(self, invoke_cli):
        """Test list command help."""
        result = invoke_cli(["test-cases", "list", "--help"])
        assert "List test cases" in result.stdout

    def test_list_all(self, invoke_cli):
        """Test listing all test cases."""
        result = invoke_cli(["test-cases", "list"])
        assert result.exit_code == 0
        assert "Provider" in result.stdout

    def test_list_with_provider_filter(self, invoke_cli):
        """Test listing test cases with provider filter."""
        result = invoke_cli(["test-cases", "list", "--provider", "example"])
        assert result.exit_code == 0

    def test_list_shows_no_test_data_spec(self, invoke_cli, mocker):
        """Test that list shows diagnostics without test_data_spec."""
        mock_diag = MagicMock(slug="no-spec-diag", test_data_spec=None)
        mock_provider = MagicMock(slug="test-provider")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(["test-cases", "list"])
        assert result.exit_code == 0
        assert "no test_data_spec" in result.stdout

    def test_list_filters_by_provider(self, invoke_cli, mocker):
        """Test that list command filters providers correctly."""
        mock_diag1 = MagicMock(slug="diag1", test_data_spec=None)
        mock_diag2 = MagicMock(slug="diag2", test_data_spec=None)

        mock_provider1 = MagicMock(slug="provider1")
        mock_provider1.diagnostics.return_value = [mock_diag1]
        mock_provider2 = MagicMock(slug="provider2")
        mock_provider2.diagnostics.return_value = [mock_diag2]

        mock_registry = MagicMock(providers=[mock_provider1, mock_provider2])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(["test-cases", "list", "--provider", "provider1"])
        assert result.exit_code == 0
        assert "provider1" in result.stdout
        # provider2 should be filtered out
        assert "provider2" not in result.stdout


class TestRunTestCaseCommand:
    """Tests for run test case CLI command."""

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
        mock_diag = MagicMock(test_data_spec=None)
        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diag)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_nonexistent_test_case(self, invoke_cli, mocker, mock_diagnostic):
        """Test running non-existent test case."""
        mock_diagnostic.test_data_spec.has_case.return_value = False
        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)

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

    def test_run_no_test_case_dir(self, invoke_cli, mocker, mock_diagnostic):
        """Test run command when test case directory can't be resolved."""
        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=None)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_with_fetch_flag(self, invoke_cli, mocker, mock_diagnostic):
        """Test run command with --fetch flag fetches data first."""
        mock_result = MagicMock(successful=True, metric_bundle_filename=None, output_bundle_filename=None)
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch("climate_ref.cli.test_cases._fetch_and_build_catalog", return_value=MagicMock())
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=None)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test", "--fetch"],
        )
        assert result.exit_code == 0

    def test_run_with_fetch_flag_raises_error(self, invoke_cli, mocker, mock_diagnostic):
        """Test run command with --fetch flag handles DatasetResolutionError."""
        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch(
            "climate_ref.cli.test_cases._fetch_and_build_catalog",
            side_effect=DatasetResolutionError("No datasets found"),
        )

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test", "--fetch"],
            expected_exit_code=1,
        )

    @pytest.mark.parametrize(
        "exception_cls,exception_msg",
        [
            (DatasetResolutionError, "No datasets found"),
            (NoTestDataSpecError, "No test data spec"),
            (TestCaseNotFoundError, "Test case not found"),
            (Exception, "Unexpected error"),
        ],
    )
    def test_run_handles_exceptions(
        self, invoke_cli, mocker, tmp_path, mock_diagnostic, exception_cls, exception_msg
    ):
        """Test run command handles various exceptions from runner."""
        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path

        mock_runner = MagicMock()
        mock_runner.run.side_effect = exception_cls(exception_msg)

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_successful_execution(self, invoke_cli, mocker, tmp_path, mock_diagnostic):
        """Test run command with successful execution."""
        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path
        mock_paths.regression = test_case_dir / "regression"

        mock_result = MagicMock(
            successful=True, metric_bundle_filename="metrics.json", output_bundle_filename="output.json"
        )
        mock_result.to_output_path.side_effect = lambda x: tmp_path / x
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        # First call (loading catalog), second call (regression) - both return mock_paths, then None
        mocker.patch(
            "climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", side_effect=[mock_paths, None]
        )
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
        )
        assert result.exit_code == 0

    def test_run_unsuccessful_execution(self, invoke_cli, mocker, tmp_path, mock_diagnostic):
        """Test run command with unsuccessful execution result."""
        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path

        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock(successful=False)

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=1,
        )

    def test_run_with_force_regen(self, invoke_cli, mocker, tmp_path, mock_diagnostic):
        """Test run command with force_regen regenerates baseline."""
        # Set up test case directory structure
        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        # Create a mock output directory with files
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        metrics_file = output_dir / "metrics.json"
        metrics_file.write_text('{"test": "data"}')

        # Regression dir is within the test case directory
        regression_dir = test_case_dir / "regression"

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path
        mock_paths.root = test_case_dir
        mock_paths.regression = regression_dir
        mock_paths.test_data_dir = tmp_path

        mock_definition = MagicMock()
        mock_definition.output_directory = output_dir

        mock_result = MagicMock(
            successful=True, metric_bundle_filename="metrics.json", output_bundle_filename=None
        )
        mock_result.to_output_path.return_value = metrics_file
        mock_result.definition = mock_definition
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test", "--force-regen"],
        )
        assert result.exit_code == 0

        # The full output directory is now copied to regression dir
        assert regression_dir.exists()
        assert (regression_dir / "metrics.json").exists()

    def test_run_with_existing_baseline(self, invoke_cli, mocker, tmp_path, mock_diagnostic):
        """Test run command logs when baseline exists."""
        # Set up test case directory structure with existing regression data
        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        # Create existing regression directory with baseline file
        regression_dir = test_case_dir / "regression"
        regression_dir.mkdir(parents=True)
        baseline_file = regression_dir / "metrics.json"
        baseline_file.write_text('{"existing": "baseline"}')

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path
        mock_paths.regression = regression_dir

        mock_result = MagicMock(
            successful=True, metric_bundle_filename="metrics.json", output_bundle_filename=None
        )
        mock_result.to_output_path.return_value = tmp_path / "metrics.json"
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch("climate_ref.cli.test_cases._find_diagnostic", return_value=mock_diagnostic)
        mocker.patch("climate_ref.cli.test_cases.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref.cli.test_cases.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.cli.test_cases.TestCaseRunner", return_value=mock_runner)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
        )
        assert result.exit_code == 0
        # Without --force-regen, baseline should be unchanged
        assert baseline_file.read_text() == '{"existing": "baseline"}'
