import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climate_ref.cli.test_cases import (
    _build_catalog,
    _fetch_and_build_catalog,
    _iter_test_cases,
    _solve_test_case,
)
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.provider_registry import ProviderRegistry
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)
from climate_ref_core.pycmec.output import CMECOutput


def _find_diagnostic(
    registry: ProviderRegistry, provider_slug: str, diagnostic_slug: str
) -> Diagnostic | None:
    """Find a diagnostic by provider and diagnostic slugs."""
    for provider_instance in registry.providers:
        if provider_instance.slug == provider_slug:
            for d in provider_instance.diagnostics():
                if d.slug == diagnostic_slug:
                    return d
    return None


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

    def test_successful_catalog_build(self):
        """Test successful catalog build with matching files."""
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "path": ["/path/to/tas.nc", "/path/to/pr.nc"],
                "variable_id": ["tas", "pr"],
            }
        )

        file_paths = [Path("/path/to/tas.nc")]
        result = _build_catalog(mock_adapter, file_paths)

        # Only the matching file should be in the result
        assert len(result) == 1
        assert result.iloc[0]["variable_id"] == "tas"

    def test_multiple_parent_dirs(self):
        """Test catalog build with files from multiple parent directories."""
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = [
            pd.DataFrame(
                {
                    "path": ["/dir1/tas.nc"],
                    "variable_id": ["tas"],
                }
            ),
            pd.DataFrame(
                {
                    "path": ["/dir2/pr.nc"],
                    "variable_id": ["pr"],
                }
            ),
        ]

        file_paths = [Path("/dir1/tas.nc"), Path("/dir2/pr.nc")]
        result = _build_catalog(mock_adapter, file_paths)

        assert len(result) == 2

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
        """Failures in every parent_dir are logged and yield an empty DataFrame."""
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = Exception("Parse error")

        file_paths = [Path("/path/to/file.nc")]
        result = _build_catalog(mock_adapter, file_paths)

        assert result.empty
        assert "Failed to parse" in caplog.text


class TestFetchAndBuildCatalog:
    """Tests for _fetch_and_build_catalog function."""

    def test_no_requests_raises_error(self, mock_diagnostic, mock_test_case):
        """Test with no requests raises DatasetResolutionError."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame()

        with patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher):
            with pytest.raises(DatasetResolutionError):
                _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

    def test_cmip6_data_returned(self, mock_diagnostic, mock_test_case, mock_fetcher, mock_adapter):
        """Test with CMIP6 data returns properly grouped catalog."""
        mock_datasets = MagicMock()

        with (
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result, written = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets
        assert written is False  # No paths means no catalog written

    def test_saves_catalog_yaml(self, tmp_path, mock_diagnostic, mock_test_case, mock_fetcher, mock_adapter):
        """Test that catalog YAML is saved when path is available."""
        mock_datasets = MagicMock()
        test_case_dir = tmp_path / "test-diag" / "default"

        mock_paths = MagicMock()
        mock_paths.catalog = test_case_dir / "catalog.yaml"

        with (
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths),
            patch("climate_ref_core.testing.save_datasets_to_yaml", return_value=True) as mock_save,
        ):
            _, written = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        # Catalog is saved to paths.catalog with force=False by default
        mock_save.assert_called_once_with(mock_datasets, test_case_dir / "catalog.yaml", force=False)
        assert written is True

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
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.Obs4MIPsDatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result, written = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets
        assert written is False  # No paths means no catalog written

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

        with patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher):
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

    @pytest.mark.parametrize(
        "exception",
        [
            DatasetResolutionError("No datasets found for tc1"),
            ValueError("No valid executions found for diagnostic test-diag"),
        ],
    )
    def test_fetch_continues_on_test_case_failure(self, invoke_cli, mocker, exception):
        """A per-test-case failure is logged and does not abort the loop.

        Regression test for the cascade where a single bad parent_dir caused
        ``pd.concat([])`` (or an empty solver result) to crash the entire
        ``ref test-cases fetch`` command instead of moving on to the next
        test case.
        """
        tc_bad = MagicMock(name="bad", description="bad", requests=[MagicMock()])
        tc_bad.name = "bad"
        tc_good = MagicMock(name="good", description="good", requests=[MagicMock()])
        tc_good.name = "good"
        mock_spec = MagicMock(test_cases=[tc_bad, tc_good])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")

        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        fetch_mock = mocker.patch(
            "climate_ref.cli.test_cases._fetch_and_build_catalog",
            side_effect=[exception, (MagicMock(), True)],
        )

        result = invoke_cli(["test-cases", "fetch"])

        assert result.exit_code == 0
        assert fetch_mock.call_count == 2


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
        assert "Run test cases for diagnostics" in result.stdout

    def test_run_nonexistent_diagnostic(self, invoke_cli):
        """Test running non-existent diagnostic."""
        invoke_cli(
            ["test-cases", "run", "--provider", "nonexistent", "--diagnostic", "fake"],
            expected_exit_code=1,
        )

    def test_run_diagnostic_no_test_data_spec(self, invoke_cli, mocker):
        """Test running diagnostic without test_data_spec shows warning and exits 0."""
        mock_diag = MagicMock(slug="test", test_data_spec=None)
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        # No test cases found, but exit 0 (not an error)
        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test"],
            expected_exit_code=0,
        )
        assert "No test cases found" in result.stderr

    def test_run_nonexistent_test_case(self, invoke_cli, mocker):
        """Test running non-existent test case shows warning and exits 0."""
        mock_tc = MagicMock(name="default", description="test")
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        # Filtering for nonexistent test case finds nothing, exit 0
        result = invoke_cli(
            [
                "test-cases",
                "run",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
                "--test-case",
                "nonexistent",
            ],
            expected_exit_code=0,
        )
        assert "No test cases found" in result.stderr

    def test_run_no_test_case_dir(self, invoke_cli, mocker):
        """Test run command when test case directory can't be resolved fails the test case."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None)

        # Without paths, the test case fails and we get exit code 1
        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )

    def test_run_with_fetch_flag(self, invoke_cli, mocker):
        """Test run command with --fetch flag fetches data first."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mock_result = MagicMock(successful=True, metric_bundle_filename=None, output_bundle_filename=None)
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref.cli.test_cases._fetch_and_build_catalog", return_value=(MagicMock(), True))
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag", "--fetch"],
        )
        assert result.exit_code == 0

    def test_run_with_fetch_flag_raises_error(self, invoke_cli, mocker):
        """Test run command with --fetch flag handles DatasetResolutionError."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch(
            "climate_ref.cli.test_cases._fetch_and_build_catalog",
            side_effect=DatasetResolutionError("No datasets found"),
        )

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag", "--fetch"],
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
    def test_run_handles_exceptions(self, invoke_cli, mocker, tmp_path, exception_cls, exception_msg):
        """Test run command handles various exceptions from runner."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path

        mock_runner = MagicMock()
        mock_runner.run.side_effect = exception_cls(exception_msg)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )

    def test_run_successful_execution(self, invoke_cli, mocker, tmp_path):
        """Test run command with successful execution."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        # Create existing regression dir to avoid copy being triggered
        regression_dir = test_case_dir / "regression"
        regression_dir.mkdir()

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path
        mock_paths.regression = regression_dir  # Use real dir that exists

        mock_result = MagicMock(
            successful=True, metric_bundle_filename="metrics.json", output_bundle_filename="output.json"
        )
        mock_result.to_output_path.side_effect = lambda x: tmp_path / x
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
        )
        assert result.exit_code == 0

    def test_run_unsuccessful_execution(self, invoke_cli, mocker, tmp_path):
        """Test run command with unsuccessful execution result."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path

        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock(successful=False)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )

    def test_run_with_force_regen(self, invoke_cli, mocker, tmp_path):
        """Test run command with force_regen regenerates the committed bundle and manifest."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        # Set up test case directory structure
        test_case_dir = tmp_path / "test-diag" / "default"
        test_case_dir.mkdir(parents=True)
        catalog_path = test_case_dir / "catalog.yaml"
        catalog_path.touch()

        # Create a scratch output directory holding the curated committed bundle.
        # capture_execution copies from scratch_root / fragment into a temp results dir,
        # so output_dir.parent (scratch_root) must differ from the temp dir it picks.
        scratch_root = tmp_path / "scratch"
        fragment = "frag"
        output_dir = scratch_root / fragment
        output_dir.mkdir(parents=True)
        (output_dir / "diagnostic.json").write_text('{"test": "data"}')
        (output_dir / "output.json").write_text(json.dumps(CMECOutput.create_template()))
        (output_dir / "series.json").write_text("[]")

        # Regression dir + manifest live within the test case directory.
        regression_dir = test_case_dir / "regression"

        from climate_ref_core.testing import TestCasePaths

        real_paths = TestCasePaths(root=test_case_dir)

        mock_definition = MagicMock()
        mock_definition.output_directory = output_dir
        mock_definition.output_fragment.return_value = Path(fragment)

        mock_result = MagicMock(
            successful=True,
            metric_bundle_filename=Path("diagnostic.json"),
            output_bundle_filename=Path("output.json"),
            series_filename=Path("series.json"),
        )
        mock_result.to_output_path.side_effect = lambda f: output_dir / f
        mock_result.definition = mock_definition
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=real_paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value="abc123")

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag", "--force-regen"],
        )
        assert result.exit_code == 0

        # The committed CMEC bundle (not the full output dir) is written to regression.
        assert regression_dir.exists()
        assert (regression_dir / "diagnostic.json").exists()
        assert (regression_dir / "output.json").exists()
        assert (regression_dir / "series.json").exists()

        # A seeded manifest exists with the native block left untouched (mint-owned).
        from climate_ref_core.regression.manifest import Manifest

        manifest = Manifest.load(real_paths.manifest)
        assert manifest.test_case_version == 1
        assert manifest.native == {}
        assert set(manifest.committed) == {"diagnostic.json", "output.json", "series.json"}

    def test_run_with_existing_baseline(self, invoke_cli, mocker, tmp_path):
        """Test run command logs when baseline exists."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

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

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
        )
        assert result.exit_code == 0
        # Without --force-regen, baseline should be unchanged
        assert baseline_file.read_text() == '{"existing": "baseline"}'

    def test_run_with_only_missing_skips_existing(self, invoke_cli, mocker, tmp_path):
        """Test run command with --only-missing skips test cases with existing regression data."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        # Set up paths with existing regression data
        regression_dir = tmp_path / "regression"
        regression_dir.mkdir()

        mock_paths = MagicMock()
        mock_paths.regression = regression_dir  # exists() returns True

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)

        # With --only-missing, test case should be skipped and exit 0
        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--only-missing"],
            expected_exit_code=0,
        )
        assert "No test cases found" in result.stderr or "skipped" in result.stderr.lower()

    def test_run_with_if_changed_skips_unchanged(self, invoke_cli, mocker, tmp_path):
        """Test run command with --if-changed skips test cases with unchanged catalogs."""
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "test"
        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mock_paths = MagicMock()

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)
        # Catalog not changed since regression
        mocker.patch("climate_ref_core.testing.catalog_changed_since_regression", return_value=False)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--if-changed"],
            expected_exit_code=0,
        )
        assert "No test cases found" in result.stderr


class TestFetchAndBuildCatalogSourceTypes:
    """Tests for _fetch_and_build_catalog with different source types."""

    def test_cmip7_data_returned(self, mock_diagnostic, mock_test_case):
        """Test with CMIP7 data returns properly grouped catalog."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["CMIP7"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "path": ["/path/to/tas.nc"],
            }
        )
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["CMIP7.test.tas"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "path": ["/path/to/tas.nc"],
            }
        )
        mock_datasets = MagicMock()

        with (
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.CMIP7DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result, written = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets
        assert written is False

    def test_pmp_climatology_data_returned(self, mock_diagnostic, mock_test_case):
        """Test with PMPClimatology data returns properly grouped catalog."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["PMPClimatology"],
                "source_id": ["obs-clim"],
                "variable_id": ["pr"],
                "path": ["/path/to/pr.nc"],
            }
        )
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["PMPClim.test.pr"],
                "source_id": ["obs-clim"],
                "variable_id": ["pr"],
                "path": ["/path/to/pr.nc"],
            }
        )
        mock_datasets = MagicMock()

        with (
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.PMPClimatologyDatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result, written = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets
        assert written is False

    def test_force_flag_passed_to_save(self, tmp_path, mock_diagnostic, mock_test_case):
        """Test that force=True is passed to save_datasets_to_yaml."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["CMIP6"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "path": ["/path/to/tas.nc"],
            }
        )
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["CMIP6.test.tas"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "path": ["/path/to/tas.nc"],
            }
        )
        mock_datasets = MagicMock()
        mock_paths = MagicMock()
        mock_paths.catalog = tmp_path / "catalog.yaml"

        with (
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.CMIP6DatasetAdapter", return_value=mock_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths),
            patch("climate_ref_core.testing.save_datasets_to_yaml", return_value=True) as mock_save,
        ):
            _fetch_and_build_catalog(mock_diagnostic, mock_test_case, force=True)

        mock_save.assert_called_once_with(mock_datasets, mock_paths.catalog, force=True)

    def test_mixed_source_types(self, mock_diagnostic, mock_test_case):
        """Test with mixed CMIP6 and obs4MIPs data."""
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_for_test_case.return_value = pd.DataFrame(
            {
                "source_type": ["CMIP6", "obs4MIPs"],
                "source_id": ["ACCESS-ESM1-5", "ERA-5"],
                "variable_id": ["tas", "tas"],
                "path": ["/path/to/cmip6_tas.nc", "/path/to/obs_tas.nc"],
            }
        )
        mock_cmip6_adapter = MagicMock()
        mock_cmip6_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["CMIP6.test.tas"],
                "source_id": ["ACCESS-ESM1-5"],
                "variable_id": ["tas"],
                "path": ["/path/to/cmip6_tas.nc"],
            }
        )
        mock_obs_adapter = MagicMock()
        mock_obs_adapter.find_local_datasets.return_value = pd.DataFrame(
            {
                "instance_id": ["obs4MIPs.test.tas"],
                "source_id": ["ERA-5"],
                "variable_id": ["tas"],
                "path": ["/path/to/obs_tas.nc"],
            }
        )
        mock_datasets = MagicMock()

        with (
            patch("climate_ref_core.esgf.ESGFFetcher", return_value=mock_fetcher),
            patch("climate_ref.datasets.CMIP6DatasetAdapter", return_value=mock_cmip6_adapter),
            patch("climate_ref.datasets.Obs4MIPsDatasetAdapter", return_value=mock_obs_adapter),
            patch("climate_ref.cli.test_cases._solve_test_case", return_value=mock_datasets),
            patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None),
        ):
            result, _written = _fetch_and_build_catalog(mock_diagnostic, mock_test_case)

        assert result is mock_datasets


class TestSolveTestCase:
    """Tests for _solve_test_case function."""

    def test_solve_returns_first_execution(self):
        """Test that _solve_test_case returns datasets from the first execution."""
        mock_datasets = MagicMock()
        mock_execution = MagicMock()
        mock_execution.datasets = mock_datasets

        mock_diag = MagicMock()
        mock_diag.slug = "test-diag"
        mock_diag.provider = MagicMock(slug="test-provider")

        with patch(
            "climate_ref.solver.solve_executions",
            return_value=iter([mock_execution]),
        ):
            result = _solve_test_case(mock_diag, {})

        assert result is mock_datasets

    def test_solve_no_executions_raises_error(self):
        """Test that _solve_test_case raises ValueError when no executions found."""
        mock_diag = MagicMock()
        mock_diag.slug = "test-diag"
        mock_diag.provider = MagicMock(slug="test-provider")

        with patch(
            "climate_ref.solver.solve_executions",
            return_value=iter([]),
        ):
            with pytest.raises(ValueError, match="No valid executions found"):
                _solve_test_case(mock_diag, {})


class TestFetchTestDataCommandEdgeCases:
    """Additional edge case tests for fetch test data CLI command."""

    def test_fetch_nonexistent_provider(self, invoke_cli):
        """Test fetch command with non-existent provider exits with error."""
        result = invoke_cli(
            ["test-cases", "fetch", "--provider", "nonexistent"],
            expected_exit_code=1,
        )
        assert "not configured" in result.stderr

    def test_fetch_no_diagnostics_with_spec(self, invoke_cli, mocker):
        """Test fetch command when no diagnostics have test_data_spec."""
        mock_diag = MagicMock(slug="no-spec", test_data_spec=None)
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(
            ["test-cases", "fetch", "--provider", "example"],
            expected_exit_code=0,
        )
        assert "No diagnostics with test_data_spec found" in result.stderr


class TestListCasesCommandEdgeCases:
    """Additional edge case tests for list test cases CLI command."""

    def test_list_nonexistent_provider(self, invoke_cli):
        """Test list command with non-existent provider exits with error."""
        invoke_cli(
            ["test-cases", "list", "--provider", "nonexistent"],
            expected_exit_code=1,
        )

    def test_list_with_test_cases_and_paths(self, invoke_cli, mocker, tmp_path):
        """Test list shows catalog and regression status correctly."""
        # Create a test case with paths
        catalog_path = tmp_path / "catalog.yaml"
        catalog_path.touch()
        regression_path = tmp_path / "regression"
        # Don't create regression_path - simulates missing regression data

        mock_paths = MagicMock()
        mock_paths.catalog = catalog_path
        mock_paths.regression = regression_path

        mock_request = MagicMock(slug="req1", source_type="CMIP6")

        # Use a real object for the test case to avoid MagicMock rendering issues
        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_tc.description = "A test case"
        mock_tc.requests = [mock_request]

        mock_spec = MagicMock()
        mock_spec.test_cases = [mock_tc]

        mock_diag = MagicMock()
        mock_diag.slug = "test-diag"
        mock_diag.test_data_spec = mock_spec

        mock_provider = MagicMock()
        mock_provider.slug = "test-provider"
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=mock_paths)

        result = invoke_cli(["test-cases", "list"])
        assert result.exit_code == 0
        assert "test-provider" in result.stdout
        assert "test-diag" in result.stdout


class TestIterTestCases:
    """Tests for the _iter_test_cases helper."""

    def _registry(self):
        mock_tc_a = MagicMock()
        mock_tc_a.name = "default"
        mock_tc_b = MagicMock()
        mock_tc_b.name = "short"
        spec = MagicMock()
        spec.test_cases = [mock_tc_a, mock_tc_b]
        diag = MagicMock(slug="diag-1", test_data_spec=spec)
        provider = MagicMock(slug="example")
        provider.diagnostics.return_value = [diag]
        return MagicMock(providers=[provider]), diag, mock_tc_a, mock_tc_b

    def test_yields_all_cases(self):
        registry, diag, tc_a, tc_b = self._registry()
        result = list(_iter_test_cases(registry))
        assert result == [(diag, tc_a), (diag, tc_b)]

    def test_filters_by_test_case(self):
        registry, diag, tc_a, _ = self._registry()
        result = list(_iter_test_cases(registry, test_case="default"))
        assert result == [(diag, tc_a)]

    def test_skips_diag_without_spec(self):
        diag = MagicMock(slug="diag-1", test_data_spec=None)
        provider = MagicMock(slug="example")
        provider.diagnostics.return_value = [diag]
        registry = MagicMock(providers=[provider])
        assert list(_iter_test_cases(registry)) == []


def _make_case_mocks():
    """Build a (registry, diag, tc) trio with realistic slugs for the native verbs."""
    tc = MagicMock()
    tc.name = "default"
    tc.description = "test"
    spec = MagicMock()
    spec.test_cases = [tc]
    diag = MagicMock(slug="test-diag", test_data_spec=spec)
    diag.provider = MagicMock(slug="example")
    provider = MagicMock(slug="example")
    provider.diagnostics.return_value = [diag]
    registry = MagicMock(providers=[provider])
    return registry, diag, tc


class TestSyncCommand:
    """Tests for the `test-cases sync` CLI verb."""

    def test_sync_help(self, invoke_cli):
        result = invoke_cli(["test-cases", "sync", "--help"])
        assert "native baseline blobs" in result.stdout

    def test_sync_fetches_missing_blob(self, invoke_cli, mocker, tmp_path):
        from climate_ref_core.regression.manifest import Manifest, NativeEntry
        from climate_ref_core.regression.store import LocalFilesystemStore
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        # A populated store + a manifest referencing one of its blobs.
        store = LocalFilesystemStore(root=tmp_path / "store")
        blob = tmp_path / "blob.nc"
        blob.write_bytes(b"native-data")
        digest = store.put(blob)

        case_dir = tmp_path / "td" / "test-diag" / "default"
        case_dir.mkdir(parents=True)
        paths = TestCasePaths(root=case_dir)
        Manifest(
            schema=1,
            test_case_version=1,
            committed={},
            native={"out.nc": NativeEntry(sha256=digest, size=len(b"native-data"))},
        ).dump(paths.manifest)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        result = invoke_cli(["test-cases", "sync", "--provider", "example"])
        assert result.exit_code == 0

    def test_sync_hard_fails_on_unservable_blob(self, invoke_cli, mocker, tmp_path):
        from climate_ref_core.regression.manifest import Manifest, NativeEntry
        from climate_ref_core.regression.store import LocalFilesystemStore
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        store = LocalFilesystemStore(root=tmp_path / "store")  # empty store
        case_dir = tmp_path / "td" / "test-diag" / "default"
        case_dir.mkdir(parents=True)
        paths = TestCasePaths(root=case_dir)
        Manifest(
            schema=1,
            test_case_version=1,
            committed={},
            native={"out.nc": NativeEntry(sha256="ab" * 32, size=1)},
        ).dump(paths.manifest)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        invoke_cli(["test-cases", "sync", "--provider", "example"], expected_exit_code=1)


class TestReplayCommand:
    """Tests for the `test-cases replay` CLI verb."""

    def test_replay_help(self, invoke_cli):
        result = invoke_cli(["test-cases", "replay", "--help"])
        assert "Replay committed baselines" in result.stdout

    def test_replay_empty_native_is_hard_failure(self, invoke_cli, mocker, tmp_path):
        from climate_ref_core.regression.manifest import Manifest
        from climate_ref_core.regression.store import LocalFilesystemStore
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        case_dir = tmp_path / "td" / "test-diag" / "default"
        regression_dir = case_dir / "regression"
        regression_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        # Committed bundle present and consistent with the manifest digests.
        from climate_ref_core.regression.manifest import compute_committed_digests

        (regression_dir / "diagnostic.json").write_text("{}")
        digests = compute_committed_digests(regression_dir)
        paths = TestCasePaths(root=case_dir)
        Manifest(schema=1, test_case_version=1, committed=digests, native={}).dump(paths.manifest)

        store = LocalFilesystemStore(root=tmp_path / "store")
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        result = invoke_cli(
            ["test-cases", "replay", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )
        assert "not yet minted" in result.stderr.lower() or "mint" in result.stderr.lower()

    def test_replay_integrity_mismatch_warns_and_continues(self, invoke_cli, mocker, tmp_path):
        """An integrity mismatch is advisory, not a gate."""
        from climate_ref_core.regression.manifest import Manifest
        from climate_ref_core.regression.store import LocalFilesystemStore
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        case_dir = tmp_path / "td" / "test-diag" / "default"
        regression_dir = case_dir / "regression"
        regression_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        (regression_dir / "diagnostic.json").write_text("{}")
        paths = TestCasePaths(root=case_dir)
        # Committed digest deliberately wrong -> integrity mismatch (now a warning, not a gate).
        Manifest(
            schema=1,
            test_case_version=1,
            committed={"diagnostic.json": "00" * 32},
            native={},
        ).dump(paths.manifest)

        store = LocalFilesystemStore(root=tmp_path / "store")
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        result = invoke_cli(
            ["test-cases", "replay", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )
        # The mismatch warned rather than gated, and execution fell through to the native guard.
        assert "differs from the digests recorded" in result.stderr
        assert "not yet minted" in result.stderr.lower()

    def test_replay_reconciles_integrity_mismatch_within_tolerance(self, invoke_cli, mocker, tmp_path):
        """A byte-level baseline difference forgiven by the tolerant comparison is reported as reconciled.

        This proves the follow-up comparison actually ran after the integrity warning: the success line
        names the bundle files compared and states they are equivalent within tolerance, rather than
        silently claiming a match.
        """
        from climate_ref_core.regression.manifest import Manifest, NativeEntry
        from climate_ref_core.regression.store import LocalFilesystemStore
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        case_dir = tmp_path / "td" / "test-diag" / "default"
        regression_dir = case_dir / "regression"
        regression_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        for name in ("series.json", "diagnostic.json", "output.json"):
            (regression_dir / name).write_text("{}")
        paths = TestCasePaths(root=case_dir)
        # Wrong committed digest -> integrity warns; native present -> passes the mint guard.
        Manifest(
            schema=1,
            test_case_version=1,
            committed={"series.json": "00" * 32},
            native={"out.nc": NativeEntry(sha256="ab" * 32, size=1)},
        ).dump(paths.manifest)

        store = LocalFilesystemStore(root=tmp_path / "store")
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)
        # Stub the heavy replay internals so we deterministically reach the success branch:
        # native materialisation, placeholder expansion, dataset load, definition, and the tolerant
        # comparison itself (a no-op assert == "equivalent within tolerance").
        mocker.patch("climate_ref_core.regression.materialise_native")
        mocker.patch("climate_ref_core.output_files.from_placeholders")
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=[])
        mocker.patch("climate_ref_core.diagnostics.ExecutionDefinition")
        mocker.patch("climate_ref_core.regression.assert_bundle_regression")

        result = invoke_cli(["test-cases", "replay", "--provider", "example", "--diagnostic", "test-diag"])
        assert "Replay reconciled committed bundle" in result.stderr
        assert "equivalent within tolerance" in result.stderr
        assert "3 bundle file(s)" in result.stderr


class TestMintCommand:
    """Tests for the `test-cases mint` CLI verb."""

    def test_mint_help(self, invoke_cli):
        result = invoke_cli(["test-cases", "mint", "--help"])
        assert "Mint canonical native baselines" in result.stdout

    def test_mint_refuses_without_writable_store(self, invoke_cli, mocker, tmp_path):
        registry, _diag, _tc = _make_case_mocks()

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            side_effect=NotImplementedError("R2 backend deferred"),
        )

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "example"],
            expected_exit_code=1,
        )
        assert "Cannot mint" in result.stderr

    def test_mint_writes_blobs_and_manifest(self, invoke_cli, mocker, tmp_path):
        from climate_ref_core.regression.manifest import Manifest
        from climate_ref_core.regression.store import LocalFilesystemStore
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        case_dir = tmp_path / "td" / "test-diag" / "default"
        case_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        paths = TestCasePaths(root=case_dir)

        # A scratch execution holding the curated bundle + a native data file.
        scratch_root = tmp_path / "scratch"
        fragment = "frag"
        output_dir = scratch_root / fragment
        output_dir.mkdir(parents=True)
        (output_dir / "diagnostic.json").write_text("{}")
        out_bundle = CMECOutput.create_template()
        (output_dir / "output.json").write_text(json.dumps(out_bundle))
        (output_dir / "series.json").write_text("[]")

        defn = MagicMock()
        defn.output_directory = output_dir
        defn.output_fragment.return_value = Path(fragment)
        result_obj = MagicMock(
            successful=True,
            metric_bundle_filename=Path("diagnostic.json"),
            output_bundle_filename=Path("output.json"),
            series_filename=Path("series.json"),
        )
        result_obj.definition = defn
        runner = MagicMock()
        runner.run.return_value = result_obj

        store = LocalFilesystemStore(root=tmp_path / "store")

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=runner)
        mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value=None)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        result = invoke_cli(["test-cases", "mint", "--provider", "example"])
        assert result.exit_code == 0

        manifest = Manifest.load(paths.manifest)
        assert manifest.test_case_version == 1
        # mint authored the native block from the captured snapshot.
        assert set(manifest.native) == {"diagnostic.json", "output.json", "series.json"}
        # Each native blob was PUT into the store.
        for entry in manifest.native.values():
            assert store.has(entry.sha256)
