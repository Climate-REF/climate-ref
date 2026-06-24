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

    def test_adapter_exception_raises(self):
        """A parse failure in any parent_dir aborts with DatasetResolutionError (fail-fast)."""
        mock_adapter = MagicMock()
        mock_adapter.find_local_datasets.side_effect = Exception("Parse error")

        file_paths = [Path("/path/to/file.nc")]
        with pytest.raises(DatasetResolutionError, match="Failed to parse fetched datasets"):
            _build_catalog(mock_adapter, file_paths)


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
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=None,
            ),
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
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=mock_paths,
            ),
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
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=None,
            ),
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

    def test_fetch_with_unknown_diagnostic_filter_fails(self, invoke_cli):
        """Test fetch command with a diagnostic typo fails instead of silently selecting nothing."""
        result = invoke_cli(
            ["test-cases", "fetch", "--diagnostic", "nonexistent", "--dry-run"],
            expected_exit_code=1,
        )
        assert "Diagnostic 'nonexistent' was not found" in result.stderr

    def test_fetch_with_unknown_test_case_filter_fails(self, invoke_cli, mocker):
        """Test fetch command with a test-case typo fails instead of silently selecting nothing."""
        mock_tc = MagicMock(description="test", requests=[])
        mock_tc.name = "default"
        mock_spec = MagicMock(test_cases=[mock_tc])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(
            ["test-cases", "fetch", "--provider", "example", "--test-case", "missing", "--dry-run"],
            expected_exit_code=1,
        )
        assert "Test case 'missing' was not found" in result.stderr

    def test_fetch_with_test_case_filter(self, invoke_cli, mocker):
        """Test fetch command with test case filter in dry run."""
        mock_request = MagicMock(slug="test-request", source_type="CMIP6")
        mock_test_case = MagicMock(description="test", requests=[mock_request])
        mock_test_case.name = "specific-case"
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
            "climate_ref.cli.test_cases.discovery._fetch_and_build_catalog",
            side_effect=[exception, (MagicMock(), True)],
        )

        result = invoke_cli(["test-cases", "fetch"])

        assert result.exit_code == 0
        assert fetch_mock.call_count == 2

    def test_fetch_existing_catalog_explains_refresh(self, invoke_cli, mocker, tmp_path):
        """Existing catalogs are refreshed by default, with a pointer to --only-missing."""
        mock_tc = MagicMock(description="test", requests=[MagicMock()])
        mock_tc.name = "default"
        mock_spec = MagicMock(test_cases=[mock_tc])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        catalog = case_dir / "catalog.yaml"
        catalog.touch()
        paths = MagicMock(catalog=catalog)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        fetch_mock = mocker.patch(
            "climate_ref.cli.test_cases.discovery._fetch_and_build_catalog",
            return_value=(MagicMock(), False),
        )

        result = invoke_cli(["test-cases", "fetch", "--provider", "example"])

        assert result.exit_code == 0
        assert "Refreshing existing catalog for default" in result.stderr
        assert "use --only-missing to skip existing catalogs" in result.stderr
        fetch_mock.assert_called_once()

    def test_fetch_only_missing_skips_existing_catalog(self, invoke_cli, mocker, tmp_path):
        """--only-missing avoids ESGF work when a catalog already exists."""
        mock_tc = MagicMock(description="test", requests=[MagicMock()])
        mock_tc.name = "default"
        mock_spec = MagicMock(test_cases=[mock_tc])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        case_dir = tmp_path / "case"
        case_dir.mkdir()
        catalog = case_dir / "catalog.yaml"
        catalog.touch()
        paths = MagicMock(catalog=catalog)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        fetch_mock = mocker.patch("climate_ref.cli.test_cases.discovery._fetch_and_build_catalog")

        result = invoke_cli(["test-cases", "fetch", "--provider", "example", "--only-missing"])

        assert result.exit_code == 0
        assert "Skipping test case: default (catalog exists)" in result.stderr
        fetch_mock.assert_not_called()


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

    def test_list_does_not_configure_providers(self, invoke_cli, mocker):
        """Listing metadata avoids provider configure hooks with setup side effects."""
        mock_registry = MagicMock(providers=[])
        build_registry = mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )

        result = invoke_cli(["test-cases", "list"])

        assert result.exit_code == 0
        assert build_registry.call_args.kwargs["configure"] is False

    def test_list_compact_default_hides_descriptions(self, invoke_cli, mocker):
        """The default list view stays scannable by omitting descriptions."""
        mock_tc = MagicMock(description="long description", requests=[])
        mock_tc.name = "default"
        mock_spec = MagicMock(test_cases=[mock_tc])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None)

        result = invoke_cli(["test-cases", "list"])

        assert result.exit_code == 0
        assert "long description" not in result.stdout
        assert "test-diag" in result.stdout

    def test_list_summary_counts_missing_artifacts(self, invoke_cli, mocker, tmp_path):
        """The list summary reports missing catalog and regression counts."""
        from climate_ref_core.testing import TestCasePaths

        tc_catalog_only = MagicMock(description="catalog", requests=[])
        tc_catalog_only.name = "catalog-only"
        tc_regression_only = MagicMock(description="regression", requests=[])
        tc_regression_only.name = "regression-only"
        mock_spec = MagicMock(test_cases=[tc_catalog_only, tc_regression_only])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        catalog_only = TestCasePaths(root=tmp_path / "catalog-only")
        catalog_only.root.mkdir()
        catalog_only.catalog.touch()
        regression_only = TestCasePaths(root=tmp_path / "regression-only")
        regression_only.root.mkdir()
        regression_only.regression.mkdir()

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch(
            "climate_ref_core.testing.TestCasePaths.from_diagnostic",
            side_effect=[catalog_only, regression_only],
        )

        result = invoke_cli(["test-cases", "list"])

        assert result.exit_code == 0
        assert "2 test case(s); 1 missing catalog(s); 1 missing regression baseline(s)" in result.stdout

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

        result = invoke_cli(["test-cases", "list", "--verbose"])
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
        assert "Scratch directory" in result.stdout
        assert "output/<label>" in result.stdout

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
        """Test running a non-existent test case fails instead of silently selecting nothing."""
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
            expected_exit_code=1,
        )
        assert "Test case 'nonexistent' was not found" in result.stderr

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

    def test_run_with_fetch_flag(self, invoke_cli, mocker, tmp_path):
        """`run --fetch` fetches data, executes, and materialises the output slot (exit 0)."""
        paths, _scratch, _regression, _runner = _setup_real_run(mocker, tmp_path, fetch=True)

        result = invoke_cli(
            [
                "test-cases",
                "run",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
                "--fetch",
            ],
        )
        assert result.exit_code == 0
        # The slot was populated with the executed native set.
        assert (paths.output_slot("latest") / "diagnostic.json").exists()

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
            "climate_ref.cli.test_cases.run._fetch_and_build_catalog",
            side_effect=DatasetResolutionError("No datasets found"),
        )

        invoke_cli(
            [
                "test-cases",
                "run",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
                "--fetch",
            ],
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
        """A diagnostic execution exception is caught and fails the test case (exit 1)."""
        _setup_real_run(mocker, tmp_path, runner_result=exception_cls(exception_msg))

        invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )

    def test_run_successful_execution(self, invoke_cli, mocker, tmp_path):
        """A successful execution writes the slot and exits 0 (baseline already present)."""
        paths, _scratch, _regression, _runner = _setup_real_run(mocker, tmp_path, regression_files={})

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
        )
        assert result.exit_code == 0
        # The slot holds the rebuilt committed bundle even when the baseline is untouched.
        assert (paths.output_slot("latest") / "regression" / "diagnostic.json").exists()

    def test_run_output_directory_logs_scratch_and_slot_semantics(self, invoke_cli, mocker, tmp_path):
        """--output-directory is explained as scratch output, not the only written location."""
        paths, _scratch, _regression, _runner = _setup_real_run(mocker, tmp_path, regression_files={})
        scratch = tmp_path / "scratch-output"

        result = invoke_cli(
            [
                "test-cases",
                "run",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
                "--output-directory",
                str(scratch),
            ],
        )

        assert result.exit_code == 0
        assert f"Using {scratch} as the execution scratch directory" in result.stderr
        assert "gitignored output/latest slot" in result.stderr
        assert (paths.output_slot("latest") / "regression" / "diagnostic.json").exists()

    def test_run_unsuccessful_execution(self, invoke_cli, mocker, tmp_path):
        """An unsuccessful execution result fails the test case (exit 1)."""
        _setup_real_run(mocker, tmp_path, runner_result="unsuccessful")

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
        mocker.patch(
            "climate_ref_core.testing.TestCasePaths.from_diagnostic",
            return_value=real_paths,
        )
        mocker.patch("climate_ref_core.testing.load_datasets_from_yaml", return_value=MagicMock())
        mocker.patch("climate_ref.testing.TestCaseRunner", return_value=mock_runner)
        mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value="abc123")

        result = invoke_cli(
            [
                "test-cases",
                "run",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
                "--force-regen",
            ],
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
        assert set(manifest.committed) == {
            "diagnostic.json",
            "output.json",
            "series.json",
        }

    def test_run_with_existing_baseline(self, invoke_cli, mocker, tmp_path):
        """Without --force-regen, an existing baseline is left untouched (promotion gate)."""
        _paths, _scratch, regression_dir, _runner = _setup_real_run(
            mocker, tmp_path, regression_files={"metrics.json": '{"existing": "baseline"}'}
        )
        baseline_file = regression_dir / "metrics.json"

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag"],
        )
        assert result.exit_code == 0
        # Without --force-regen, the committed baseline is not promoted.
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
        mocker.patch(
            "climate_ref_core.testing.TestCasePaths.from_diagnostic",
            return_value=mock_paths,
        )

        # With --only-missing, test case should be skipped and exit 0
        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--only-missing"],
            expected_exit_code=0,
        )
        assert "All 1 matching test case(s) skipped" in result.stderr
        assert "No test cases found" not in result.stderr

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
        mocker.patch(
            "climate_ref_core.testing.TestCasePaths.from_diagnostic",
            return_value=mock_paths,
        )
        # Catalog not changed since regression
        mocker.patch(
            "climate_ref_core.testing.catalog_changed_since_regression",
            return_value=False,
        )

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--if-changed"],
            expected_exit_code=0,
        )
        assert "All 1 matching test case(s) skipped because catalogs are unchanged" in result.stderr

    def test_run_label_slots_coexist(self, invoke_cli, mocker, tmp_path):
        """Two runs under different --label slots persist side by side."""
        paths, _scratch, _regression, _runner = _setup_real_run(mocker, tmp_path)

        for label in ("before", "after"):
            assert (
                invoke_cli(
                    [
                        "test-cases",
                        "run",
                        "--provider",
                        "example",
                        "--diagnostic",
                        "test-diag",
                        "--label",
                        label,
                        "--force-regen",
                    ]
                ).exit_code
                == 0
            )

        assert (paths.output_slot("before") / "diagnostic.json").exists()
        assert (paths.output_slot("after") / "diagnostic.json").exists()
        # The default slot was never written.
        assert not paths.output_slot("latest").exists()

    def test_run_force_regen_promotes_over_existing_baseline(self, invoke_cli, mocker, tmp_path):
        """--force-regen replaces an existing committed baseline with the freshly built bundle."""
        _paths, _scratch, regression_dir, _runner = _setup_real_run(
            mocker, tmp_path, regression_files={"diagnostic.json": '{"old": "baseline"}'}
        )

        assert (
            invoke_cli(
                ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag", "--force-regen"]
            ).exit_code
            == 0
        )
        # The stale baseline was promoted over by the rebuilt bundle.
        assert (regression_dir / "diagnostic.json").read_text() != '{"old": "baseline"}'

    def test_run_warns_when_native_stale(self, invoke_cli, mocker, tmp_path):
        """force-regen warns (does not block) when the rebuilt native differs from the minted baseline."""
        from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry

        paths, _scratch, _regression, _runner = _setup_real_run(mocker, tmp_path, regression_files={})
        # A previously minted manifest whose native digests will not match the freshly built slot.
        Manifest(
            schema=SCHEMA_VERSION,
            test_case_version=3,
            committed={},
            native={"diagnostic.json": NativeEntry(sha256="00" * 32, size=1)},
        ).dump(paths.manifest)

        result = invoke_cli(
            ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag", "--force-regen"]
        )
        assert result.exit_code == 0
        assert "native baseline differs" in result.stderr
        # run preserves the mint-owned native block and version (it never authors native).
        manifest = Manifest.load(paths.manifest)
        assert manifest.test_case_version == 3
        assert set(manifest.native) == {"diagnostic.json"}


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
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=None,
            ),
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
            patch(
                "climate_ref.datasets.PMPClimatologyDatasetAdapter",
                return_value=mock_adapter,
            ),
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=None,
            ),
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
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=mock_paths,
            ),
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
            patch(
                "climate_ref.datasets.CMIP6DatasetAdapter",
                return_value=mock_cmip6_adapter,
            ),
            patch(
                "climate_ref.datasets.Obs4MIPsDatasetAdapter",
                return_value=mock_obs_adapter,
            ),
            patch(
                "climate_ref.cli.test_cases._catalog._solve_test_case",
                return_value=mock_datasets,
            ),
            patch(
                "climate_ref_core.testing.TestCasePaths.from_diagnostic",
                return_value=None,
            ),
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
        mocker.patch(
            "climate_ref_core.testing.TestCasePaths.from_diagnostic",
            return_value=mock_paths,
        )

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


def _setup_real_run(mocker, tmp_path, *, runner_result="success", regression_files=None, fetch=False):
    """
    Wire a ``run``/``build`` test against a real ``TestCasePaths`` and scratch execution dir.

    The slot-based ``run`` materialises ``output/<label>/`` on every invocation, so it needs
    real on-disk paths (a ``MagicMock`` slot cannot be wiped or recreated). This builds a real
    test case directory plus a scratch output dir holding a curated CMEC bundle, and a mock
    runner whose result points at that scratch dir.

    Parameters
    ----------
    runner_result
        ``"success"`` (default) for a successful result, ``"unsuccessful"`` for
        ``successful=False``, or an ``Exception`` instance for the runner to raise.
    regression_files
        ``{filename: text}`` pre-written into ``regression/`` (creating it), or ``None`` to
        leave the baseline absent.
    fetch
        When True, patch ``_fetch_and_build_catalog`` (for ``run --fetch``) instead of
        ``load_datasets_from_yaml``.

    Returns
    -------
    :
        ``(paths, scratch_output_dir, regression_dir, runner)``.
    """
    from climate_ref_core.pycmec.output import CMECOutput
    from climate_ref_core.testing import TestCasePaths

    registry, _diag, _tc = _make_case_mocks()

    test_case_dir = tmp_path / "td" / "test-diag" / "default"
    test_case_dir.mkdir(parents=True)
    # A minimal but valid catalog so the real load_datasets_from_yaml returns an empty
    # ExecutionDatasetCollection. The slot stages load the catalog directly (via the stages
    # module's own binding), so patching only climate_ref_core.testing would not reach them.
    (test_case_dir / "catalog.yaml").write_text("_metadata:\n  hash: abc123\n")
    paths = TestCasePaths(root=test_case_dir)

    if regression_files is not None:
        paths.regression.mkdir(parents=True)
        for name, text in regression_files.items():
            (paths.regression / name).write_text(text)

    scratch_output_dir = tmp_path / "scratch" / "frag"
    scratch_output_dir.mkdir(parents=True)
    (scratch_output_dir / "diagnostic.json").write_text('{"test": "data"}')
    (scratch_output_dir / "output.json").write_text(json.dumps(CMECOutput.create_template()))
    (scratch_output_dir / "series.json").write_text("[]")

    defn = MagicMock()
    defn.output_directory = scratch_output_dir
    defn.output_fragment.return_value = Path("frag")
    result = MagicMock(
        successful=(runner_result == "success"),
        metric_bundle_filename=Path("diagnostic.json"),
        output_bundle_filename=Path("output.json"),
        series_filename=Path("series.json"),
    )
    result.to_output_path.side_effect = lambda f: scratch_output_dir / f
    result.definition = defn

    runner = MagicMock()
    if isinstance(runner_result, BaseException):
        runner.run.side_effect = runner_result
    else:
        runner.run.return_value = result

    mocker.patch(
        "climate_ref.provider_registry.ProviderRegistry.build_from_config",
        return_value=registry,
    )
    mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
    mocker.patch("climate_ref.testing.TestCaseRunner", return_value=runner)
    mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value="abc123")
    if fetch:
        mocker.patch(
            "climate_ref.cli.test_cases.run._fetch_and_build_catalog",
            return_value=(MagicMock(), True),
        )

    return paths, scratch_output_dir, paths.regression, runner


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

    def test_sync_nonexistent_provider(self, invoke_cli, mocker):
        """sync with an unknown provider exits 1 rather than silently syncing nothing."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "sync", "--provider", "nonexistent"],
            expected_exit_code=1,
        )
        assert "not configured" in result.stderr

    def test_sync_nonexistent_diagnostic(self, invoke_cli, mocker):
        """sync with a bad diagnostic filter fails instead of reporting a zero sync."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "sync", "--provider", "example", "--diagnostic", "missing"],
            expected_exit_code=1,
        )
        assert "Diagnostic 'missing' was not found" in result.stderr

    def test_sync_nonexistent_test_case(self, invoke_cli, mocker):
        """sync with a bad test-case filter fails instead of reporting a zero sync."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "sync", "--provider", "example", "--test-case", "missing"],
            expected_exit_code=1,
        )
        assert "Test case 'missing' was not found" in result.stderr


class TestStageCompare:
    """`stage_compare` drives replay verification from the manifest's committed set."""

    @staticmethod
    def _paths_and_slot(tmp_path):
        from climate_ref_core.testing import TestCasePaths

        case_dir = tmp_path / "td" / "test-diag" / "default"
        case_dir.mkdir(parents=True)
        paths = TestCasePaths(root=case_dir)
        paths.regression.mkdir(parents=True)
        slot = tmp_path / "slot"
        (slot / "regression").mkdir(parents=True)
        return paths, slot

    def test_missing_baseline_is_hard_failure(self, tmp_path):
        """A committed file the manifest expects but that is absent from regression/ fails the replay."""
        from climate_ref.cli.test_cases._stages import stage_compare

        paths, slot = self._paths_and_slot(tmp_path)
        # The rebuild produced a bundle, but the tracked baseline is missing entirely.
        (slot / "regression" / "diagnostic.json").write_text("{}")

        failures, compared = stage_compare(slot=slot, paths=paths, slug="x", expected=["diagnostic.json"])
        assert compared == []
        assert any("baseline file missing" in f for f in failures)

    def test_missing_regenerated_is_hard_failure(self, tmp_path):
        """A committed file present in the baseline but not regenerated by the rebuild fails the replay."""
        from climate_ref.cli.test_cases._stages import stage_compare

        paths, slot = self._paths_and_slot(tmp_path)
        (paths.regression / "diagnostic.json").write_text("{}")

        failures, compared = stage_compare(slot=slot, paths=paths, slug="x", expected=["diagnostic.json"])
        assert compared == []
        assert any("did not regenerate" in f for f in failures)

    def test_empty_expected_is_hard_failure(self, tmp_path):
        """A replay that would compare nothing must fail rather than report a green match."""
        from climate_ref.cli.test_cases._stages import stage_compare

        paths, slot = self._paths_and_slot(tmp_path)
        failures, compared = stage_compare(slot=slot, paths=paths, slug="x", expected=[])
        assert compared == []
        assert failures and "no committed bundle" in failures[0]

    def test_present_on_both_sides_is_compared(self, tmp_path, mocker):
        """When both sides hold the expected file, it is compared (and reported as compared)."""
        from climate_ref.cli.test_cases import _stages

        paths, slot = self._paths_and_slot(tmp_path)
        (paths.regression / "diagnostic.json").write_text("{}")
        (slot / "regression" / "diagnostic.json").write_text("{}")
        # Isolate from CMEC bundle parsing: a no-op comparator stands in for "no drift".
        assert_mock = mocker.patch.object(_stages, "assert_bundle_regression")

        failures, compared = _stages.stage_compare(
            slot=slot, paths=paths, slug="x", expected=["diagnostic.json"]
        )
        assert failures == []
        assert compared == ["diagnostic.json"]
        assert_mock.assert_called_once()

    def test_unparseable_baseline_is_hard_failure(self, tmp_path):
        """A committed baseline that no longer parses as JSON fails cleanly rather than crashing."""
        from climate_ref.cli.test_cases._stages import stage_compare

        paths, slot = self._paths_and_slot(tmp_path)
        # A corrupted committed baseline: valid JSON with trailing garbage appended.
        (paths.regression / "diagnostic.json").write_text("{}\n// drift\n")
        (slot / "regression" / "diagnostic.json").write_text("{}")

        failures, compared = stage_compare(slot=slot, paths=paths, slug="x", expected=["diagnostic.json"])
        assert compared == []
        assert any("not valid JSON" in f for f in failures)


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
            [
                "test-cases",
                "replay",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
            ],
            expected_exit_code=1,
        )
        assert "not yet minted" in result.stderr.lower() or "mint" in result.stderr.lower()

    def test_replay_nonexistent_provider(self, invoke_cli, mocker):
        """replay with an unknown provider exits 1 rather than reporting 'no test cases'."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "replay", "--provider", "nonexistent"],
            expected_exit_code=1,
        )
        assert "not configured" in result.stderr

    def test_replay_nonexistent_diagnostic(self, invoke_cli, mocker):
        """replay with a bad diagnostic filter fails before opening the native store."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store_builder = mocker.patch("climate_ref_core.regression.store.build_native_store")

        result = invoke_cli(
            ["test-cases", "replay", "--provider", "example", "--diagnostic", "missing"],
            expected_exit_code=1,
        )
        assert "Diagnostic 'missing' was not found" in result.stderr
        store_builder.assert_not_called()

    def test_replay_nonexistent_test_case(self, invoke_cli, mocker):
        """replay with a bad test-case filter fails before opening the native store."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store_builder = mocker.patch("climate_ref_core.regression.store.build_native_store")

        result = invoke_cli(
            ["test-cases", "replay", "--provider", "example", "--test-case", "missing"],
            expected_exit_code=1,
        )
        assert "Test case 'missing' was not found" in result.stderr
        store_builder.assert_not_called()

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
            [
                "test-cases",
                "replay",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
            ],
            expected_exit_code=1,
        )
        # The mismatch warned rather than gated, and execution fell through to the native guard.
        assert "differs from the digests recorded" in result.stderr
        assert "not yet minted" in result.stderr.lower()

    def test_replay_reconciles_integrity_mismatch_within_tolerance(self, invoke_cli, mocker, tmp_path):
        """A byte-level baseline difference forgiven by the tolerant comparison is reported as reconciled.

        This proves the comparison stage actually ran after the integrity warning: the success line
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
        # Stub the slot stages so we deterministically reach the success branch: materialise +
        # rebuild, committed-bundle build, and the tolerant comparison (no drift, 3 files compared).
        mocker.patch("climate_ref.cli.test_cases.baselines.stage_materialise")
        mocker.patch("climate_ref.cli.test_cases.baselines.stage_build")
        mocker.patch(
            "climate_ref.cli.test_cases.baselines.stage_compare",
            return_value=([], ["series.json", "diagnostic.json", "output.json"]),
        )

        result = invoke_cli(
            [
                "test-cases",
                "replay",
                "--provider",
                "example",
                "--diagnostic",
                "test-diag",
            ]
        )
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

    def test_mint_fails_fast_on_preflight_error(self, invoke_cli, mocker):
        """A store auth/connectivity failure must abort before any diagnostic runs."""
        from climate_ref_core.regression.store import NativeStoreUnavailableError

        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store = MagicMock()
        store.preflight.side_effect = NativeStoreUnavailableError(
            "Native store authentication failed (HTTP 401) for bucket 'ref-baselines-public'"
        )
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            return_value=store,
        )

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "example"],
            expected_exit_code=1,
        )
        assert "Cannot mint" in result.stderr
        assert "401" in result.stderr
        store.preflight.assert_called_once()
        # Fail-fast: the diagnostic runner must never have been reached.
        store.put.assert_not_called()

    def test_mint_nonexistent_provider(self, invoke_cli, mocker):
        """mint with an unknown provider exits 1 before opening the writable store."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store_builder = mocker.patch("climate_ref_core.regression.store.build_native_store")

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "nonexistent"],
            expected_exit_code=1,
        )
        assert "not configured" in result.stderr
        store_builder.assert_not_called()

    def test_mint_nonexistent_diagnostic(self, invoke_cli, mocker):
        """mint with a bad diagnostic filter fails before opening the writable store."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store_builder = mocker.patch("climate_ref_core.regression.store.build_native_store")

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "example", "--diagnostic", "missing", "--dry-run"],
            expected_exit_code=1,
        )
        assert "Diagnostic 'missing' was not found" in result.stderr
        store_builder.assert_not_called()

    def test_mint_nonexistent_test_case(self, invoke_cli, mocker):
        """mint with a bad test-case filter fails before opening the writable store."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store_builder = mocker.patch("climate_ref_core.regression.store.build_native_store")

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "example", "--test-case", "missing", "--dry-run"],
            expected_exit_code=1,
        )
        assert "Test case 'missing' was not found" in result.stderr
        store_builder.assert_not_called()

    def test_mint_dry_run_lists_without_running(self, invoke_cli, mocker):
        """--dry-run preflights + lists the cases, but runs no diagnostics and uploads nothing."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        store = MagicMock()  # preflight passes (no side effect)
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            return_value=store,
        )
        runner_cls = mocker.patch("climate_ref.testing.TestCaseRunner")

        result = invoke_cli(["test-cases", "mint", "--provider", "example", "--dry-run"])
        assert "Dry run" in result.stdout
        assert "example/test-diag/default" in result.stdout
        store.preflight.assert_called_once()
        store.put.assert_not_called()
        runner_cls.assert_not_called()

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

    def test_mint_skips_unchanged_native_on_remint(self, invoke_cli, mocker, tmp_path):
        """Re-executing a mint with byte-identical native uploads nothing (changed-digest skip)."""
        from climate_ref_core.regression.store import LocalFilesystemStore

        _setup_real_run(mocker, tmp_path)
        store = LocalFilesystemStore(root=tmp_path / "store")
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        first = invoke_cli(["test-cases", "mint", "--provider", "example"])
        assert first.exit_code == 0
        assert "Uploaded 3 native blob(s), skipped 0" in first.stderr  # first mint uploads everything

        # A second mint with identical outputs re-executes but uploads nothing new.
        second = invoke_cli(["test-cases", "mint", "--provider", "example"])
        assert second.exit_code == 0
        assert "Uploaded 0 native blob(s), skipped 3" in second.stderr

    def test_mint_from_replay_reauthors_without_reexecuting(self, invoke_cli, mocker, tmp_path):
        """`mint --from-replay` rebuilds from stored native, uploading nothing and not re-executing."""
        from climate_ref_core.regression.manifest import Manifest
        from climate_ref_core.regression.store import LocalFilesystemStore

        paths, _scratch, _regression, runner = _setup_real_run(mocker, tmp_path)
        store = LocalFilesystemStore(root=tmp_path / "store")
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        # First mint executes the diagnostic and uploads the native set.
        first = invoke_cli(["test-cases", "mint", "--provider", "example"])
        assert first.exit_code == 0
        minted = Manifest.load(paths.manifest)
        assert set(minted.native) == {"diagnostic.json", "output.json", "series.json"}
        assert runner.run.call_count == 1
        assert "Uploaded 3 native blob(s), skipped 0" in first.stderr

        # Re-mint from the stored native: no execution, no new uploads, identical native block.
        result = invoke_cli(["test-cases", "mint", "--provider", "example", "--from-replay"])
        assert result.exit_code == 0
        assert runner.run.call_count == 1  # unchanged: the diagnostic was not re-run
        assert "Uploaded 0 native blob(s), skipped 3" in result.stderr
        replayed = Manifest.load(paths.manifest)
        assert {k: v.sha256 for k, v in replayed.native.items()} == {
            k: v.sha256 for k, v in minted.native.items()
        }
        # The slot records the from-replay source.
        stamp = json.loads((paths.output_slot("latest") / ".source.json").read_text())
        assert stamp["verb"] == "mint"
        assert stamp["source"] == "materialise"

    def test_mint_from_replay_preserves_stored_native(self, invoke_cli, mocker, tmp_path):
        """`mint --from-replay` keeps the stored native verbatim even if the rebuilt slot drifts.

        stage_materialise hydrates the slot's native in place (placeholders -> concrete paths) while
        rebuilding, so snapshotting the slot can capture non-portable, slot-specific blobs. The minted
        manifest must preserve the previously stored native rather than re-author it from that drift.
        """
        from climate_ref_core.regression.manifest import Manifest, NativeEntry
        from climate_ref_core.testing import TestCasePaths

        registry, _diag, _tc = _make_case_mocks()

        case_dir = tmp_path / "td" / "test-diag" / "default"
        case_dir.mkdir(parents=True)
        (case_dir / "catalog.yaml").touch()
        paths = TestCasePaths(root=case_dir)
        stored_native = {"output.json": NativeEntry(sha256="aa" * 32, size=10)}
        Manifest(
            schema=1,
            test_case_version=1,
            committed={"diagnostic.json": "00" * 32},
            native=stored_native,
        ).dump(paths.manifest)

        store = MagicMock()  # preflight passes; has() True so the (no-op) upload skips
        store.has.return_value = True
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref_core.testing.get_catalog_hash", return_value=None)
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        # Heavy stages stubbed; the rebuild "mutates" the slot so a fresh snapshot would drift.
        mocker.patch("climate_ref.cli.test_cases.baselines.stage_materialise")
        mocker.patch(
            "climate_ref.cli.test_cases.baselines.stage_build",
            return_value={"diagnostic.json": "11" * 32},
        )
        snapshot = mocker.patch(
            "climate_ref.cli.test_cases.baselines.snapshot_native",
            return_value={"output.json": NativeEntry(sha256="bb" * 32, size=20)},
        )
        mocker.patch("climate_ref.cli.test_cases.baselines.promote_to_baseline")

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "example", "--diagnostic", "test-diag", "--from-replay"]
        )
        assert result.exit_code == 0

        written = Manifest.load(paths.manifest)
        # The native block is the stored one, NOT the mutated slot snapshot ("bb"*32).
        assert {k: v.sha256 for k, v in written.native.items()} == {
            k: v.sha256 for k, v in stored_native.items()
        }
        # The from-replay path must not snapshot the hydrated slot at all.
        snapshot.assert_not_called()

    def test_mint_from_replay_requires_minted_manifest(self, invoke_cli, mocker, tmp_path):
        """`mint --from-replay` fails when there is no existing minted native to replay from."""
        from climate_ref_core.regression.store import LocalFilesystemStore

        _setup_real_run(mocker, tmp_path)
        store = LocalFilesystemStore(root=tmp_path / "store")
        mocker.patch("climate_ref_core.regression.store.build_native_store", return_value=store)

        result = invoke_cli(
            ["test-cases", "mint", "--provider", "example", "--from-replay"],
            expected_exit_code=1,
        )
        assert "--from-replay needs an existing minted manifest" in result.stderr


class TestBuildCommand:
    """Tests for the `test-cases build` CLI verb."""

    def test_build_help(self, invoke_cli):
        result = invoke_cli(["test-cases", "build", "--help"])
        assert "Rebuild the committed bundle" in result.stdout

    def test_build_fails_without_slot(self, invoke_cli, mocker, tmp_path):
        """build refuses when the named output slot has no native to rebuild from."""
        _setup_real_run(mocker, tmp_path)  # catalog present, but no output slot materialised
        result = invoke_cli(
            ["test-cases", "build", "--provider", "example", "--diagnostic", "test-diag"],
            expected_exit_code=1,
        )
        assert "no native in output slot" in result.stderr.lower()

    def test_build_nonexistent_provider(self, invoke_cli, mocker):
        """build with an unknown provider exits 1 rather than reporting 'no test cases'."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "build", "--provider", "nonexistent"],
            expected_exit_code=1,
        )
        assert "not configured" in result.stderr

    def test_build_nonexistent_diagnostic(self, invoke_cli, mocker):
        """build with a bad diagnostic filter fails instead of silently selecting nothing."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "build", "--provider", "example", "--diagnostic", "missing"],
            expected_exit_code=1,
        )
        assert "Diagnostic 'missing' was not found" in result.stderr

    def test_build_nonexistent_test_case(self, invoke_cli, mocker):
        """build with a bad test-case filter fails instead of silently selecting nothing."""
        registry, _diag, _tc = _make_case_mocks()
        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=registry,
        )

        result = invoke_cli(
            ["test-cases", "build", "--provider", "example", "--test-case", "missing"],
            expected_exit_code=1,
        )
        assert "Test case 'missing' was not found" in result.stderr

    def test_build_rebuilds_from_slot_without_executing(self, invoke_cli, mocker, tmp_path):
        """build regenerates the committed bundle from an existing slot, never re-running."""
        paths, _scratch, _regression, runner = _setup_real_run(mocker, tmp_path)

        # Seed a slot (and baseline) with one run.
        assert (
            invoke_cli(
                ["test-cases", "run", "--provider", "example", "--diagnostic", "test-diag", "--force-regen"]
            ).exit_code
            == 0
        )
        assert runner.run.call_count == 1

        # Rebuild the committed bundle from that slot, without executing the diagnostic.
        result = invoke_cli(["test-cases", "build", "--provider", "example", "--diagnostic", "test-diag"])
        assert result.exit_code == 0
        assert runner.run.call_count == 1  # build did not re-run the diagnostic
        assert (paths.output_slot("latest") / "regression" / "diagnostic.json").exists()
        stamp = json.loads((paths.output_slot("latest") / ".source.json").read_text())
        assert stamp["verb"] == "build"
        assert stamp["source"] == "rebuild"


def test_output_slot_pattern_is_gitignored(tmp_path):
    """The repo .gitignore shadows materialised output slots under any test-data tree."""
    from git import Repo

    repo_gitignore = Path(__file__).resolve().parents[5] / ".gitignore"
    repo = Repo.init(tmp_path)
    (tmp_path / ".gitignore").write_text(repo_gitignore.read_text())

    slot = tmp_path / "packages" / "pkg" / "tests" / "test-data" / "diag" / "default" / "output" / "latest"
    slot.mkdir(parents=True)
    (slot / "native.nc").write_bytes(b"x")

    rel = "packages/pkg/tests/test-data/diag/default/output/latest/native.nc"
    # git check-ignore echoes the path when it is ignored (and exits non-zero otherwise).
    assert repo.git.check_ignore(rel) == rel


class TestCheckStoreCommand:
    """Tests for the ``ref test-cases check-store`` command."""

    def test_check_store_help(self, invoke_cli):
        result = invoke_cli(["test-cases", "check-store", "--help"])
        assert "writable native baseline store" in result.stdout

    def test_check_store_ok(self, invoke_cli, mocker):
        store = MagicMock()  # preflight passes (no side effect)
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            return_value=store,
        )
        result = invoke_cli(["test-cases", "check-store"])
        assert "Native store OK" in result.stdout
        store.preflight.assert_called_once()

    def test_check_store_reports_auth_failure(self, invoke_cli, mocker):
        from climate_ref_core.regression.store import NativeStoreUnavailableError

        store = MagicMock()
        store.preflight.side_effect = NativeStoreUnavailableError(
            "Native store authentication failed (HTTP 401) for bucket 'ref-baselines-public'"
        )
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            return_value=store,
        )
        result = invoke_cli(["test-cases", "check-store"], expected_exit_code=1)
        assert "401" in result.stderr

    def test_check_store_reports_unconfigured(self, invoke_cli, mocker):
        mocker.patch(
            "climate_ref_core.regression.store.build_native_store",
            side_effect=NotImplementedError("R2 backend deferred"),
        )
        result = invoke_cli(["test-cases", "check-store"], expected_exit_code=1)
        assert "not configured" in result.stderr


class TestCIGateCommand:
    """Tests for the ``ref test-cases ci-gate`` command."""

    def _setup(
        self,
        mocker,
        tmp_path,
        *,
        current_version=None,
        committed_content=None,
        native=None,
    ):
        """
        Wire a single example test case for the gate.

        Writes the *current* manifest (when ``current_version`` is set) alongside real
        committed files in ``regression/``, so the gate's on-disk integrity check sees a
        manifest that faithfully describes the bundle.
        The base-branch manifest is left absent by default (``git show`` raises);
        a test sets ``repo.git.show`` to supply one,
        using the returned committed digests so they line up with the current bundle.

        Parameters
        ----------
        current_version
            ``test_case_version`` of the current manifest,
            or ``None`` to leave the manifest absent (never managed / deleted on this branch).
        committed_content
            ``{filename: text}`` written to ``regression/`` for the current manifest.
            Defaults to a single ``output.json``.
        native
            Native entries for the current manifest.

        Returns
        -------
        :
            ``(repo, paths, committed_digests)`` — the mock repo, the test case paths,
            and the digests computed from the on-disk committed bundle.
        """
        from git import GitCommandError

        from climate_ref_core.regression.manifest import (
            SCHEMA_VERSION,
            Manifest,
            compute_committed_digests,
        )
        from climate_ref_core.testing import TestCasePaths

        mock_tc = MagicMock()
        mock_tc.name = "default"
        mock_spec = MagicMock(test_cases=[mock_tc])
        mock_diag = MagicMock(slug="test-diag", test_data_spec=mock_spec)
        mock_diag.provider = MagicMock(slug="example")
        mock_provider = MagicMock(slug="example")
        mock_provider.diagnostics.return_value = [mock_diag]
        mock_registry = MagicMock(providers=[mock_provider])

        case_dir = tmp_path / "test-diag" / "default"
        case_dir.mkdir(parents=True)
        paths = TestCasePaths(root=case_dir)

        committed_digests: dict[str, str] = {}
        if current_version is not None:
            content = committed_content if committed_content is not None else {"output.json": '{"x": 1}\n'}
            paths.regression.mkdir(parents=True, exist_ok=True)
            for name, text in content.items():
                (paths.regression / name).write_text(text, encoding="utf-8")
            committed_digests = compute_committed_digests(paths.regression)
            Manifest(
                schema=SCHEMA_VERSION,
                test_case_version=current_version,
                committed=committed_digests,
                native=native or {},
            ).dump(paths.manifest)

        repo = MagicMock()
        repo.working_dir = str(tmp_path)
        repo.git.diff.return_value = ""
        # Base manifest absent by default; tests that need one override repo.git.show.
        repo.git.show.side_effect = GitCommandError("show", 128)

        mocker.patch(
            "climate_ref.provider_registry.ProviderRegistry.build_from_config",
            return_value=mock_registry,
        )
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=paths)
        mocker.patch("climate_ref.cli.test_cases.ci_gate.get_repo_for_path", return_value=repo)
        return repo, paths, committed_digests

    @staticmethod
    def _set_base(repo, version, committed, native=None):
        """Configure ``repo.git.show`` to return a base manifest with these fields."""
        import json as _json

        from attrs import asdict

        from climate_ref_core.regression.manifest import SCHEMA_VERSION

        payload = {
            "schema": SCHEMA_VERSION,
            "test_case_version": version,
            "committed": committed,
            "native": {relpath: asdict(entry) for relpath, entry in (native or {}).items()},
        }
        repo.git.show.side_effect = None
        repo.git.show.return_value = _json.dumps(payload, indent=2, sort_keys=True)

    def test_seeding_replays(self, invoke_cli, mocker, tmp_path):
        # Current manifest present with a native baseline, no base manifest -> seeding -> REPLAY.
        from climate_ref_core.regression.manifest import NativeEntry

        self._setup(mocker, tmp_path, current_version=1, native={"data.nc": NativeEntry("a" * 64, 10)})

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "replay" in result.output

    def test_ci_gate_nonexistent_provider(self, invoke_cli, mocker, tmp_path):
        # An unknown provider must fail closed (exit 1), not gate nothing and silently exit 0.
        # The registry mock only knows the 'example' provider.
        self._setup(mocker, tmp_path, current_version=1)

        result = invoke_cli(
            ["test-cases", "ci-gate", "--provider", "nonexistent"],
            expected_exit_code=1,
        )
        assert result.exit_code == 1
        assert "not configured" in result.stderr

    def test_ci_gate_nonexistent_diagnostic(self, invoke_cli, mocker, tmp_path):
        # A bad diagnostic selector must fail closed instead of emitting an empty decision list.
        self._setup(mocker, tmp_path, current_version=1)

        result = invoke_cli(
            ["test-cases", "ci-gate", "--provider", "example", "--diagnostic", "missing"],
            expected_exit_code=1,
        )
        assert result.exit_code == 1
        assert "Diagnostic 'missing' was not found" in result.stderr

    def test_ci_gate_nonexistent_test_case(self, invoke_cli, mocker, tmp_path):
        # A bad test-case selector must fail closed instead of emitting an empty decision list.
        self._setup(mocker, tmp_path, current_version=1)

        result = invoke_cli(
            ["test-cases", "ci-gate", "--provider", "example", "--test-case", "missing"],
            expected_exit_code=1,
        )
        assert result.exit_code == 1
        assert "Test case 'missing' was not found" in result.stderr

    def test_seeding_without_native_skips(self, invoke_cli, mocker, tmp_path):
        # Current manifest present with an empty native set, no base manifest -> seeding -> SKIP.
        self._setup(mocker, tmp_path, current_version=1)

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "skip" in result.output

    def test_version_bump_executes(self, invoke_cli, mocker, tmp_path):
        repo, _, digests = self._setup(mocker, tmp_path, current_version=2)
        self._set_base(repo, 1, digests)

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "execute" in result.output

    def test_committed_change_without_bump_fails(self, invoke_cli, mocker, tmp_path):
        repo, _, _ = self._setup(mocker, tmp_path, current_version=1)
        # Base committed differs from the on-disk bundle, with no version bump.
        self._set_base(repo, 1, {"output.json": "f" * 64})

        result = invoke_cli(["test-cases", "ci-gate"], expected_exit_code=1)
        assert result.exit_code == 1
        assert "fail" in result.output

    def test_no_change_skips(self, invoke_cli, mocker, tmp_path):
        repo, _, digests = self._setup(mocker, tmp_path, current_version=1)
        self._set_base(repo, 1, digests)

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "skip" in result.output

    def test_core_change_replays(self, invoke_cli, mocker, tmp_path):
        repo, _, digests = self._setup(mocker, tmp_path, current_version=1)
        self._set_base(repo, 1, digests)
        repo.git.diff.return_value = "packages/climate-ref-core/src/climate_ref_core/regression/compare.py"

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "replay" in result.output

    def test_native_change_replays(self, invoke_cli, mocker, tmp_path):
        from climate_ref_core.regression.manifest import NativeEntry

        repo, _, digests = self._setup(
            mocker,
            tmp_path,
            current_version=1,
            native={"data.nc": NativeEntry("a" * 64, 10)},
        )
        # Committed + version unchanged, but native blob re-minted -> REPLAY.
        self._set_base(repo, 1, digests, native={"data.nc": NativeEntry("b" * 64, 12)})

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "replay" in result.output

    def test_committed_integrity_drift_fails(self, invoke_cli, mocker, tmp_path):
        repo, paths, digests = self._setup(mocker, tmp_path, current_version=1)
        self._set_base(repo, 1, digests)
        # Edit the on-disk committed bundle without regenerating the manifest.
        (paths.regression / "output.json").write_text('{"x": 999}\n', encoding="utf-8")

        result = invoke_cli(["test-cases", "ci-gate"], expected_exit_code=1)
        assert result.exit_code == 1
        assert "fail" in result.output

    def test_json_output(self, invoke_cli, mocker, tmp_path):
        repo, _, digests = self._setup(mocker, tmp_path, current_version=2)
        self._set_base(repo, 1, digests)

        result = invoke_cli(["test-cases", "ci-gate", "--json"])
        assert result.exit_code == 0
        # The test runner mixes loguru's stderr lines into the captured output; in real
        # use stdout (the JSON) is captured separately. Slice from the JSON array start.
        payload = json.loads(result.output[result.output.index("[") :])
        assert payload[0]["action"] == "execute"
        assert payload[0]["case"] == "example/test-diag/default"

    def test_manifest_deletion_fails(self, invoke_cli, mocker, tmp_path):
        # Manifest absent on this branch but present on the base -> deletion -> FAIL.
        repo, _, _ = self._setup(mocker, tmp_path, current_version=None)
        self._set_base(repo, 1, {"output.json": "a" * 64})

        result = invoke_cli(["test-cases", "ci-gate"], expected_exit_code=1)
        assert result.exit_code == 1
        assert "fail" in result.output

    def test_never_managed_skips(self, invoke_cli, mocker, tmp_path):
        # No current manifest and none on the base -> never managed -> SKIP.
        self._setup(mocker, tmp_path, current_version=None)

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "skip" in result.output

    def test_not_in_repo_fails(self, invoke_cli, mocker, tmp_path):
        self._setup(mocker, tmp_path, current_version=1)
        mocker.patch("climate_ref.cli.test_cases.ci_gate.get_repo_for_path", return_value=None)

        result = invoke_cli(["test-cases", "ci-gate"], expected_exit_code=1)
        assert result.exit_code == 1

    def test_paths_none_skips(self, invoke_cli, mocker, tmp_path):
        # A non-dev checkout: from_diagnostic cannot locate the test case directory.
        self._setup(mocker, tmp_path, current_version=1)
        mocker.patch("climate_ref_core.testing.TestCasePaths.from_diagnostic", return_value=None)

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "skip" in result.output

    def test_corrupt_current_manifest_fails(self, invoke_cli, mocker, tmp_path):
        # A valid manifest is written, then overwritten with garbage.
        _, paths, _ = self._setup(mocker, tmp_path, current_version=1)
        paths.manifest.write_text("not json{", encoding="utf-8")

        result = invoke_cli(["test-cases", "ci-gate"], expected_exit_code=1)
        assert result.exit_code == 1
        assert "fail" in result.output

    def test_corrupt_base_manifest_treated_as_seeding(self, invoke_cli, mocker, tmp_path):
        # git show returns non-JSON for the base manifest: must not crash; seed (replay).
        from climate_ref_core.regression.manifest import NativeEntry

        repo, _, _ = self._setup(
            mocker, tmp_path, current_version=1, native={"data.nc": NativeEntry("a" * 64, 10)}
        )
        repo.git.show.side_effect = None
        repo.git.show.return_value = "}}not json"

        result = invoke_cli(["test-cases", "ci-gate"])
        assert result.exit_code == 0
        assert "replay" in result.output

    def test_catalog_drift_fails(self, invoke_cli, mocker, tmp_path):
        # The manifest records a catalog_hash that the on-disk catalog no longer matches.
        from climate_ref_core.regression.manifest import Manifest

        _, paths, _ = self._setup(mocker, tmp_path, current_version=1)
        manifest = Manifest.load(paths.manifest)
        Manifest(
            schema=manifest.schema,
            test_case_version=manifest.test_case_version,
            committed=manifest.committed,
            native=manifest.native,
            catalog_hash="expected_hash",
        ).dump(paths.manifest)
        paths.catalog.write_text("_metadata:\n  hash: different_hash\ncmip6:\n  datasets: []\n")

        result = invoke_cli(["test-cases", "ci-gate"], expected_exit_code=1)
        assert result.exit_code == 1
        assert "fail" in result.output
