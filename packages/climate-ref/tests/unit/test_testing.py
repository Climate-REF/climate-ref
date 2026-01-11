"""
Tests for the test harness fixtures.
"""

from unittest.mock import MagicMock, patch

import pytest

from climate_ref.testing import (
    TestCaseRunner,
    get_provider_catalog_path,
    get_provider_regression_path,
)
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)
from climate_ref_core.testing import TestCase, TestDataSpecification


class TestGetProviderPaths:
    """Tests for provider path resolution functions."""

    def test_catalog_path_returns_none_when_module_not_loaded(self):
        """Test returns None when provider module is not in sys.modules."""
        mock_diag = MagicMock()
        mock_diag.provider.__class__.__module__ = "nonexistent_module"

        result = get_provider_catalog_path(mock_diag, "default")
        assert result is None

    def test_regression_path_returns_none_when_module_not_loaded(self):
        """Test returns None when provider module is not in sys.modules."""
        mock_diag = MagicMock()
        mock_diag.provider.__class__.__module__ = "nonexistent_module"

        result = get_provider_regression_path(mock_diag, "default")
        assert result is None

    def test_catalog_path_returns_correct_path(self, tmp_path):
        """Test returns correct catalog path for loaded provider."""
        # Create mock module structure
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.provider.__class__.__module__ = "test_provider.provider"
        mock_diag.slug = "my-diag"

        # Create the catalog directory
        catalog_dir = tmp_path / "tests" / "test-data" / "catalogs"
        catalog_dir.mkdir(parents=True)

        with patch.dict("sys.modules", {"test_provider": mock_module}):
            result = get_provider_catalog_path(mock_diag, "default")
            assert result == catalog_dir / "my-diag" / "default.yaml"

    def test_regression_path_returns_correct_path(self, tmp_path):
        """Test returns correct regression path for loaded provider."""
        # Create mock module structure
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.provider.__class__.__module__ = "test_provider.provider"
        mock_diag.provider.slug = "my-provider"
        mock_diag.slug = "my-diag"

        # Create tests directory (indicates dev checkout)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True)

        with patch.dict("sys.modules", {"test_provider": mock_module}):
            result = get_provider_regression_path(mock_diag, "default")
            expected = tmp_path / "tests" / "test-data" / "regression" / "my-provider" / "my-diag" / "default"
            assert result == expected

    def test_returns_none_when_tests_dir_missing(self, tmp_path):
        """Test returns None when tests/ directory doesn't exist (not a dev checkout)."""
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.provider.__class__.__module__ = "test_provider.provider"
        mock_diag.slug = "my-diag"

        # Don't create tests/ directory
        with patch.dict("sys.modules", {"test_provider": mock_module}):
            assert get_provider_catalog_path(mock_diag, "default") is None
            assert get_provider_regression_path(mock_diag, "default") is None

    def test_catalog_path_creates_dirs_when_create_true(self, tmp_path):
        """Test that create=True creates the catalogs directory."""
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.provider.__class__.__module__ = "test_provider.provider"
        mock_diag.slug = "my-diag"

        # Create tests/ but not test-data/catalogs
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True)

        with patch.dict("sys.modules", {"test_provider": mock_module}):
            result = get_provider_catalog_path(mock_diag, "default", create=True)
            assert result is not None
            assert (tmp_path / "tests" / "test-data" / "catalogs").exists()


class TestTestCaseRunnerClass:
    """Tests for the TestCaseRunner class directly."""

    def test_run_no_test_data_spec(self, config):
        """Test that run raises NoTestDataSpecError when diagnostic has no test_data_spec."""
        runner = TestCaseRunner(config=config, datasets=None)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = None
        mock_diagnostic.slug = "test-diag"

        with pytest.raises(NoTestDataSpecError, match="no test_data_spec"):
            runner.run(mock_diagnostic)

    def test_run_test_case_not_found(self, config):
        """Test error when test case doesn't exist."""
        runner = TestCaseRunner(config=config, datasets=None)

        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(TestCase(name="other", description="Other case"),)
        )

        with pytest.raises(TestCaseNotFoundError, match="not found"):
            runner.run(mock_diagnostic, "nonexistent")

    def test_run_no_datasets_raises_error(self, config):
        """Test that run raises DatasetResolutionError when datasets is None."""
        runner = TestCaseRunner(config=config, datasets=None)

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(TestCase(name="default", description="Default"),)
        )

        with pytest.raises(DatasetResolutionError, match="No datasets provided"):
            runner.run(mock_diagnostic)

    def test_run_uses_default_output_dir(self, config):
        """Test that run uses default output directory when not provided."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)
        runner = TestCaseRunner(config=config, datasets=mock_datasets)

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"
        mock_diagnostic.provider.slug = "test-provider"
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(TestCase(name="default", description="Default"),)
        )
        mock_result = MagicMock(successful=True)
        mock_diagnostic.run.return_value = mock_result

        result = runner.run(mock_diagnostic, "default")

        assert result.successful
        # Verify diagnostic.run was called with an ExecutionDefinition
        call_args = mock_diagnostic.run.call_args[0][0]
        assert "test-cases" in str(call_args.output_directory)
        assert "test-provider" in str(call_args.output_directory)
        assert "test-diag" in str(call_args.output_directory)

    def test_run_with_explicit_datasets(self, config, tmp_path):
        """Test running with explicit datasets provided to runner."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)
        runner = TestCaseRunner(config=config, datasets=mock_datasets)

        # Create mock diagnostic with test case
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"
        mock_diagnostic.provider.slug = "test-provider"
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(
                TestCase(
                    name="default",
                    description="Default test",
                ),
            )
        )

        # Mock the run method to avoid actual execution
        mock_result = MagicMock()
        mock_result.successful = True
        mock_diagnostic.run.return_value = mock_result

        result = runner.run(mock_diagnostic, "default", output_dir=tmp_path)

        assert result.successful
        mock_diagnostic.run.assert_called_once()
        execution_definition = mock_diagnostic.run.call_args[0][0]
        assert execution_definition.datasets == mock_datasets


class TestTestCaseRunnerPytestFixture:
    """Tests for the pytest fixture wrapper."""

    def test_run_no_test_data_spec_skips(self, run_test_case):
        """Test that the fixture skips when diagnostic has no test_data_spec."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = None
        mock_diagnostic.slug = "test-diag"

        with pytest.raises(pytest.skip.Exception):
            run_test_case.run(mock_diagnostic)
