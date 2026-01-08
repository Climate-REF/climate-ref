"""
Tests for the test harness fixtures.
"""

from unittest.mock import MagicMock, patch

import pytest

from climate_ref.testing import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
    TestCaseRunner,
    get_catalog_path,
)
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.testing import TestCase, TestDataSpecification


class TestGetCatalogPath:
    """Tests for get_catalog_path function."""

    def test_returns_none_when_esgf_data_dir_is_none(self):
        """Test that get_catalog_path returns None when ESGF_DATA_DIR is None."""
        with patch("climate_ref.testing.ESGF_DATA_DIR", None):
            result = get_catalog_path("provider", "diagnostic", "test_case")
            assert result is None

    def test_returns_path_when_esgf_data_dir_exists(self, tmp_path):
        """Test that get_catalog_path returns correct path."""
        with patch("climate_ref.testing.ESGF_DATA_DIR", tmp_path):
            result = get_catalog_path("my-provider", "my-diag", "default")
            assert result == tmp_path / ".catalogs" / "my-provider" / "my-diag" / "default.yaml"


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
