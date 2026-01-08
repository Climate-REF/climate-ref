"""
Tests for the test harness fixtures.
"""

from unittest.mock import MagicMock

import pytest

from climate_ref.testing import (
    NoTestDataSpecError,
    TestCaseNotFoundError,
    TestCaseRunner,
)
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.testing import TestCase, TestDataSpecification


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

    def test_run_with_explicit_datasets(self, config, tmp_path):
        """Test running with explicit datasets in test case."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)
        runner = TestCaseRunner(config=config, datasets=mock_datasets)

        # Create mock diagnostic with test case that has explicit datasets
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"
        mock_diagnostic.provider.slug = "test-provider"
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(
                TestCase(
                    name="default",
                    description="Default test",
                    datasets=mock_datasets,
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


class TestTestCaseRunnerPytestFixture:
    """Tests for the pytest fixture wrapper."""

    def test_run_no_test_data_spec_skips(self, run_test_case):
        """Test that the fixture skips when diagnostic has no test_data_spec."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.test_data_spec = None
        mock_diagnostic.slug = "test-diag"

        with pytest.raises(pytest.skip.Exception):
            run_test_case.run(mock_diagnostic)

    def test_run_with_explicit_datasets(self, run_test_case, tmp_path):
        """Test running with explicit datasets via fixture."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"
        mock_diagnostic.provider.slug = "test-provider"
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(
                TestCase(
                    name="default",
                    description="Default test",
                    datasets=mock_datasets,
                ),
            )
        )

        mock_result = MagicMock()
        mock_result.successful = True
        mock_diagnostic.run.return_value = mock_result

        result = run_test_case.run(mock_diagnostic, "default", output_dir=tmp_path)

        assert result.successful
