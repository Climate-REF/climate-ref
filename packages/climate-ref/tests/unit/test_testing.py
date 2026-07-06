"""
Tests for the test harness fixtures.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climate_ref.testing import (
    TestCaseRunner,
    assert_test_case_no_drift,
    create_no_drift_test,
)
from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)
from climate_ref_core.source_types import SourceDatasetType
from climate_ref_core.testing import (
    TestCase,
    TestCasePaths,
    TestDataSpecification,
)


class TestTestCasePathsFromDiagnostic:
    """Tests for TestCasePaths.from_diagnostic() with provider resolution."""

    def test_returns_none_when_module_not_loaded(self):
        """Test returns None when provider module is not in sys.modules."""
        mock_diag = MagicMock()
        mock_diag.__class__.__module__ = "nonexistent_module.diagnostics"

        result = TestCasePaths.from_diagnostic(mock_diag, "default")
        assert result is None

    def test_returns_correct_paths(self, tmp_path):
        """Test returns correct paths for loaded provider."""
        # Create mock module structure
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.__class__.__module__ = "test_provider.diagnostics"
        mock_diag.slug = "my-diag"

        # Create tests directory (indicates dev checkout)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir(parents=True)

        with patch.dict("sys.modules", {"test_provider": mock_module}):
            paths = TestCasePaths.from_diagnostic(mock_diag, "default")
            assert paths is not None
            assert paths.root == tmp_path / "tests" / "test-data" / "my-diag" / "default"
            assert paths.catalog == tmp_path / "tests" / "test-data" / "my-diag" / "default" / "catalog.yaml"
            assert paths.regression == tmp_path / "tests" / "test-data" / "my-diag" / "default" / "regression"

    def test_returns_none_when_tests_dir_missing(self, tmp_path):
        """Test returns None when tests/ directory doesn't exist (not a dev checkout)."""
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.__class__.__module__ = "test_provider.diagnostics"
        mock_diag.slug = "my-diag"

        # Don't create tests/ directory
        with patch.dict("sys.modules", {"test_provider": mock_module}):
            assert TestCasePaths.from_diagnostic(mock_diag, "default") is None


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
        # The runner delegates to Diagnostic.run with an ExecutionDefinition.
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

        # Mock the execution to avoid actual execution
        mock_result = MagicMock()
        mock_result.successful = True
        mock_diagnostic.run.return_value = mock_result

        result = runner.run(mock_diagnostic, "default", output_dir=tmp_path)

        assert result.successful
        # The runner delegates to Diagnostic.run with the regression-capture hook enabled.
        mock_diagnostic.run.assert_called_once()
        execution_definition = mock_diagnostic.run.call_args[0][0]
        assert mock_diagnostic.run.call_args.kwargs["capture_regression"] is True
        assert execution_definition.datasets == mock_datasets

    def test_run_without_paths(self, config, tmp_path):
        """Test running with explicit datasets provided to runner."""
        mock_datasets = ExecutionDatasetCollection(
            {
                # No 'path' column in the datasets DataFrame
                SourceDatasetType.CMIP6: DatasetCollection(
                    datasets=pd.DataFrame([{"instance_id": "test-instance"}]), slug_column="instance_id"
                )
            }
        )
        runner = TestCaseRunner(config=config, datasets=mock_datasets)

        mock_diagnostic = MagicMock()

        with pytest.raises(DatasetResolutionError, match="missing the required 'path' column"):
            runner.run(mock_diagnostic, "default", output_dir=tmp_path)

    def test_run_clean_removes_existing_output_dir(self, config, tmp_path):
        """``clean=True`` wipes a pre-existing output directory before running."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)
        mock_datasets.items.return_value = []
        runner = TestCaseRunner(config=config, datasets=mock_datasets)

        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "test-diag"
        mock_diagnostic.provider.slug = "test-provider"
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(TestCase(name="default", description="Default"),)
        )
        mock_diagnostic.run.return_value = MagicMock(successful=True)

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        stale = output_dir / "stale.txt"
        stale.write_text("stale")

        runner.run(mock_diagnostic, "default", output_dir=output_dir, clean=True)

        # The stale artefact from a prior run is gone, but the directory is recreated.
        assert output_dir.is_dir()
        assert not stale.exists()


class TestCreateNoDriftTest:
    """Tests for the per-provider drift-test factory."""

    def test_returns_marked_test(self, provider):
        """The factory returns a test function carrying the standard marks."""
        test_fn = create_no_drift_test(provider)

        marks = {mark.name for mark in test_fn.pytestmark}
        assert marks == {"slow", "test_cases", "parametrize"}

    def test_collects_provider_cases(self, provider, mock_diagnostic):
        """The parametrization contains one case per diagnostic test case."""
        mock_diagnostic.test_data_spec = TestDataSpecification(
            test_cases=(TestCase(name="default", description="Default case"),)
        )

        test_fn = create_no_drift_test(provider)

        parametrize = next(mark for mark in test_fn.pytestmark if mark.name == "parametrize")
        assert [param.id for param in parametrize.args[1]] == ["mock/default"]

    def _make_paths(self, *, catalog=True, manifest=True, regression=True):
        """Build a TestCasePaths mock with configurable file existence."""
        paths = MagicMock()
        paths.catalog.exists.return_value = catalog
        paths.manifest.exists.return_value = manifest
        paths.regression.exists.return_value = regression
        return paths

    def test_body_skips_when_no_test_data_dir(self, provider, config, tmp_path):
        """The generated test skips when the provider is not a development checkout."""
        test_fn = create_no_drift_test(provider)
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"

        with (
            patch("climate_ref.testing.TestCasePaths.from_diagnostic", return_value=None),
            pytest.raises(pytest.skip.Exception, match="not a development checkout"),
        ):
            test_fn(diagnostic, "default", config, tmp_path)

    def test_body_skips_when_catalog_missing(self, provider, config, tmp_path):
        """The generated test skips (pointing at fetch) when the catalog is absent."""
        test_fn = create_no_drift_test(provider)
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"
        paths = self._make_paths(catalog=False)

        with (
            patch("climate_ref.testing.TestCasePaths.from_diagnostic", return_value=paths),
            pytest.raises(pytest.skip.Exception, match="Run `ref test-cases fetch"),
        ):
            test_fn(diagnostic, "default", config, tmp_path)

    def test_body_skips_when_baseline_missing(self, provider, config, tmp_path):
        """The generated test skips when the committed baseline has not been minted."""
        test_fn = create_no_drift_test(provider)
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"
        paths = self._make_paths(manifest=False)

        with (
            patch("climate_ref.testing.TestCasePaths.from_diagnostic", return_value=paths),
            pytest.raises(pytest.skip.Exception, match="No committed baseline"),
        ):
            test_fn(diagnostic, "default", config, tmp_path)

    def test_body_delegates_when_everything_present(self, provider, config, tmp_path):
        """When catalog and baseline exist, the body delegates to assert_test_case_no_drift."""
        test_fn = create_no_drift_test(provider)
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"
        paths = self._make_paths()

        with (
            patch("climate_ref.testing.TestCasePaths.from_diagnostic", return_value=paths),
            patch("climate_ref.testing.assert_test_case_no_drift") as assert_no_drift,
        ):
            test_fn(diagnostic, "default", config, tmp_path)

        diagnostic.provider.configure.assert_called_once_with(config)
        assert_no_drift.assert_called_once_with(config, diagnostic, "default", paths, tmp_path)


class TestAssertTestCaseNoDrift:
    """Tests for the execute/build/compare drift assertion."""

    def test_raises_without_test_data_spec(self, config, tmp_path):
        """A diagnostic with no test_data_spec cannot be drift-checked."""
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"
        diagnostic.test_data_spec = None

        with pytest.raises(NoTestDataSpecError, match="no test_data_spec"):
            assert_test_case_no_drift(config, diagnostic, "default", MagicMock(), tmp_path)

    def _patch_stages(self, failures):
        """Patch the stage helpers and the manifest loader used by the drift assertion."""
        stages = "climate_ref.cli.test_cases._stages"
        return (
            patch(f"{stages}.baseline_placeholders"),
            patch(f"{stages}.stage_execute"),
            patch(f"{stages}.stage_build"),
            patch(f"{stages}.stage_compare", return_value=(failures, [])),
            patch("climate_ref.testing.load_datasets_from_yaml"),
            patch("climate_ref.testing.Manifest"),
        )

    def test_passes_when_no_failures(self, config, tmp_path):
        """The happy path wires execute -> build -> compare and returns quietly."""
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"
        diagnostic.provider.slug = "my-provider"
        paths = MagicMock()

        baseline_ph, execute, build, compare, load_yaml, manifest = self._patch_stages(failures=[])
        with baseline_ph, execute as execute_m, build as build_m, compare as compare_m, load_yaml, manifest:
            assert_test_case_no_drift(config, diagnostic, "default", paths, tmp_path)

        execute_m.assert_called_once()
        build_m.assert_called_once()
        compare_m.assert_called_once()
        # The output slot is created under the per-test work directory.
        assert (tmp_path / "slot").is_dir()

    def test_raises_on_drift(self, config, tmp_path):
        """Compare failures surface as an AssertionError naming the provider/diagnostic/case."""
        diagnostic = MagicMock()
        diagnostic.slug = "my-diag"
        diagnostic.provider.slug = "my-provider"
        paths = MagicMock()

        failures = ["diagnostic.json: value drift beyond tolerance"]
        baseline_ph, execute, build, compare, load_yaml, manifest = self._patch_stages(failures)
        with baseline_ph, execute, build, compare, load_yaml, manifest:
            with pytest.raises(AssertionError, match="my-provider/my-diag/default: committed bundle drift"):
                assert_test_case_no_drift(config, diagnostic, "default", paths, tmp_path)
