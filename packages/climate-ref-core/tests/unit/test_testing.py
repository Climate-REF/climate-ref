import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.testing import (
    RegressionValidator,
    TestCase,
    TestCasePaths,
    TestDataSpecification,
    _get_provider_test_data_dir,
    load_datasets_from_yaml,
    save_datasets_to_yaml,
    validate_cmec_bundles,
)


class TestTestCase:
    """Tests for TestCase class."""

    def test_init_basic(self):
        """Test basic initialization with required fields only."""
        test_case = TestCase(
            name="default",
            description="Default test case",
        )
        assert test_case.name == "default"
        assert test_case.description == "Default test case"
        assert test_case.requests is None
        assert test_case.datasets_file is None

    def test_init_with_requests(self):
        """Test initialization with ESGF requests."""
        mock_request = MagicMock()
        test_case = TestCase(
            name="default",
            description="Test with requests",
            requests=(mock_request,),
        )
        assert test_case.requests == (mock_request,)

    def test_init_with_datasets_file(self):
        """Test initialization with datasets file path."""
        test_case = TestCase(
            name="default",
            description="Test with file",
            datasets_file="tests/data/datasets.yaml",
        )
        assert test_case.datasets_file == "tests/data/datasets.yaml"


class TestTestDataSpecification:
    """Tests for TestDataSpecification class."""

    def test_init_empty(self):
        """Test initialization with no test cases."""
        spec = TestDataSpecification()
        assert spec.test_cases == ()
        assert spec.case_names == []

    def test_init_with_cases(self):
        """Test initialization with multiple test cases."""
        case1 = TestCase(name="default", description="Default")
        case2 = TestCase(name="edge-case", description="Edge case")

        spec = TestDataSpecification(test_cases=(case1, case2))

        assert len(spec.test_cases) == 2
        assert spec.case_names == ["default", "edge-case"]

    def test_get_case(self):
        """Test getting a test case by name."""
        case1 = TestCase(name="default", description="Default")
        case2 = TestCase(name="edge-case", description="Edge case")

        spec = TestDataSpecification(test_cases=(case1, case2))

        assert spec.get_case("default") == case1
        assert spec.get_case("edge-case") == case2

    def test_get_case_missing(self):
        """Test error when requested case doesn't exist."""
        case1 = TestCase(name="default", description="Default")

        spec = TestDataSpecification(test_cases=(case1,))

        with pytest.raises(StopIteration):
            spec.get_case("nonexistent")

    def test_has_case(self):
        """Test checking if a case exists."""
        case1 = TestCase(name="default", description="Default")

        spec = TestDataSpecification(test_cases=(case1,))

        assert spec.has_case("default") is True
        assert spec.has_case("nonexistent") is False

    def test_case_names(self):
        """Test getting list of case names."""
        case1 = TestCase(name="default", description="Default")
        case2 = TestCase(name="edge-case", description="Edge case")
        case3 = TestCase(name="minimal", description="Minimal")

        spec = TestDataSpecification(test_cases=(case1, case2, case3))

        assert spec.case_names == ["default", "edge-case", "minimal"]


class TestTestCasePaths:
    """Tests for TestCasePaths class."""

    def test_from_test_data_dir(self, tmp_path):
        """Test creating from explicit test data dir."""
        paths = TestCasePaths.from_test_data_dir(tmp_path, "my-diag", "default")

        assert paths.root == tmp_path / "my-diag" / "default"
        assert paths.catalog == tmp_path / "my-diag" / "default" / "catalog.yaml"
        assert paths.catalog_paths == tmp_path / "my-diag" / "default" / "catalog.paths.yaml"
        assert paths.regression == tmp_path / "my-diag" / "default" / "regression"

    def test_test_data_dir_property(self, tmp_path):
        """Test that test_data_dir returns the base directory."""
        paths = TestCasePaths.from_test_data_dir(tmp_path, "my-diag", "default")
        assert paths.test_data_dir == tmp_path

    def test_exists_false_when_not_created(self, tmp_path):
        """Test exists returns False when directory doesn't exist."""
        paths = TestCasePaths.from_test_data_dir(tmp_path, "my-diag", "default")
        assert not paths.exists()

    def test_create_makes_directory(self, tmp_path):
        """Test create makes the test case directory."""
        paths = TestCasePaths.from_test_data_dir(tmp_path, "my-diag", "default")
        assert not paths.root.exists()

        paths.create()

        assert paths.root.exists()
        assert paths.root.is_dir()

    def test_different_test_cases(self, tmp_path):
        """Test different test case names produce different paths."""
        default = TestCasePaths.from_test_data_dir(tmp_path, "diag", "default")
        custom = TestCasePaths.from_test_data_dir(tmp_path, "diag", "custom")

        assert default.root != custom.root
        assert default.regression != custom.regression


class TestYamlSerialization:
    """Tests for YAML serialization functions."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test that saving and loading datasets preserves data."""
        # Create test data
        df = pd.DataFrame(
            {
                "instance_id": ["CMIP6.test.dataset1", "CMIP6.test.dataset2"],
                "source_id": ["MODEL1", "MODEL2"],
                "variable_id": ["tas", "tas"],
                "path": ["/path/to/file1.nc", "/path/to/file2.nc"],
            }
        )

        collection = DatasetCollection(
            datasets=df,
            slug_column="instance_id",
            selector=(("source_id", "MODEL1"),),
        )

        datasets = ExecutionDatasetCollection({SourceDatasetType.CMIP6: collection})

        # Save to YAML
        yaml_path = tmp_path / "test_datasets.yaml"
        save_datasets_to_yaml(datasets, yaml_path)

        assert yaml_path.exists()

        # Load from YAML
        loaded = load_datasets_from_yaml(yaml_path)

        # Verify
        assert SourceDatasetType.CMIP6 in loaded
        loaded_collection = loaded[SourceDatasetType.CMIP6]
        assert loaded_collection.slug_column == "instance_id"
        assert len(loaded_collection.datasets) == 2

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test that save creates parent directories if needed."""
        df = pd.DataFrame({"instance_id": ["test"], "path": ["/test.nc"]})
        collection = DatasetCollection(datasets=df, slug_column="instance_id", selector=())
        datasets = ExecutionDatasetCollection({SourceDatasetType.CMIP6: collection})

        yaml_path = tmp_path / "nested" / "deep" / "test.yaml"
        assert not yaml_path.parent.exists()

        save_datasets_to_yaml(datasets, yaml_path)

        assert yaml_path.exists()
        assert yaml_path.parent.exists()

    def test_load_with_selector(self, tmp_path):
        """Test loading YAML with selector information."""
        yaml_content = """
cmip6:
  slug_column: instance_id
  selector:
    source_id: ACCESS-ESM1-5
    experiment_id: historical
  datasets:
    - instance_id: CMIP6.test.dataset
      path: /path/to/file.nc
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        loaded = load_datasets_from_yaml(yaml_path)

        collection = loaded[SourceDatasetType.CMIP6]
        # Selector is stored as sorted tuple of tuples
        selector_dict = dict(collection.selector)
        assert selector_dict["source_id"] == "ACCESS-ESM1-5"
        assert selector_dict["experiment_id"] == "historical"

    def test_load_multiple_source_types(self, tmp_path):
        """Test loading YAML with multiple source types."""
        yaml_content = """
cmip6:
  slug_column: instance_id
  selector: {}
  datasets:
    - instance_id: CMIP6.test
      path: /cmip6.nc
obs4mips:
  slug_column: instance_id
  selector: {}
  datasets:
    - instance_id: obs4MIPs.test
      path: /obs.nc
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        loaded = load_datasets_from_yaml(yaml_path)

        assert SourceDatasetType.CMIP6 in loaded
        assert SourceDatasetType.obs4MIPs in loaded

    def test_load_empty_datasets(self, tmp_path):
        """Test loading YAML with empty datasets list."""
        yaml_content = """
cmip6:
  slug_column: instance_id
  selector: {}
  datasets: []
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        loaded = load_datasets_from_yaml(yaml_path)

        collection = loaded[SourceDatasetType.CMIP6]
        assert len(collection.datasets) == 0

    def test_load_without_paths_file(self, tmp_path):
        """Test loading YAML when paths file does not exist."""
        yaml_content = """
cmip6:
  slug_column: instance_id
  selector: {}
  datasets:
    - instance_id: CMIP6.test
      variable_id: tas
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        # Ensure paths file does not exist
        paths_file = tmp_path / "test.paths.yaml"
        assert not paths_file.exists()

        loaded = load_datasets_from_yaml(yaml_path)

        collection = loaded[SourceDatasetType.CMIP6]
        assert len(collection.datasets) == 1
        # No path should be set since paths file doesn't exist
        assert "path" not in collection.datasets.columns or pd.isna(
            collection.datasets.iloc[0].get("path", None)
        )


class TestTestCasePathsFromDiagnostic:
    """Tests for TestCasePaths.from_diagnostic() and _get_provider_test_data_dir()."""

    def test_from_diagnostic_returns_none_when_module_not_loaded(self):
        """Test returns None when provider module is not in sys.modules."""
        mock_diag = MagicMock()
        mock_diag.__class__.__module__ = "nonexistent_module.diagnostics"

        result = TestCasePaths.from_diagnostic(mock_diag, "default")
        assert result is None

    def test_from_diagnostic_returns_correct_paths(self, tmp_path):
        """Test returns correct paths for loaded provider."""
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

    def test_from_diagnostic_returns_none_when_tests_dir_missing(self, tmp_path):
        """Test returns None when tests/ directory doesn't exist."""
        mock_module = MagicMock()
        mock_module.__file__ = str(tmp_path / "src" / "test_provider" / "__init__.py")

        mock_diag = MagicMock()
        mock_diag.__class__.__module__ = "test_provider.diagnostics"
        mock_diag.slug = "my-diag"

        # Don't create tests/ directory
        with patch.dict("sys.modules", {"test_provider": mock_module}):
            result = TestCasePaths.from_diagnostic(mock_diag, "default")
            assert result is None

    def test_get_provider_test_data_dir_module_no_file(self):
        """Test _get_provider_test_data_dir returns None when module has no __file__."""
        mock_module = MagicMock()
        mock_module.__file__ = None

        mock_diag = MagicMock()
        mock_diag.__class__.__module__ = "test_provider.diagnostics"

        with patch.dict("sys.modules", {"test_provider": mock_module}):
            result = _get_provider_test_data_dir(mock_diag)
            assert result is None


class TestValidateCmecBundles:
    """Tests for validate_cmec_bundles function."""

    def test_validate_cmec_bundles_success(self, tmp_path):
        """Test successful validation of CMEC bundles."""
        # Create valid metric bundle - RESULTS must have nested dicts with scalars at leaf level
        # The leaf dict maps dimension values (from last dimension) to numbers
        metric_bundle = {
            "DIMENSIONS": {
                "json_structure": ["source_id", "metric", "statistic"],
                "source_id": {"MODEL1": {}},
                "metric": {"rmse": {}},
                "statistic": {"value": {}},
            },
            "RESULTS": {"MODEL1": {"rmse": {"value": 0.5}}},
        }

        # Create valid output bundle with required provenance fields
        output_bundle = {
            "data": {},
            "plots": {},
            "metrics": {},
            "provenance": {
                "environment": {},
                "modeldata": "",
                "obsdata": "",
                "log": "",
            },
        }

        # Write bundles to files
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "diagnostic.json").write_text(json.dumps(metric_bundle))
        (output_dir / "output.json").write_text(json.dumps(output_bundle))

        # Create mock diagnostic with matching facets
        mock_diagnostic = MagicMock()
        mock_diagnostic.facets = ("source_id", "metric", "statistic")

        # Create mock result with spec to avoid MagicMock attribute issues
        mock_result = MagicMock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = Path("diagnostic.json")
        mock_result.output_bundle_filename = Path("output.json")
        mock_result.to_output_path = lambda f: output_dir / f

        validate_cmec_bundles(mock_diagnostic, mock_result)

    def test_validate_cmec_bundles_fails_on_unsuccessful(self, tmp_path):
        """Test validation fails when result is not successful."""
        mock_diagnostic = MagicMock()
        mock_result = MagicMock()
        mock_result.successful = False

        with pytest.raises(AssertionError, match="Execution failed"):
            validate_cmec_bundles(mock_diagnostic, mock_result)

    def test_validate_cmec_bundles_fails_on_dimension_mismatch(self, tmp_path):
        """Test validation fails when dimensions don't match diagnostic facets."""
        metric_bundle = {
            "DIMENSIONS": {
                "json_structure": ["source_id", "metric", "statistic"],
                "source_id": {"MODEL1": {}},
                "metric": {"rmse": {}},
                "statistic": {"value": {}},
            },
            "RESULTS": {"MODEL1": {"rmse": {"value": 0.5}}},
        }

        output_bundle = {
            "data": {},
            "plots": {},
            "metrics": {},
            "provenance": {"environment": {}, "modeldata": "", "obsdata": "", "log": ""},
        }

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "diagnostic.json").write_text(json.dumps(metric_bundle))
        (output_dir / "output.json").write_text(json.dumps(output_bundle))

        mock_diagnostic = MagicMock()
        mock_diagnostic.facets = ("different", "facets", "here")  # Mismatch

        mock_result = MagicMock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = Path("diagnostic.json")
        mock_result.output_bundle_filename = Path("output.json")
        mock_result.to_output_path = lambda f: output_dir / f

        with pytest.raises(AssertionError, match="don't match diagnostic facets"):
            validate_cmec_bundles(mock_diagnostic, mock_result)


class TestRegressionValidator:
    """Tests for RegressionValidator class."""

    def test_paths_property(self, tmp_path):
        """Test paths property returns correct TestCasePaths."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "my-diag"

        validator = RegressionValidator(
            diagnostic=mock_diagnostic,
            test_case_name="default",
            test_data_dir=tmp_path,
        )

        paths = validator.paths
        assert paths.root == tmp_path / "my-diag" / "default"

    def test_has_regression_data_false_when_missing(self, tmp_path):
        """Test has_regression_data returns False when no regression data."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "my-diag"

        validator = RegressionValidator(
            diagnostic=mock_diagnostic,
            test_case_name="default",
            test_data_dir=tmp_path,
        )

        assert validator.has_regression_data() is False

    def test_has_regression_data_true_when_present(self, tmp_path):
        """Test has_regression_data returns True when regression data exists."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "my-diag"

        # Create regression directory with diagnostic.json
        regression_dir = tmp_path / "my-diag" / "default" / "regression"
        regression_dir.mkdir(parents=True)
        (regression_dir / "diagnostic.json").write_text("{}")

        validator = RegressionValidator(
            diagnostic=mock_diagnostic,
            test_case_name="default",
            test_data_dir=tmp_path,
        )

        assert validator.has_regression_data() is True

    def test_load_regression_definition_missing_data(self, tmp_path):
        """Test load_regression_definition raises FileNotFoundError when no data."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "my-diag"

        validator = RegressionValidator(
            diagnostic=mock_diagnostic,
            test_case_name="default",
            test_data_dir=tmp_path,
        )

        with pytest.raises(FileNotFoundError, match="No catalog file"):
            validator.load_regression_definition(tmp_path / "tmp")

    def test_load_regression_definition_success(self, tmp_path):
        """Test load_regression_definition copies data and replaces placeholders."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "my-diag"

        metric_bundle = {
            "DIMENSIONS": {"json_structure": []},
            "RESULTS": {},
            "path": "<OUTPUT_DIR>/result.nc",
            "data_path": "<TEST_DATA_DIR>/input.nc",
        }

        validator = RegressionValidator(
            diagnostic=mock_diagnostic,
            test_case_name="default",
            test_data_dir=tmp_path,
        )

        # Create regression data
        validator.paths.create()
        validator.paths.catalog.write_text("{}")
        regression_dir = validator.paths.regression
        regression_dir.mkdir(parents=True)
        (regression_dir / "diagnostic.json").write_text(json.dumps(metric_bundle))
        (regression_dir / "output.json").write_text('{"html": {}}')

        work_dir = tmp_path / "work"
        definition = validator.load_regression_definition(work_dir)

        assert definition.diagnostic == mock_diagnostic
        assert definition.key == "test-default"
        assert definition.output_directory == work_dir / "output"

        # Check placeholders were replaced
        loaded_content = (work_dir / "output" / "diagnostic.json").read_text()
        assert "<OUTPUT_DIR>" not in loaded_content
        assert "<TEST_DATA_DIR>" not in loaded_content
        assert str(work_dir / "output") in loaded_content
        assert str(tmp_path) in loaded_content

    def test_validate_calls_validate_cmec_bundles(self, tmp_path):
        """Test validate method calls validate_cmec_bundles."""
        mock_diagnostic = MagicMock()
        mock_diagnostic.slug = "my-diag"
        mock_diagnostic.facets = ("source_id", "metric", "statistic")

        # Create output directory with valid bundles
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create mock result with spec and proper attributes
        mock_result = MagicMock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = Path("diagnostic.json")
        mock_result.output_bundle_filename = Path("output.json")
        mock_result.to_output_path = lambda f: output_dir / f
        mock_diagnostic.build_execution_result.return_value = mock_result

        # Create mock definition
        mock_definition = MagicMock()
        mock_definition.to_output_path = lambda f: output_dir / f if f else output_dir

        # Create valid CMEC bundles (RESULTS needs nested dicts with scalars at leaf)
        metric_bundle = {
            "DIMENSIONS": {
                "json_structure": ["source_id", "metric", "statistic"],
                "source_id": {"M1": {}},
                "metric": {"rmse": {}},
                "statistic": {"value": {}},
            },
            "RESULTS": {"M1": {"rmse": {"value": 0.5}}},
        }
        output_bundle = {
            "data": {},
            "plots": {},
            "metrics": {},
            "provenance": {"environment": {}, "modeldata": "", "obsdata": "", "log": ""},
        }
        (output_dir / "diagnostic.json").write_text(json.dumps(metric_bundle))
        (output_dir / "output.json").write_text(json.dumps(output_bundle))
        (output_dir / "out.log").touch()  # Create log file that validate touches

        validator = RegressionValidator(
            diagnostic=mock_diagnostic,
            test_case_name="default",
            test_data_dir=tmp_path,
        )

        validator.validate(mock_definition)

        # Verify build_execution_result was called
        mock_diagnostic.build_execution_result.assert_called_once_with(mock_definition)
