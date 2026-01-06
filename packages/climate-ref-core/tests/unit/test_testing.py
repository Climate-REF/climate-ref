from unittest.mock import MagicMock

import pandas as pd
import pytest

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.testing import (
    TestCase,
    TestDataSpecification,
    load_datasets_from_yaml,
    save_datasets_to_yaml,
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
        assert test_case.datasets is None
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

    def test_init_with_datasets(self):
        """Test initialization with explicit datasets."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)
        test_case = TestCase(
            name="default",
            description="Test with datasets",
            datasets=mock_datasets,
        )
        assert test_case.datasets == mock_datasets

    def test_init_with_datasets_file(self):
        """Test initialization with datasets file path."""
        test_case = TestCase(
            name="default",
            description="Test with file",
            datasets_file="tests/data/datasets.yaml",
        )
        assert test_case.datasets_file == "tests/data/datasets.yaml"

    def test_resolve_datasets_explicit(self):
        """Test resolving datasets when explicit datasets are provided."""
        mock_datasets = MagicMock(spec=ExecutionDatasetCollection)
        mock_diagnostic = MagicMock()

        test_case = TestCase(
            name="default",
            description="Test with datasets",
            datasets=mock_datasets,
        )

        result = test_case.resolve_datasets(None, mock_diagnostic)
        assert result == mock_datasets

    def test_resolve_datasets_file_missing_package_dir(self):
        """Test error when datasets_file is used without package_dir."""
        mock_diagnostic = MagicMock()

        test_case = TestCase(
            name="default",
            description="Test with file",
            datasets_file="tests/data/datasets.yaml",
        )

        with pytest.raises(ValueError, match="package_dir required"):
            test_case.resolve_datasets(None, mock_diagnostic)

    def test_resolve_datasets_no_catalog(self):
        """Test error when no datasets, file, or catalog provided."""
        mock_diagnostic = MagicMock()

        test_case = TestCase(
            name="default",
            description="Test without data",
        )

        with pytest.raises(ValueError, match="Cannot resolve datasets"):
            test_case.resolve_datasets(None, mock_diagnostic)


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

    def test_default_property(self):
        """Test getting the default test case."""
        case1 = TestCase(name="default", description="Default")
        case2 = TestCase(name="other", description="Other")

        spec = TestDataSpecification(test_cases=(case1, case2))

        assert spec.default == case1
        assert spec.default.name == "default"

    def test_default_property_missing(self):
        """Test error when no default test case exists."""
        case1 = TestCase(name="not-default", description="Not default")

        spec = TestDataSpecification(test_cases=(case1,))

        with pytest.raises(StopIteration):
            _ = spec.default

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
