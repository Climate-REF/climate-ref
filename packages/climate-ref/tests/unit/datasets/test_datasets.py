from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from climate_ref.datasets import get_dataset_adapter
from climate_ref.datasets.base import DatasetAdapter
from climate_ref_core.datasets import SourceDatasetType


class MockDatasetAdapter(DatasetAdapter):
    dataset_model: pd.DataFrame
    slug_column: str = "dataset_slug"
    dataset_specific_metadata: tuple[str, ...] = ("metadata1", "metadata2", "dataset_slug")
    file_specific_metadata: tuple[str, ...] = ("file_name", "file_size")

    def pretty_subset(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        # Just an example implementation that returns the file_name column
        return data_catalog[["file_name"]]

    def find_local_datasets(self, directories: Sequence[Path]) -> pd.DataFrame:
        # Mock implementation, return a DataFrame with fake data
        _dir = directories[0]
        data = {
            "dataset_slug": [f"{_dir.stem}_001", f"{_dir.stem}_001"],
            "metadata1": ["value1", "value1"],
            "metadata2": ["value2", "value2"],
            "time_range": ["2020-01-01", "2020-01-01"],
            "file_name": [_dir.name, _dir.name + "_2"],
            "file_size": [100, 100],
        }
        return pd.DataFrame(data)

    def register_dataset(self, config, db, data_catalog_dataset: pd.DataFrame) -> pd.DataFrame | None:
        # Returning the input as a stand-in "registered" dataset
        return data_catalog_dataset


def test_validate_data_catalog_complete_data():
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))

    validated_catalog = adapter.validate_data_catalog(data_catalog)
    assert not validated_catalog.empty


def test_validate_data_catalog_extra_columns():
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))
    data_catalog["extra_column"] = "extra"

    adapter.validate_data_catalog(data_catalog)


def test_validate_data_catalog_missing_columns():
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))
    with pytest.raises(ValueError, match="Data catalog is missing required columns: {'metadata1'}"):
        adapter.validate_data_catalog(data_catalog.drop(columns=["metadata1"]))

    with pytest.raises(ValueError, match="Data catalog is missing required columns: {'file_name'}"):
        adapter.validate_data_catalog(data_catalog.drop(columns=["file_name"]))


def test_validate_data_catalog_metadata_variance(caplog):
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))
    # file_name differs between datasets
    adapter.dataset_specific_metadata = (*adapter.dataset_specific_metadata, "file_name")

    exp_message = "Dataset dataset_001 has varying metadata:\n   file_name  time_range\n0    dataset  2020-01-01\n1  dataset_2  2020-01-01"  # noqa: E501

    with pytest.raises(
        ValueError,
        match="Dataset specific metadata varies by dataset",
    ):
        adapter.validate_data_catalog(data_catalog)
    assert len(caplog.records) == 1
    assert caplog.records[0].message == exp_message

    caplog.clear()
    assert len(adapter.validate_data_catalog(data_catalog, skip_invalid=True)) == 0
    assert len(caplog.records) == 1
    assert caplog.records[0].message == exp_message


@pytest.mark.parametrize(
    "source_type, expected_adapter",
    [
        (SourceDatasetType.CMIP6.value, "climate_ref.datasets.cmip6.CMIP6DatasetAdapter"),
        (SourceDatasetType.obs4MIPs.value, "climate_ref.datasets.obs4mips.Obs4MIPsDatasetAdapter"),
    ],
)
def test_get_dataset_adapter_valid(source_type, expected_adapter):
    adapter = get_dataset_adapter(source_type)
    assert adapter.__class__.__module__ + "." + adapter.__class__.__name__ == expected_adapter


def test_get_dataset_adapter_invalid():
    with pytest.raises(ValueError, match="Unknown source type: INVALID_TYPE"):
        get_dataset_adapter("INVALID_TYPE")
