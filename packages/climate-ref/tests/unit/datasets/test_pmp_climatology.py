import os
import shutil

import pytest

from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter


@pytest.fixture
def test_empty_dir():
    dir_path = "test_empty_directory"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    shutil.rmtree(dir_path)


class TestPMPClimatologyAdapter:
    def test_catalog_empty(self, db):
        adapter = PMPClimatologyDatasetAdapter()
        df = adapter.load_catalog(db)
        assert df.empty

    def test_load_local_datasets(self, sample_data_dir, catalog_regression):
        adapter = PMPClimatologyDatasetAdapter()
        data_catalog = adapter.find_local_datasets(str(sample_data_dir) + "/obs4REF")

        # TODO: add time_range to the db?
        assert sorted(data_catalog.columns.tolist()) == sorted(
            [*adapter.dataset_specific_metadata, *adapter.file_specific_metadata, "time_range"]
        )

        catalog_regression(
            data_catalog.sort_values(["instance_id", "start_time"]), basename="pmp_catalog_local"
        )

    def test_load_local_CMIP6_datasets(self, sample_data_dir):
        with pytest.raises(ValueError) as excinfo:
            adapter = PMPClimatologyDatasetAdapter()
            adapter.find_local_datasets(str(sample_data_dir) + "/CMIP6")
        assert str(excinfo.value) == "No obs4MIPs-compliant datasets found"

    def test_empty_directory_exception(self, test_empty_dir):
        with pytest.raises(ValueError) as excinfo:
            adapter = PMPClimatologyDatasetAdapter()
            adapter.find_local_datasets(test_empty_dir)
        assert str(excinfo.value) == "No obs4MIPs-compliant datasets found"
