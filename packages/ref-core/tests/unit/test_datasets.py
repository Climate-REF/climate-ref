import pathlib

import pytest

from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType


class TestMetricDataset:
    def test_get_item(self, metric_dataset):
        assert metric_dataset["cmip6"] == metric_dataset._collection[SourceDatasetType.CMIP6]
        assert metric_dataset[SourceDatasetType.CMIP6] == metric_dataset._collection[SourceDatasetType.CMIP6]

    def test_get_item_missing(self, metric_dataset):
        with pytest.raises(KeyError):
            metric_dataset["cmip7"]

    def test_python_hash(self, metric_dataset, cmip6_data_catalog, data_regression):
        dataset_hash = hash(metric_dataset)

        # The python hash is different to the hash of the dataset
        assert hash(metric_dataset.hash) == dataset_hash
        assert isinstance(dataset_hash, int)

        # Check that the hash changes if the dataset changes
        assert dataset_hash != hash(
            MetricDataset(
                {
                    SourceDatasetType.CMIP6: DatasetCollection(
                        cmip6_data_catalog[cmip6_data_catalog.variable_id != "tas"], "instance_id"
                    )
                }
            )
        )

        # This will change if the data catalog changes
        # Specifically if more tas datasets are provided
        data_regression.check(metric_dataset.hash, basename="metric_dataset_hash")

    def test_to_abs_paths(self, metric_dataset):
        assert not metric_dataset["cmip6"].datasets.path.str.startswith("/absolute/data").all()

        result = metric_dataset.to_abs_paths(
            data_directory=pathlib.Path("/absolute/data"),
        )

        # A new object is returned
        assert id(result) != id(metric_dataset)

        # The paths are now absolute
        assert result["cmip6"].datasets.path.str.startswith("/absolute/data").all()


class TestDatasetCollection:
    def test_get_item(self, dataset_collection):
        expected = dataset_collection.datasets.instance_id
        assert dataset_collection["instance_id"].equals(expected)

    def test_get_attr(self, dataset_collection):
        expected = dataset_collection.datasets.instance_id
        assert dataset_collection.instance_id.equals(expected)

    def test_hash(self, dataset_collection, cmip6_data_catalog, data_regression):
        tas_datasets = cmip6_data_catalog[cmip6_data_catalog.variable_id == "tas"]
        dataset_hash = hash(DatasetCollection(tas_datasets, "instance_id"))
        assert isinstance(dataset_hash, int)

        assert dataset_hash != hash(DatasetCollection(tas_datasets.iloc[[0, 1]], "instance_id"))

        # This hash will change if the data catalog changes
        # Specifically if more tas datasets are provided
        data_regression.check(dataset_hash, basename="dataset_collection_hash")
