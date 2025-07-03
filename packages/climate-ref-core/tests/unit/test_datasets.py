import pandas as pd
import pytest

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType


@pytest.fixture
def dataset_collection(cmip6_data_catalog) -> DatasetCollection:
    return DatasetCollection(
        selector=(("variable_id", "tas"),),
        datasets=cmip6_data_catalog[cmip6_data_catalog.variable_id == "tas"],
        slug_column="instance_id",
    )


@pytest.fixture
def dataset_collection_obs4mips(obs4mips_data_catalog) -> DatasetCollection:
    return DatasetCollection(
        obs4mips_data_catalog[obs4mips_data_catalog.variable_id == "ta"],
        "instance_id",
    )


@pytest.fixture
def metric_dataset(dataset_collection) -> ExecutionDatasetCollection:
    return ExecutionDatasetCollection({SourceDatasetType.CMIP6: dataset_collection})


class TestMetricDataset:
    def test_get_item(self, metric_dataset):
        assert metric_dataset["cmip6"] == metric_dataset._collection[SourceDatasetType.CMIP6]
        assert metric_dataset[SourceDatasetType.CMIP6] == metric_dataset._collection[SourceDatasetType.CMIP6]

    def test_get_item_missing(self, metric_dataset):
        with pytest.raises(KeyError):
            metric_dataset["cmip7"]

    def test_iter(self, metric_dataset):
        assert tuple(iter(metric_dataset)) == tuple(iter(metric_dataset._collection))

    def test_keys(self, metric_dataset):
        assert metric_dataset.keys() == metric_dataset._collection.keys()

    def test_values(self, metric_dataset):
        assert tuple(metric_dataset.values()) == tuple(metric_dataset._collection.values())

    def test_items(self, metric_dataset):
        assert metric_dataset.items() == metric_dataset._collection.items()

    def test_python_hash(self, metric_dataset, cmip6_data_catalog, data_regression):
        dataset_hash = hash(metric_dataset)

        # The python hash is different to the hash of the dataset
        assert hash(metric_dataset.hash) == dataset_hash
        assert isinstance(dataset_hash, int)

        # Check that the hash changes if the dataset changes
        assert dataset_hash != hash(
            ExecutionDatasetCollection(
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


class TestDatasetCollection:
    def test_get_item(self, dataset_collection):
        expected = dataset_collection.datasets.instance_id
        assert dataset_collection["instance_id"].equals(expected)

    def test_selector_ordered(self):
        dc = DatasetCollection(
            selector=(
                ("variable_id", "tas"),
                ("grid_label", "gn"),
            ),
            datasets=pd.DataFrame(),
            slug_column="instance_id",
        )
        # Alphabetically sorted by dimension
        assert dc.selector == (
            ("grid_label", "gn"),
            ("variable_id", "tas"),
        )

    def test_selector_dict(self):
        dc = DatasetCollection(
            selector=(
                ("variable_id", "tas"),
                ("grid_label", "gn"),
            ),
            datasets=pd.DataFrame(),
            slug_column="instance_id",
        )
        # Alphabetically sorted by dimension
        assert dc.selector_dict() == {
            "grid_label": "gn",
            "variable_id": "tas",
        }
        assert list(dc.selector_dict().keys()) == ["grid_label", "variable_id"]

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


class TestDatasetCollectionObs4MIPs:
    def test_get_item(self, dataset_collection_obs4mips):
        expected = dataset_collection_obs4mips.datasets.instance_id
        assert dataset_collection_obs4mips["instance_id"].equals(expected)

    def test_get_attr(self, dataset_collection_obs4mips):
        expected = dataset_collection_obs4mips.datasets.instance_id
        assert dataset_collection_obs4mips.instance_id.equals(expected)

    def test_hash(self, dataset_collection_obs4mips, obs4mips_data_catalog, data_regression):
        ts_datasets = obs4mips_data_catalog[obs4mips_data_catalog.variable_id == "ts"]
        dataset_hash = hash(DatasetCollection(ts_datasets, "instance_id"))
        assert isinstance(dataset_hash, int)

        assert dataset_hash != hash(DatasetCollection(ts_datasets.iloc[[0, 0]], "instance_id"))

        # This hash will change if the data catalog changes
        # Specifically if more tas datasets are provided
        data_regression.check(dataset_hash, basename="dataset_collection_obs4mips_hash")
