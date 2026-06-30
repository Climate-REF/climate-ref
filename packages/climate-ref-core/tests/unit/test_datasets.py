import pandas as pd
import pytest

from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
    select_latest_version,
    version_sort_key,
)


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
def execution_dataset(dataset_collection) -> ExecutionDatasetCollection:
    return ExecutionDatasetCollection({SourceDatasetType.CMIP6: dataset_collection})


class TestExecutionDatasetCollection:
    def test_repr(self, execution_dataset: ExecutionDatasetCollection) -> None:
        assert repr(execution_dataset) == f"ExecutionDatasetCollection({execution_dataset._collection})"

    def test_get_item(self, execution_dataset):
        assert execution_dataset["cmip6"] == execution_dataset._collection[SourceDatasetType.CMIP6]
        assert (
            execution_dataset[SourceDatasetType.CMIP6]
            == execution_dataset._collection[SourceDatasetType.CMIP6]
        )

    def test_get_item_missing(self, execution_dataset):
        with pytest.raises(KeyError):
            execution_dataset["cmip7"]

    def test_iter(self, execution_dataset):
        assert tuple(iter(execution_dataset)) == tuple(iter(execution_dataset._collection))

    def test_keys(self, execution_dataset):
        assert execution_dataset.keys() == execution_dataset._collection.keys()

    def test_values(self, execution_dataset):
        assert tuple(execution_dataset.values()) == tuple(execution_dataset._collection.values())

    def test_items(self, execution_dataset):
        assert execution_dataset.items() == execution_dataset._collection.items()

    def test_python_hash(self, execution_dataset, cmip6_data_catalog, data_regression):
        dataset_hash = hash(execution_dataset)

        # The python hash is different to the hash of the dataset
        assert hash(execution_dataset.hash) == dataset_hash
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
        data_regression.check(execution_dataset.hash, basename="execution_dataset_hash")


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


class TestVersionSortKey:
    """``version_sort_key`` orders versions numerically, not lexicographically."""

    def test_vn_form_orders_numerically(self):
        # The regression this fixes: v10 must beat v2, unlike string ``.max()``.
        assert version_sort_key("v10") > version_sort_key("v2")
        assert version_sort_key("v2") > version_sort_key("v0")

    def test_date_form_orders_chronologically(self):
        assert version_sort_key("v20250622") > version_sort_key("v20240101")

    def test_v0_is_zero(self):
        assert version_sort_key("v0") == 0

    @pytest.mark.parametrize("value", ["latest", "", "1", "v", "vX", None])
    def test_non_conforming_values_sort_lowest(self, value):
        assert version_sort_key(value) == -1


class TestSelectLatestVersion:
    """``select_latest_version`` keeps the numerically-latest rows, optionally per group."""

    def test_ungrouped_keeps_single_latest_version(self):
        df = pd.DataFrame({"version": ["v1", "v2", "v10"], "path": ["a", "b", "c"]})
        result = select_latest_version(df)
        assert list(result["version"]) == ["v10"]

    def test_grouped_keeps_latest_per_group(self):
        df = pd.DataFrame(
            {
                "source_id": ["A", "A", "B", "B"],
                "version": ["v2", "v10", "v9", "v1"],
                "path": ["a", "b", "c", "d"],
            }
        )
        result = select_latest_version(df, group_by=["source_id"])
        assert set(zip(result["source_id"], result["version"])) == {("A", "v10"), ("B", "v9")}

    def test_custom_version_column(self):
        df = pd.DataFrame({"ver": ["v2", "v10"], "path": ["a", "b"]})
        result = select_latest_version(df, version_column="ver")
        assert list(result["ver"]) == ["v10"]
