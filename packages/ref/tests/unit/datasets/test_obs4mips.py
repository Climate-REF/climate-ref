import pandas as pd
import pytest

from cmip_ref.datasets.obs4mips import OBS4MIPSDatasetAdapter


@pytest.fixture
def catalog_regression(data_regression, sample_data_dir):
    def check(df: pd.DataFrame, basename: str):
        # Strip the path to make the test more robust
        df["path"] = df["path"].str.replace(str(sample_data_dir), "{esgf_data_dir}")

        data_regression.check(df.to_dict(orient="records"), basename=basename)

    return check


class Testobs4MIPsAdapter:
    def test_catalog_empty(self, db):
        adapter = OBS4MIPSDatasetAdapter()
        df = adapter.load_catalog(db)
        assert df.empty

    def test_load_catalog(self, db_seeded, catalog_regression, sample_data_dir):
        adapter = OBS4MIPSDatasetAdapter()
        df = adapter.load_catalog(db_seeded)
        for k in adapter.dataset_specific_metadata + adapter.file_specific_metadata:
            assert k in df.columns

        # The order of the rows may be flakey due to sqlite ordering and the created time resolution
        catalog_regression(df.sort_values(["instance_id", "start_time"]), basename="obs4mips_catalog_db")

    def test_round_trip(self, db_seeded, sample_data_dir):
        # Indexes and ordering may be different
        adapter = OBS4MIPSDatasetAdapter()
        local_data_catalog = (
            adapter.find_local_datasets(sample_data_dir)
            .drop(columns=["time_range"])
            .sort_values(["start_time"])
            .reset_index(drop=True)
        )

        db_data_catalog = adapter.load_catalog(db_seeded).sort_values(["start_time"]).reset_index(drop=True)

        # TODO: start_time has a different dtype from the database due to pandas dt coercion
        db_data_catalog["start_time"] = db_data_catalog["start_time"].astype(object)
        pd.testing.assert_frame_equal(local_data_catalog, db_data_catalog, check_like=True)

    def test_load_local_datasets(self, sample_data_dir, catalog_regression):
        adapter = OBS4MIPSDatasetAdapter()
        data_catalog = adapter.find_local_datasets(sample_data_dir)

        # TODO: add time_range to the db?
        assert sorted(data_catalog.columns.tolist()) == sorted(
            [*adapter.dataset_specific_metadata, *adapter.file_specific_metadata, "time_range"]
        )

        catalog_regression(
            data_catalog.sort_values(["instance_id", "start_time"]), basename="obs4mips_catalog_local"
        )
