import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from cmip_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter, parse_obs4mips
from cmip_ref.testing import TEST_DATA_DIR


@pytest.fixture
def catalog_regression(data_regression, sample_data_dir):
    def check(df: pd.DataFrame, basename: str):
        # Strip the path to make the test more robust
        df["path"] = df["path"].str.replace(str(sample_data_dir), "{esgf_data_dir}")

        data_regression.check(df.to_dict(orient="records"), basename=basename)

    return check


@pytest.fixture
def test_empty_dir():
    dir_path = "test_empty_directory"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.mark.parametrize(
    "file_fragment, exp",
    (
        (
            Path("obs4MIPs")
            / "NASA-JPL"
            / "AIRS-2-1"
            / "ta"
            / "gn"
            / "v20201110"
            / "ta_AIRS-2-1_gn_200209-201609.nc",
            {
                "activity_id": "obs4MIPs",
                "frequency": "mon",
                "grid": "1x1 degree latxlon grid",
                "grid_label": "gn",
                "institution_id": "JPL",
                "nominal_resolution": "1x1 degree",
                "product": "observations",
                "realm": "atmos",
                "source_id": "AIRS-obs4MIPs-ta-v2.1",
                "source_type": "satellite_blended",
                "variable_id": "ta",
                "variant_label": "BE",
                "long_name": "Temperature",
                "units": "K",
                "vertical_levels": 17,
                "start_time": "2002-09-16 00:00:00",
                "end_time": "2016-09-16 00:00:00",
                "time_range": "2002-09-16 00:00:00-2016-09-16 00:00:00",
                "path": str(TEST_DATA_DIR)
                + "/sample-data/obs4MIPs/NASA-JPL/AIRS-2-1/ta/gn/v20201110/ta_AIRS-2-1_gn_200209-201609.nc",
                "source_version_number": "v20201110",
            },
        ),
        (
            Path("CMIP6")
            / "CMIP"
            / "CSIRO"
            / "ACCESS-ESM1-5"
            / "historical"
            / "r1i1p1f1"
            / "Amon"
            / "tas"
            / "gn"
            / "v20191115"
            / "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-201412.nc",
            {
                "INVALID_ASSET": str(TEST_DATA_DIR)
                + "/sample-data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/Amon/tas/gn/v20191115/"
                + "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-201412.nc",
                "TRACEBACK": str(TEST_DATA_DIR)
                + "/sample-data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/Amon/tas/gn/v20191115/"
                + "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-201412.nc"
                + " is not an obs4MIPs dataset",
            },
        ),
    ),
)
def test_parse_obs4mips(sample_data_dir, file_fragment, exp):
    result = parse_obs4mips(str(sample_data_dir / file_fragment))

    assert result == exp


class Testobs4MIPsAdapter:
    def test_catalog_empty(self, db):
        adapter = Obs4MIPsDatasetAdapter()
        df = adapter.load_catalog(db)
        assert df.empty

    def test_load_catalog(self, db_seeded, catalog_regression, sample_data_dir):
        adapter = Obs4MIPsDatasetAdapter()
        df = adapter.load_catalog(db_seeded)
        for k in adapter.dataset_specific_metadata + adapter.file_specific_metadata:
            assert k in df.columns

        # The order of the rows may be flakey due to sqlite ordering and the created time resolution
        catalog_regression(df.sort_values(["instance_id", "start_time"]), basename="obs4mips_catalog_db")

    def test_round_trip(self, db_seeded, sample_data_dir):
        # Indexes and ordering may be different
        adapter = Obs4MIPsDatasetAdapter()
        local_data_catalog = (
            adapter.find_local_datasets(str(sample_data_dir) + "/obs4MIPs")
            .drop(columns=["time_range"])
            .sort_values(["start_time"])
            .reset_index(drop=True)
        )

        db_data_catalog = adapter.load_catalog(db_seeded).sort_values(["start_time"]).reset_index(drop=True)

        # TODO: start_time has a different dtype from the database due to pandas dt coercion
        db_data_catalog["start_time"] = db_data_catalog["start_time"].astype(object)
        db_data_catalog["end_time"] = db_data_catalog["end_time"].astype(object)
        db_data_catalog["vertical_levels"] = db_data_catalog["vertical_levels"].astype(float)
        pd.testing.assert_frame_equal(local_data_catalog, db_data_catalog, check_like=True)

    def test_load_local_datasets(self, sample_data_dir, catalog_regression):
        adapter = Obs4MIPsDatasetAdapter()
        data_catalog = adapter.find_local_datasets(str(sample_data_dir) + "/obs4MIPs")

        # TODO: add time_range to the db?
        assert sorted(data_catalog.columns.tolist()) == sorted(
            [*adapter.dataset_specific_metadata, *adapter.file_specific_metadata, "time_range"]
        )

        catalog_regression(
            data_catalog.sort_values(["instance_id", "start_time"]), basename="obs4mips_catalog_local"
        )

    def test_load_local_CMIP6_datasets(self, sample_data_dir):
        with pytest.raises(ValueError) as excinfo:
            adapter = Obs4MIPsDatasetAdapter()
            adapter.find_local_datasets(str(sample_data_dir) + "/CMIP6")
        assert str(excinfo.value) == "No obs4MIPs-compliant datasets found"

    def test_empty_directory_exception(self, test_empty_dir):
        with pytest.raises(ValueError) as excinfo:
            adapter = Obs4MIPsDatasetAdapter()
            adapter.find_local_datasets(test_empty_dir)
        assert str(excinfo.value) == "asset list provided is None. Please run `.get_assets()` first"
