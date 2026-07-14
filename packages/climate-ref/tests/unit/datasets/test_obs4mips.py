import os
import shutil
import warnings
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter, parse_obs4mips
from climate_ref.datasets.utils import sort_data_catalog
from climate_ref.models.dataset import Obs4MIPsDataset
from climate_ref.testing import TEST_DATA_DIR


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
            Path("obs4REF")
            / "obs4REF"
            / "MOHC"
            / "HadISST-1-1"
            / "mon"
            / "ts"
            / "gn"
            / "v20210727"
            / "ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc",
            {
                "activity_id": "obs4MIPs",
                "end_time": "2019-07-16 12:00:00",
                "frequency": "mon",
                "grid": "1x1 degree latitude x longitude",
                "grid_label": "gn",
                "institution_id": "MOHC",
                "long_name": "Surface Temperature",
                "nominal_resolution": "250 km",
                "product": "observations",
                "realm": "atmos",
                "source_id": "HadISST-1-1",
                "source_type": "satellite_blended",
                "source_version_number": "1-1",
                "start_time": "1870-01-16 11:59:59.464417",
                "time_range": "1870-01-16 11:59:59.464417-2019-07-16 12:00:00",
                "units": "K",
                "variable_id": "ts",
                "variant_label": "PCMDI",
                "version": "v20210727",
                "vertical_levels": 1,
                "path": str(
                    TEST_DATA_DIR
                    / "sample-data"
                    / "obs4REF"
                    / "obs4REF"
                    / "MOHC"
                    / "HadISST-1-1"
                    / "mon"
                    / "ts"
                    / "gn"
                    / "v20210727"
                    / "ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc"
                ),
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
            / "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc",
            {
                "INVALID_ASSET": str(TEST_DATA_DIR)
                + "/sample-data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/Amon/tas/gn/v20191115/"
                + "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc",
                "TRACEBACK": str(TEST_DATA_DIR)
                + "/sample-data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r1i1p1f1/Amon/tas/gn/v20191115/"
                + "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc"
                + " is not an obs4MIPs or obs4REF dataset",
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

        catalog_regression(sort_data_catalog(df), basename="obs4mips_catalog_db")

    @pytest.mark.xfail(reason="The database seems to store only the latest version of a dataset.")
    def test_round_trip(self, db_seeded, obs4mips_data_catalog, sample_data_dir):
        # Indexes and ordering may be different
        adapter = Obs4MIPsDatasetAdapter()
        local_data_catalog = (
            obs4mips_data_catalog.drop(columns=["time_range"])
            .sort_values(["instance_id"])
            .reset_index(drop=True)
        )

        db_data_catalog = adapter.load_catalog(db_seeded).sort_values(["instance_id"]).reset_index(drop=True)

        pd.testing.assert_frame_equal(local_data_catalog, db_data_catalog, check_like=True)

    def test_load_local_datasets(self, sample_data_dir, catalog_regression):
        adapter = Obs4MIPsDatasetAdapter()
        data_catalog = adapter.find_local_datasets(str(sample_data_dir / "obs4REF"))

        # TODO: add time_range to the db?
        assert sorted(data_catalog.columns.tolist()) == sorted(
            [*adapter.dataset_specific_metadata, *adapter.file_specific_metadata, "time_range"]
        )

        catalog_regression(
            sort_data_catalog(data_catalog),
            basename="obs4mips_catalog_local",
        )

    def test_load_local_CMIP6_datasets(self, sample_data_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with pytest.raises(ValueError) as excinfo:
                adapter = Obs4MIPsDatasetAdapter()
                adapter.find_local_datasets(str(sample_data_dir) + "/CMIP6")
            assert str(excinfo.value) == "No obs4MIPs-compliant datasets found"

    def test_empty_directory_exception(self, test_empty_dir):
        with pytest.raises(ValueError, match="No files matching"):
            adapter = Obs4MIPsDatasetAdapter()
            adapter.find_local_datasets(test_empty_dir)


def _write_file_without_variable_long_name(path: Path) -> None:
    """Write an obs4MIPs file whose variable carries no ``long_name`` attribute."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.activity_id = "obs4MIPs"
        ds.frequency = "mon"
        ds.grid = "site"
        ds.grid_label = "site"
        ds.institution_id = "TESTORG"
        ds.nominal_resolution = "site"
        ds.realm = "land"
        ds.product = "site-observations"
        ds.source_id = "TEST-SRC"
        ds.source_type = "insitu"
        ds.variable_id = "gpp"
        ds.variant_label = "v1"
        ds.source_version_number = "1"

        ds.createDimension("time", 3)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 1850-01-01"
        time_var.calendar = "standard"
        time_var[:] = [0, 30, 60]

        # units but deliberately no long_name -- obs4MIPs does not require it, and the published
        # obs4REF FLUXNET2015 gpp dataset omits it.
        gpp_var = ds.createVariable("gpp", "f4", ("time",))
        gpp_var.units = "g m-2 d-1"
        gpp_var[:] = np.array([1.0, 2.0, 3.0])


@pytest.fixture
def obs4mips_dir_without_long_name(tmp_path):
    fixture_dir = tmp_path / "no_long_name"
    _write_file_without_variable_long_name(
        fixture_dir / "obs4MIPs" / "TESTORG" / "TEST-SRC" / "mon" / "gpp" / "site" / "v1" / "gpp_mon.nc"
    )
    return fixture_dir


class TestObs4MIPsMissingLongName:
    """``long_name`` is optional metadata: a file without it must still ingest.

    It is read from the *variable's* attributes, which obs4MIPs does not require. The published
    obs4REF FLUXNET2015 ``gpp`` dataset has none, and a ``NOT NULL`` column previously turned that
    into an ``IntegrityError`` that aborted the entire obs4MIPs ingest.
    """

    def test_parse_returns_none_long_name(self, obs4mips_dir_without_long_name):
        file = next(obs4mips_dir_without_long_name.rglob("*.nc"))

        result = parse_obs4mips(str(file))

        assert result["long_name"] is None
        # The rest of the metadata still parses, so the dataset is not otherwise degraded
        assert result["variable_id"] == "gpp"
        assert result["units"] == "g m-2 d-1"

    def test_registers_with_null_long_name(self, db, obs4mips_dir_without_long_name):
        adapter = Obs4MIPsDatasetAdapter()
        data_catalog = adapter.find_local_datasets(obs4mips_dir_without_long_name)
        assert len(data_catalog) == 1

        with db.session.begin():
            adapter.register_dataset(db, data_catalog)

        dataset = db.session.query(Obs4MIPsDataset).one()
        assert dataset.long_name is None
        assert dataset.variable_id == "gpp"
