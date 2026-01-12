"""Tests for CMIP7DatasetAdapter."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_ref.datasets.cmip7 import CMIP7DatasetAdapter, parse_cmip7_file
from climate_ref_core.cmip6_to_cmip7 import convert_cmip6_dataset, create_cmip7_path


@pytest.fixture
def sample_cmip7_dataset():
    """Create a minimal CMIP7-style dataset for testing."""
    time = np.arange(12)
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 10)
    rng = np.random.default_rng(42)

    data = rng.random((len(time), len(lat), len(lon)))

    # Create a CMIP6 dataset and convert it to CMIP7
    ds_cmip6 = xr.Dataset(
        {"tas": (["time", "lat", "lon"], data)},
        coords={
            "time": pd.date_range("2000-01-01", periods=12, freq="ME"),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "variable_id": "tas",
            "table_id": "Amon",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "member_id": "r1i1p1f1",
            "institution_id": "CSIRO",
            "activity_id": "CMIP",
            "grid_label": "gn",
            "version": "v20191115",
            "Conventions": "CF-1.6",
        },
    )

    ds_cmip7 = convert_cmip6_dataset(ds_cmip6, rename_variables=False)
    return ds_cmip7


@pytest.fixture
def cmip7_file(tmp_path, sample_cmip7_dataset):
    """Write a CMIP7 dataset to a file."""
    # Build CMIP7 DRS path
    cmip7_subpath = create_cmip7_path(sample_cmip7_dataset.attrs)
    cmip7_dir = tmp_path / cmip7_subpath
    cmip7_dir.mkdir(parents=True, exist_ok=True)

    filepath = cmip7_dir / "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200012.nc"
    sample_cmip7_dataset.to_netcdf(filepath)
    return filepath


class TestParseCmip7File:
    def test_parse_valid_file(self, cmip7_file, sample_cmip7_dataset):
        result = parse_cmip7_file(str(cmip7_file))

        assert result["path"] == str(cmip7_file)
        assert result["mip_era"] == "CMIP7"
        assert result["variable_id"] == "tas"
        assert result["source_id"] == "ACCESS-ESM1-5"
        assert result["experiment_id"] == "historical"
        assert result["variant_label"] == "r1i1p1f1"
        assert result["institution_id"] == "CSIRO"
        assert result["activity_id"] == "CMIP"
        assert result["grid_label"] == "gn"
        assert result["region"] == "GLB"
        assert "branding_suffix" in result
        assert result["start_time"] is not None
        assert result["end_time"] is not None

    def test_parse_missing_file(self, tmp_path):
        result = parse_cmip7_file(str(tmp_path / "nonexistent.nc"))
        assert result == {}


class TestCMIP7DatasetAdapter:
    def test_adapter_attributes(self):
        adapter = CMIP7DatasetAdapter()
        assert adapter.slug_column == "instance_id"
        assert adapter.version_metadata == "version"
        assert "mip_era" in adapter.dataset_specific_metadata
        assert "region" in adapter.dataset_specific_metadata
        assert "branding_suffix" in adapter.dataset_specific_metadata

    def test_catalog_empty(self, db):
        adapter = CMIP7DatasetAdapter()
        df = adapter.load_catalog(db)
        assert df.empty

    def test_find_local_datasets(self, tmp_path, cmip7_file):
        adapter = CMIP7DatasetAdapter()
        df = adapter.find_local_datasets(tmp_path)

        assert len(df) == 1
        assert df["mip_era"].iloc[0] == "CMIP7"
        assert df["variable_id"].iloc[0] == "tas"
        assert df["source_id"].iloc[0] == "ACCESS-ESM1-5"
        assert "instance_id" in df.columns
        assert df["instance_id"].iloc[0].startswith("CMIP7.")

    def test_find_local_datasets_empty_dir(self, tmp_path):
        adapter = CMIP7DatasetAdapter()
        df = adapter.find_local_datasets(tmp_path)
        assert df.empty

    def test_dataset_id_metadata(self):
        adapter = CMIP7DatasetAdapter()
        # CMIP7 DRS components
        expected = (
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "region",
            "frequency",
            "variable_id",
            "branding_suffix",
            "grid_label",
        )
        assert adapter.dataset_id_metadata == expected

    def test_register_and_load_dataset(self, config, db, tmp_path, cmip7_file):
        adapter = CMIP7DatasetAdapter()

        # Find local datasets
        data_catalog = adapter.find_local_datasets(tmp_path)
        assert len(data_catalog) == 1

        # Register the dataset
        with db.session.begin():
            for instance_id, data_catalog_dataset in data_catalog.groupby(adapter.slug_column):
                adapter.register_dataset(config, db, data_catalog_dataset)

        # Load and verify
        loaded_catalog = adapter.load_catalog(db)
        assert len(loaded_catalog) == 1
        assert loaded_catalog["mip_era"].iloc[0] == "CMIP7"
        assert loaded_catalog["variable_id"].iloc[0] == "tas"
