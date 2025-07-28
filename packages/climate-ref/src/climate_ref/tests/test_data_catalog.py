import pandas as pd
import pytest

from climate_ref.data_catalog import DataCatalog
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter


def test_data_catalog_finalise(cmip6_data_catalog, cmip6_data_catalog_drs, db):
    data_catalog = DataCatalog(database=db, adapter=CMIP6DatasetAdapter())

    assert (~cmip6_data_catalog_drs["finalised"]).all()
    finalised_df = data_catalog.finalise(cmip6_data_catalog_drs)
    assert finalised_df["finalised"].all()

    pd.testing.assert_frame_equal(cmip6_data_catalog, finalised_df[cmip6_data_catalog.columns])
    pd.testing.assert_frame_equal(cmip6_data_catalog, data_catalog.to_frame())


def test_data_catalog_to_frame(mocker):
    mock_db = mocker.Mock()
    data_catalog = DataCatalog(database=mock_db, adapter=CMIP6DatasetAdapter())
    # Mock the adapter's load_catalog method
    data_catalog.adapter.load_catalog = lambda db: pd.DataFrame([{"test": "test"}])
    df = data_catalog.to_frame()

    assert not df.empty
    assert "test" in df.columns


def test_data_catalog_df_caching(mocker):
    mock_db = mocker.Mock()
    data_catalog = DataCatalog(database=mock_db, adapter=CMIP6DatasetAdapter())
    data_catalog.adapter.load_catalog = lambda db: pd.DataFrame([{"test": "test"}])

    # The first call should load the dataframe
    assert data_catalog.to_frame() is not None
    assert data_catalog._df is not None

    # We can check that the load_catalog method is not called again
    data_catalog.adapter.load_catalog = lambda db: pytest.fail("load_catalog should not be called again")
    assert data_catalog.to_frame() is not None
