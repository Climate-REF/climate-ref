import pandas as pd
import pytest

from climate_ref.config import Config
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter


@pytest.fixture
def catalog_regression(data_regression, sample_data_dir):
    def check(df: pd.DataFrame, basename: str):
        # Strip the path to make the test more robust
        df["path"] = df["path"].str.replace(str(sample_data_dir), "{esgf_data_dir}")

        data_regression.check(df.to_dict(orient="records"), basename=basename)

    return check


@pytest.fixture(scope="session")
def cmip6_local_catalogs(sample_data, sample_data_dir):
    """Session-cached CMIP6 local catalogs keyed by parser type.

    Avoids redundant find_local_datasets calls across tests that
    parametrize over cmip6_parser.
    """
    results = {}
    for parser in ["complete", "drs"]:
        config = Config.default()
        config.cmip6_parser = parser
        adapter = CMIP6DatasetAdapter(config=config)
        results[parser] = adapter.find_local_datasets(sample_data_dir / "CMIP6")
    return results
