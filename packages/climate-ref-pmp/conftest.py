import pandas as pd
import pytest

from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files
from climate_ref_core.datasets import SourceDatasetType


@pytest.fixture(scope="session")
def pmp_data_catalog(sample_data_dir) -> pd.DataFrame:
    # Fetch the PMP data from the registry
    registry = dataset_registry_manager["pmp-climatology"]
    fetch_all_files(registry, "pmp-climatology", output_dir=None)

    # Ingest the data
    adapter = PMPClimatologyDatasetAdapter()
    return adapter.find_local_datasets(registry.abspath / "PMP_obs4MIPsClims")


@pytest.fixture(scope="session")
def data_catalog(cmip6_data_catalog, obs4mips_data_catalog, obs4ref_data_catalog, pmp_data_catalog):
    return {
        SourceDatasetType.CMIP6: cmip6_data_catalog,
        SourceDatasetType.obs4MIPs: obs4mips_data_catalog,
        SourceDatasetType.obs4REF: obs4ref_data_catalog,
        SourceDatasetType.PMPClimatology: pmp_data_catalog,
    }
