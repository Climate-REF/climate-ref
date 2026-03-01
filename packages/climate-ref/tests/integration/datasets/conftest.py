import pytest

from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter


@pytest.fixture
def adapter_data_dir(adapter_config, sample_data_dir, cmip7_converted_file):
    """Data directory appropriate for each adapter."""
    if adapter_config.adapter_cls is CMIP6DatasetAdapter:
        return sample_data_dir / "CMIP6"

    # Note we are only using a single dataset at the moment
    return cmip7_converted_file.parent


@pytest.fixture
def adapter_local_catalogs(adapter_config, adapter_data_dir, config):
    """Local catalogs keyed by parser type ("complete", "drs")."""
    results = {}
    for parser in ["complete", "drs"]:
        setattr(config, adapter_config.parser_config_attr, parser)
        adapter = adapter_config.adapter_cls(config=config)
        results[parser] = adapter.find_local_datasets(adapter_data_dir)
    return results
