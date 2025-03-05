"""
Data registries for PMP reference data and parameters.
"""

import importlib.resources

import pooch

PMP_VERSION = "v3.8.2"
_DATASETS = {
    "HadISST-1-1": "obs4MIPs_PCMDI_monthly/MOHC/HadISST-1-1/mon/ts/gn/v20210727/ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc",  # noqa
}
"""Map of dataset names to reference registry paths."""


def build_reference_data_registry(version: str) -> pooch.Pooch:
    """
    Build a pooch registry of reference data associated with PMP that isn't currently in obs4MIPs.

    Parameters
    ----------
    version : str
        The version of the data.

        Changing the version will invalidate the cache and force a re-download of the data.

    Returns
    -------
    pooch.Pooch
        The pooch registry.
    """
    registry = pooch.create(
        path=pooch.os_cache("pmp"),
        base_url="https://pcmdiweb.llnl.gov/pss/pmpdata/",
        version=version,
        env="REF_METRICS_PMP_DATA_DIR",
    )
    registry.load_registry(importlib.resources.open_binary("cmip_ref_metrics_pmp.registry", "reference.txt"))
    return registry


_REFERENCE_REGISTRY = build_reference_data_registry(version=PMP_VERSION)


def fetch_reference_data(dataset_id: str, registry: pooch.Pooch = _REFERENCE_REGISTRY) -> str:
    """
    Fetch the reference data associated with the dataset ID.

    Parameters
    ----------
    dataset_id : str
        The dataset ID.
    registry
        The registry to use to fetch the reference data.

        If none is provided, the default registry is used.

    Returns
    -------
        The path to the reference data.
    """
    return registry.fetch(_DATASETS[dataset_id])
