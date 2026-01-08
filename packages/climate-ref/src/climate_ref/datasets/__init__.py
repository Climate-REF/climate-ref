"""
Dataset handling utilities
"""

from typing import Any

from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter
from climate_ref_core.datasets import SourceDatasetType


def get_dataset_adapter(source_type: str, **kwargs: Any) -> DatasetAdapter:
    """
    Get the appropriate adapter for the specified source type

    Parameters
    ----------
    source_type
        Type of source dataset

    Returns
    -------
    :
        DatasetAdapter instance
    """
    if source_type.lower() == SourceDatasetType.CMIP6.value:
        return CMIP6DatasetAdapter(**kwargs)
    elif source_type.lower() == SourceDatasetType.obs4MIPs.value.lower():
        return Obs4MIPsDatasetAdapter(**kwargs)
    elif source_type.lower() == SourceDatasetType.PMPClimatology.value.lower():
        return PMPClimatologyDatasetAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


__all__ = [
    "CMIP6DatasetAdapter",
    "DatasetAdapter",
    "Obs4MIPsDatasetAdapter",
    "PMPClimatologyDatasetAdapter",
    "get_dataset_adapter",
]
