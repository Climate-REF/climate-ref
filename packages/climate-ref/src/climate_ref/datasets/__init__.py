"""
Dataset handling utilities
"""

from pathlib import Path
from typing import Any

import pandas as pd
from attrs import define
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database, ModelState
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.cmip7 import CMIP7DatasetAdapter
from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter
from climate_ref_core.datasets import SourceDatasetType


@define
class IngestionStats:
    """
    Statistics from ingesting datasets into the database
    """

    datasets_created: int = 0
    datasets_updated: int = 0
    datasets_unchanged: int = 0
    files_added: int = 0
    files_updated: int = 0
    files_removed: int = 0
    files_unchanged: int = 0

    def log_summary(self, prefix: str = "") -> None:
        """Log a summary of the ingestion statistics."""
        prefix_str = f"{prefix} " if prefix else ""
        logger.info(
            f"{prefix_str}Datasets: {self.datasets_created}/{self.datasets_updated}/{self.datasets_unchanged}"
            " (created/updated/unchanged), "
            f"Files: {self.files_added}/{self.files_updated}/{self.files_removed}/{self.files_unchanged}"
            " (created/updated/removed/unchanged)"
        )


def ingest_datasets(  # noqa: PLR0913
    adapter: DatasetAdapter,
    directory: Path | None,
    config: Config,
    db: Database,
    *,
    data_catalog: pd.DataFrame | None = None,
    skip_invalid: bool = True,
) -> IngestionStats:
    """
    Ingest datasets from a directory into the database.

    This is the common ingestion logic shared between the CLI ingest command
    and provider setup.

    Parameters
    ----------
    adapter
        The dataset adapter to use for parsing and registering datasets
    directory
        Directory containing the datasets to ingest. Can be None if data_catalog is provided.
    config
        Application configuration
    db
        Database instance
    data_catalog
        Optional pre-validated data catalog. If provided, directory is ignored and
        the catalog is used directly. This avoids redundant find/validate operations.
    skip_invalid
        If True, skip datasets that fail validation (default True)

    Returns
    -------
    :
        Statistics about the ingestion (created/updated/unchanged counts)

    Raises
    ------
    ValueError
        If no valid datasets are found in the directory
    """
    if data_catalog is None:
        if directory is None:
            raise ValueError("Either directory or data_catalog must be provided")

        if not directory.exists():
            raise ValueError(f"Directory {directory} does not exist")

        # Check for .nc files
        if not list(directory.rglob("*.nc")):
            raise ValueError(f"No .nc files found in {directory}")

        data_catalog = adapter.find_local_datasets(directory)
        data_catalog = adapter.validate_data_catalog(data_catalog, skip_invalid=skip_invalid)

        if data_catalog.empty:
            raise ValueError(f"No valid datasets found in {directory}")

        logger.info(
            f"Found {len(data_catalog)} files for {len(data_catalog[adapter.slug_column].unique())} datasets"
        )

    stats = IngestionStats()

    for instance_id, data_catalog_dataset in data_catalog.groupby(adapter.slug_column):
        logger.debug(f"Processing dataset {instance_id}")
        with db.session.begin():
            results = adapter.register_dataset(config, db, data_catalog_dataset)

            if results.dataset_state == ModelState.CREATED:
                stats.datasets_created += 1
            elif results.dataset_state == ModelState.UPDATED:
                stats.datasets_updated += 1
            else:
                stats.datasets_unchanged += 1
            stats.files_added += len(results.files_added)
            stats.files_updated += len(results.files_updated)
            stats.files_removed += len(results.files_removed)
            stats.files_unchanged += len(results.files_unchanged)

    return stats


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
    elif source_type.lower() == SourceDatasetType.CMIP7.value:
        return CMIP7DatasetAdapter(**kwargs)
    elif source_type.lower() == SourceDatasetType.obs4MIPs.value.lower():
        return Obs4MIPsDatasetAdapter(**kwargs)
    elif source_type.lower() == SourceDatasetType.PMPClimatology.value.lower():
        return PMPClimatologyDatasetAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


__all__ = [
    "CMIP6DatasetAdapter",
    "CMIP7DatasetAdapter",
    "DatasetAdapter",
    "IngestionStats",
    "Obs4MIPsDatasetAdapter",
    "PMPClimatologyDatasetAdapter",
    "get_dataset_adapter",
    "ingest_datasets",
]
