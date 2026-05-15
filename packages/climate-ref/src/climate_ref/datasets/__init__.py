"""
Dataset handling utilities
"""

from pathlib import Path
from typing import Any

import pandas as pd
from attrs import define
from loguru import logger

from climate_ref.database import Database, ModelState
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.cmip7 import CMIP7DatasetAdapter
from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from climate_ref.datasets.pmp_climatology import PMPClimatologyDatasetAdapter
from climate_ref_core.datasets import SourceDatasetType

# Pre-computed slug column lookup by source type.
# TODO: This could be replaced with a constant at the moment
SLUG_COLUMN_BY_SOURCE_TYPE: dict[SourceDatasetType, str] = {
    SourceDatasetType.CMIP6: "instance_id",
    SourceDatasetType.CMIP7: "instance_id",
    SourceDatasetType.obs4MIPs: "instance_id",
    SourceDatasetType.PMPClimatology: "instance_id",
}


def get_slug_column(source_type: SourceDatasetType | str) -> str:
    """
    Get the slug column name for a source dataset type.

    Parameters
    ----------
    source_type
        Source dataset type (enum or string value)

    Returns
    -------
    :
        The slug column name for the given source type
    """
    if isinstance(source_type, str):
        source_type = SourceDatasetType(source_type)
    return SLUG_COLUMN_BY_SOURCE_TYPE[source_type]


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

    def __iadd__(self, other: "IngestionStats") -> "IngestionStats":
        """Accumulate counts in place from another :class:`IngestionStats`."""
        self.datasets_created += other.datasets_created
        self.datasets_updated += other.datasets_updated
        self.datasets_unchanged += other.datasets_unchanged
        self.files_added += other.files_added
        self.files_updated += other.files_updated
        self.files_removed += other.files_removed
        self.files_unchanged += other.files_unchanged
        return self


def _ingest_catalog(
    adapter: DatasetAdapter,
    db: Database,
    data_catalog: pd.DataFrame,
) -> IngestionStats:
    """
    Register every dataset in ``data_catalog``, committing per-dataset.

    The ORM session identity map is expired after each commit so memory
    use stays bounded by the largest single dataset, not by the size of
    ``data_catalog``.
    """
    stats = IngestionStats()

    for instance_id, data_catalog_dataset in data_catalog.groupby(adapter.slug_column):
        logger.debug(f"Processing dataset {instance_id}")
        with db.session.begin():
            results = adapter.register_dataset(db, data_catalog_dataset)

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

        # Release ORM objects from the session identity map after each commit.
        # Without this, all Dataset and DatasetFile objects accumulate in memory
        # across the entire ingestion loop.
        db.session.expire_all()

    return stats


def ingest_datasets(  # noqa: PLR0913
    adapter: DatasetAdapter,
    directory: Path | None,
    db: Database,
    *,
    data_catalog: pd.DataFrame | None = None,
    skip_invalid: bool = True,
    chunk_size: int | None = None,
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
    db
        Database instance
    data_catalog
        Optional pre-validated data catalog.

        If provided, directory is ignored and the catalog is used directly.
        This avoids redundant find/validate operations.
        When supplied, ``chunk_size`` is ignored because the catalog is already fully materialised.
    skip_invalid
        If True, skip datasets that fail validation (default True)
    chunk_size
        When provided and ``data_catalog`` is None,
        stream the directory in batches of ``chunk_size`` files so peak memory is bounded regardless
        of how many files live under ``directory``.
        Requires the adapter to implement ``iter_local_datasets``.

    Returns
    -------
    :
        Statistics about the ingestion (created/updated/unchanged counts)

    Raises
    ------
    ValueError
        If no valid datasets are found in the directory
    """
    if data_catalog is not None:
        return _ingest_catalog(adapter, db, data_catalog)

    if directory is None:
        raise ValueError("Either directory or data_catalog must be provided")

    if not directory.exists():
        raise ValueError(f"Directory {directory} does not exist")

    # Check for .nc files
    if not any(directory.rglob("*.nc")):
        raise ValueError(f"No .nc files found in {directory}")

    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        iter_fn = getattr(adapter, "iter_local_datasets", None)
        if iter_fn is None:
            raise ValueError(
                f"Adapter {type(adapter).__name__} does not support streaming ingest "
                "(missing iter_local_datasets); omit chunk_size to use whole-catalog mode."
            )

        stats = IngestionStats()
        total_files = 0
        total_datasets = 0
        emitted = False
        for raw_chunk in iter_fn(directory, chunk_size=chunk_size):
            validated_chunk = adapter.validate_data_catalog(raw_chunk, skip_invalid=skip_invalid)
            if validated_chunk.empty:
                continue
            emitted = True
            total_files += len(validated_chunk)
            total_datasets += validated_chunk[adapter.slug_column].nunique()
            stats += _ingest_catalog(adapter, db, validated_chunk)
            # Drop chunk references so the per-chunk pandas memory can be
            # reclaimed before the next chunk is parsed.
            del raw_chunk, validated_chunk

        if not emitted:
            raise ValueError(f"No valid datasets found in {directory}")

        logger.info(f"Ingested {total_files} files across approximately {total_datasets} datasets (streamed)")
        return stats

    data_catalog = adapter.find_local_datasets(directory)
    data_catalog = adapter.validate_data_catalog(data_catalog, skip_invalid=skip_invalid)

    if data_catalog.empty:
        raise ValueError(f"No valid datasets found in {directory}")

    logger.info(
        f"Found {len(data_catalog)} files for {len(data_catalog[adapter.slug_column].unique())} datasets"
    )

    return _ingest_catalog(adapter, db, data_catalog)


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
    "SLUG_COLUMN_BY_SOURCE_TYPE",
    "CMIP6DatasetAdapter",
    "CMIP7DatasetAdapter",
    "DatasetAdapter",
    "IngestionStats",
    "Obs4MIPsDatasetAdapter",
    "PMPClimatologyDatasetAdapter",
    "get_dataset_adapter",
    "get_slug_column",
    "ingest_datasets",
]
