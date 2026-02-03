"""
CMIP7 Dataset Adapter

Adapter for parsing and registering CMIP7 datasets based on CMIP7 Global Attributes v1.0.
"""

from __future__ import annotations

import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from ecgtools import Builder
from loguru import logger

from climate_ref.config import Config
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.models.dataset import CMIP7Dataset


def _parse_datetime(dt_str: pd.Series[str]) -> pd.Series[datetime | Any]:
    """
    Parse datetime strings from CMIP7 files.

    Pandas tries to coerce everything to their own datetime format, which is not what we want here.
    """

    def _inner(date_string: str | None) -> datetime | None:
        if not date_string or pd.isnull(date_string):
            return None

        # Try to parse the date string with and without milliseconds
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        # If all parsing attempts fail, log an error and return None
        logger.error(f"Failed to parse date string: {date_string}")
        return None

    return pd.Series(
        [_inner(dt) for dt in dt_str],
        index=dt_str.index,
        dtype="object",
    )


def _clean_branch_time(branch_time: pd.Series[str]) -> pd.Series[float]:
    """Clean branch time values, handling missing values and suffixes."""
    # Handle missing values (these result in nan values)
    return pd.to_numeric(branch_time.astype(str).str.replace("D", ""), errors="coerce")


def parse_cmip7_file(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Parse metadata from a CMIP7 netCDF file.

    Parameters
    ----------
    file
        Path to the CMIP7 netCDF file

    Returns
    -------
    :
        Dictionary of metadata extracted from the file
    """
    try:
        with xr.open_dataset(file, use_cftime=True) as ds:
            attrs = ds.attrs

            # Extract time bounds if available
            start_time = None
            end_time = None
            if "time" in ds:
                time = ds["time"]
                if len(time) > 0:
                    start_time = str(time.values[0])
                    end_time = str(time.values[-1])

            # Get variable metadata from the data variable
            variable_id = attrs.get("variable_id", "")
            standard_name = None
            long_name = None
            units = None
            if variable_id and variable_id in ds:
                var = ds[variable_id]
                standard_name = var.attrs.get("standard_name")
                long_name = var.attrs.get("long_name")
                units = var.attrs.get("units")

            return {
                # Core DRS attributes
                "activity_id": attrs.get("activity_id", ""),
                "institution_id": attrs.get("institution_id", ""),
                "source_id": attrs.get("source_id", ""),
                "experiment_id": attrs.get("experiment_id", ""),
                "variant_label": attrs.get("variant_label", ""),
                "variable_id": variable_id,
                "grid_label": attrs.get("grid_label", ""),
                "frequency": attrs.get("frequency", ""),
                "region": attrs.get("region", "glb"),
                "branding_suffix": attrs.get("branding_suffix", ""),
                "version": attrs.get("version", ""),
                # Additional mandatory attributes
                "mip_era": attrs.get("mip_era", "CMIP7"),
                "realm": attrs.get("realm"),
                "nominal_resolution": attrs.get("nominal_resolution"),
                # Parent info (nullable)
                "branch_time_in_child": attrs.get("branch_time_in_child"),
                "branch_time_in_parent": attrs.get("branch_time_in_parent"),
                "parent_activity_id": attrs.get("parent_activity_id"),
                "parent_experiment_id": attrs.get("parent_experiment_id"),
                "parent_mip_era": attrs.get("parent_mip_era"),
                "parent_source_id": attrs.get("parent_source_id"),
                "parent_time_units": attrs.get("parent_time_units"),
                "parent_variant_label": attrs.get("parent_variant_label"),
                # Variable metadata
                "standard_name": standard_name,
                "long_name": long_name,
                "units": units,
                # File-level metadata
                "tracking_id": attrs.get("tracking_id"),
                # Time information
                "start_time": start_time,
                "end_time": end_time,
                "time_range": f"{start_time}-{end_time}" if start_time and end_time else None,
                # Path
                "path": file,
            }
    except Exception:
        return {
            "INVALID_ASSET": file,
            "TRACEBACK": traceback.format_exc(),
        }


class CMIP7DatasetAdapter(DatasetAdapter):
    """
    Adapter for CMIP7 datasets

    Based on CMIP7 Global Attributes v1.0 (DOI: 10.5281/zenodo.17250297).
    """

    dataset_cls = CMIP7Dataset
    slug_column = "instance_id"

    dataset_specific_metadata = (
        # Core DRS attributes
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "variant_label",
        "variable_id",
        "grid_label",
        "frequency",
        "region",
        "branding_suffix",
        "version",
        # Additional mandatory attributes
        "mip_era",
        "realm",
        "nominal_resolution",
        # Parent info
        "branch_time_in_child",
        "branch_time_in_parent",
        "parent_activity_id",
        "parent_experiment_id",
        "parent_mip_era",
        "parent_source_id",
        "parent_time_units",
        "parent_variant_label",
        # Variable metadata
        "standard_name",
        "long_name",
        "units",
        # Unique identifier
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path", "tracking_id")

    version_metadata = "version"

    # CMIP7 DRS: activity_id/institution_id/source_id/experiment_id/variant_label/
    #            region/frequency/variable_id/branding_suffix/grid_label
    dataset_id_metadata = (
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "variant_label",
        "region",
        "frequency",
        "variable_id",
        "branding_suffix",
        "grid_label",
    )

    def __init__(self, n_jobs: int = 1, config: Config | None = None):
        self.n_jobs = n_jobs
        self.config = config or Config.default()

    def find_local_datasets(self, file_or_directory: Path) -> pd.DataFrame:
        """
        Generate a data catalog from the specified file or directory.

        Each dataset may contain multiple files, which are represented as rows in the data catalog.
        Each dataset has a unique identifier, which is in `slug_column`.

        Parameters
        ----------
        file_or_directory
            File or directory containing the datasets

        Returns
        -------
        :
            Data catalog containing the metadata for the dataset
        """
        with warnings.catch_warnings():
            # Ignore the DeprecationWarning from xarray
            warnings.simplefilter("ignore", DeprecationWarning)

            builder = Builder(
                paths=[str(file_or_directory)],
                depth=10,
                include_patterns=["*.nc"],
                joblib_parallel_kwargs={"n_jobs": self.n_jobs},
            ).build(parsing_func=parse_cmip7_file)

        datasets: pd.DataFrame = builder.df

        # Convert the start_time and end_time columns to datetime objects
        if "start_time" in datasets.columns:
            datasets["start_time"] = _parse_datetime(datasets["start_time"])
        if "end_time" in datasets.columns:
            datasets["end_time"] = _parse_datetime(datasets["end_time"])

        # Clean branch times
        if "branch_time_in_child" in datasets.columns:
            datasets["branch_time_in_child"] = _clean_branch_time(datasets["branch_time_in_child"])
        if "branch_time_in_parent" in datasets.columns:
            datasets["branch_time_in_parent"] = _clean_branch_time(datasets["branch_time_in_parent"])

        # Build instance_id following CMIP7 DRS format
        # CMIP7.<activity_id>.<institution_id>.<source_id>.<experiment_id>.<variant_label>.
        # <region>.<frequency>.<variable_id>.<branding_suffix>.<grid_label>.<version>
        drs_items = [
            *self.dataset_id_metadata,
            self.version_metadata,
        ]
        datasets["instance_id"] = datasets.apply(
            lambda row: "CMIP7." + ".".join([str(row[item]) for item in drs_items]), axis=1
        )

        # Add in any missing metadata columns
        missing_columns = set(self.dataset_specific_metadata + self.file_specific_metadata) - set(
            datasets.columns
        )
        if missing_columns:
            for column in missing_columns:
                datasets[column] = pd.NA

        return datasets
