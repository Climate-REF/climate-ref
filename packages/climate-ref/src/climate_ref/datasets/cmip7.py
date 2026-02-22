"""
CMIP7 Dataset Adapter

Adapter for parsing and registering CMIP7 datasets based on CMIP7 Global Attributes v1.0.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import netCDF4
import pandas as pd

from climate_ref.config import Config
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.catalog_builder import build_catalog
from climate_ref.datasets.netcdf_utils import read_time_bounds, read_variable_attrs
from climate_ref.datasets.utils import clean_branch_time, parse_datetime
from climate_ref.models.dataset import CMIP7Dataset


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
        with netCDF4.Dataset(file, "r") as ds:
            start_time, end_time = read_time_bounds(ds)

            variable_id = getattr(ds, "variable_id", "")
            var_attrs = read_variable_attrs(ds, variable_id, ["standard_name", "long_name", "units"])

            return {
                # Core DRS attributes
                "activity_id": getattr(ds, "activity_id", ""),
                "institution_id": getattr(ds, "institution_id", ""),
                "source_id": getattr(ds, "source_id", ""),
                "experiment_id": getattr(ds, "experiment_id", ""),
                "variant_label": getattr(ds, "variant_label", ""),
                "variable_id": variable_id,
                "grid_label": getattr(ds, "grid_label", ""),
                "frequency": getattr(ds, "frequency", ""),
                "region": getattr(ds, "region", "glb"),
                "branding_suffix": getattr(ds, "branding_suffix", ""),
                "branded_variable": getattr(ds, "branded_variable", ""),
                "out_name": getattr(ds, "out_name", ""),
                "version": getattr(ds, "version", ""),
                # Additional mandatory attributes
                "mip_era": getattr(ds, "mip_era", "CMIP7"),
                "realm": getattr(ds, "realm", None),
                "nominal_resolution": getattr(ds, "nominal_resolution", None),
                # Parent info (nullable)
                "branch_time_in_child": getattr(ds, "branch_time_in_child", None),
                "branch_time_in_parent": getattr(ds, "branch_time_in_parent", None),
                "parent_activity_id": getattr(ds, "parent_activity_id", None),
                "parent_experiment_id": getattr(ds, "parent_experiment_id", None),
                "parent_mip_era": getattr(ds, "parent_mip_era", None),
                "parent_source_id": getattr(ds, "parent_source_id", None),
                "parent_time_units": getattr(ds, "parent_time_units", None),
                "parent_variant_label": getattr(ds, "parent_variant_label", None),
                # Additional mandatory attributes
                "license_id": getattr(ds, "license_id", None),
                # Conditionally required attributes
                "external_variables": getattr(ds, "external_variables", None),
                # Variable metadata
                "standard_name": var_attrs["standard_name"],
                "long_name": var_attrs["long_name"],
                "units": var_attrs["units"],
                # File-level metadata
                "tracking_id": getattr(ds, "tracking_id", None),
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
        # Additional mandatory attributes
        "license_id",
        # Conditionally required attributes
        "external_variables",
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

    # CMIP7 DRS directory structure (MIP-DRS7 spec):
    #   <drs_specs>/<mip_era>/<activity_id>/<institution_id>/.../<grid_label>/<version>
    # The leading drs_specs and mip_era are fixed values ("MIP-DRS7" and "CMIP7")
    # and are omitted here. They are added as the "CMIP7." prefix when building instance_id.
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
        datasets = build_catalog(
            paths=[str(file_or_directory)],
            parsing_func=parse_cmip7_file,
            include_patterns=["*.nc"],
            depth=10,
            n_jobs=self.n_jobs,
        )

        # Convert the start_time and end_time columns to datetime objects
        if "start_time" in datasets.columns:
            datasets["start_time"] = parse_datetime(datasets["start_time"])
        if "end_time" in datasets.columns:
            datasets["end_time"] = parse_datetime(datasets["end_time"])

        # Clean branch times
        if "branch_time_in_child" in datasets.columns:
            datasets["branch_time_in_child"] = clean_branch_time(datasets["branch_time_in_child"])
        if "branch_time_in_parent" in datasets.columns:
            datasets["branch_time_in_parent"] = clean_branch_time(datasets["branch_time_in_parent"])

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

        # Add branded_variable for the raw catalog (before DB ingestion)
        datasets["branded_variable"] = datasets["variable_id"] + "_" + datasets["branding_suffix"]

        return datasets
