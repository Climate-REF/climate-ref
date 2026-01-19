"""
CMIP7 dataset adapter for parsing and registering CMIP7 datasets.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from ecgtools import Builder
from loguru import logger

from climate_ref.config import Config
from climate_ref.datasets.base import DatasetAdapter, DatasetParsingFunction
from climate_ref.models.dataset import CMIP7Dataset
from climate_ref_core.cmip6_to_cmip7 import create_cmip7_instance_id


def _parse_datetime(dt_str: pd.Series[str]) -> pd.Series[datetime | Any]:
    """
    Parse datetime strings to datetime objects.
    """

    def _inner(date_string: str | None) -> datetime | None:
        if not date_string or pd.isnull(date_string):
            return None

        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(date_string, fmt)
            except ValueError:
                continue

        logger.error(f"Failed to parse date string: {date_string}")
        return None

    return pd.Series(
        [_inner(dt) for dt in dt_str],
        index=dt_str.index,
        dtype="object",
    )


def parse_cmip7_file(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Parse a CMIP7 netCDF file and extract metadata.

    Parameters
    ----------
    file
        Path to the netCDF file
    kwargs
        Additional keyword arguments (unused, for protocol compatibility)

    Returns
    -------
    dict[str, Any]
        Dictionary of metadata extracted from the file
    """
    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(file, decode_times=time_coder)
    except Exception as e:
        logger.warning(f"Failed to open {file}: {e}")
        return {}

    attrs = ds.attrs

    # Extract time range
    start_time = None
    end_time = None
    if "time" in ds.dims:
        try:
            # Use cf accessor if available, or fall back to direct access
            start_time = str(ds.cf["T"][0].data)
            end_time = str(ds.cf["T"][-1].data)
        except (KeyError, AttributeError, ValueError):
            time_values = ds["time"].values
            if len(time_values) > 0:
                start_time = str(time_values[0])
                end_time = str(time_values[-1])

    ds.close()

    # Build metadata dictionary per CMIP7 V1.0 spec
    variable_id = attrs.get("variable_id", "")
    branding_suffix = attrs.get("branding_suffix", "")
    branded_variable = attrs.get("branded_variable", "")
    if not branded_variable and variable_id and branding_suffix:
        branded_variable = f"{variable_id}_{branding_suffix}"

    result = {
        "path": file,
        "start_time": start_time,
        "end_time": end_time,
        # Core identification (mandatory)
        "activity_id": attrs.get("activity_id", "CMIP"),
        "institution_id": attrs.get("institution_id", ""),
        "source_id": attrs.get("source_id", ""),
        "experiment_id": attrs.get("experiment_id", ""),
        "variant_label": attrs.get("variant_label", ""),
        "variable_id": variable_id,
        "grid_label": attrs.get("grid_label", "gn"),
        "version": attrs.get("version", "v1"),
        # CMIP7-specific mandatory fields
        "mip_era": attrs.get("mip_era", "CMIP7"),
        "region": attrs.get("region", "glb"),  # lowercase per spec
        "frequency": attrs.get("frequency", "mon"),
        "branding_suffix": branding_suffix,
        "branded_variable": branded_variable,
        "temporal_label": attrs.get("temporal_label", ""),
        "vertical_label": attrs.get("vertical_label", ""),
        "horizontal_label": attrs.get("horizontal_label", ""),
        "area_label": attrs.get("area_label", ""),
        # DRS and spec info
        "drs_specs": attrs.get("drs_specs", "MIP-DRS7"),
        "data_specs_version": attrs.get("data_specs_version", "MIP-DS7.1.0.0"),
        "product": attrs.get("product", "model-output"),
        "license_id": attrs.get("license_id", "CC-BY-4.0"),
        # Variant indices (CMIP7 uses prefixed strings)
        "realization_index": attrs.get("realization_index", ""),
        "initialization_index": attrs.get("initialization_index", ""),
        "physics_index": attrs.get("physics_index", ""),
        "forcing_index": attrs.get("forcing_index", ""),
        # Optional metadata
        "realm": attrs.get("realm", ""),
        "nominal_resolution": attrs.get("nominal_resolution", ""),
        "tracking_id": attrs.get("tracking_id", ""),
        "standard_name": attrs.get("standard_name", ""),
        "long_name": attrs.get("long_name", ""),
        "units": attrs.get("units", ""),
        # Parent fields (conditionally required)
        "branch_time_in_child": attrs.get("branch_time_in_child"),
        "branch_time_in_parent": attrs.get("branch_time_in_parent"),
        "parent_activity_id": attrs.get("parent_activity_id", ""),
        "parent_experiment_id": attrs.get("parent_experiment_id", ""),
        "parent_mip_era": attrs.get("parent_mip_era", ""),
        "parent_source_id": attrs.get("parent_source_id", ""),
        "parent_time_units": attrs.get("parent_time_units", ""),
        "parent_variant_label": attrs.get("parent_variant_label", ""),
        "external_variables": attrs.get("external_variables", ""),
    }

    return result


class CMIP7DatasetAdapter(DatasetAdapter):
    """
    Adapter for CMIP7 datasets.

    CMIP7 datasets follow the MIP-DRS7 specification with additional
    branding suffix information and new metadata fields.
    """

    dataset_cls = CMIP7Dataset
    slug_column = "instance_id"

    dataset_specific_metadata = (
        # Core identification (mandatory per CMIP7 V1.0 spec)
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "variant_label",
        "variable_id",
        "grid_label",
        "version",
        # CMIP7-specific mandatory fields
        "mip_era",
        "region",
        "frequency",
        "branding_suffix",
        "branded_variable",
        "temporal_label",
        "vertical_label",
        "horizontal_label",
        "area_label",
        # DRS and spec info
        "drs_specs",
        "data_specs_version",
        "product",
        "license_id",
        # Variant indices (CMIP7 uses prefixed strings)
        "realization_index",
        "initialization_index",
        "physics_index",
        "forcing_index",
        # Optional metadata
        "realm",
        "nominal_resolution",
        "tracking_id",
        "standard_name",
        "long_name",
        "units",
        # Parent fields (conditionally required)
        "branch_time_in_child",
        "branch_time_in_parent",
        "parent_activity_id",
        "parent_experiment_id",
        "parent_mip_era",
        "parent_source_id",
        "parent_time_units",
        "parent_variant_label",
        "external_variables",
        # System fields
        "finalised",
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path")

    version_metadata = "version"

    # CMIP7 DRS components (excluding version) per MIP-DRS7 spec
    # Order matches directory structure: activity/institution/source/experiment/
    #   variant/region/frequency/variable/branding/grid
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

    def get_parsing_function(self) -> DatasetParsingFunction:
        """
        Get the parsing function for CMIP7 datasets.

        Returns
        -------
        DatasetParsingFunction
            The CMIP7 file parsing function
        """
        return parse_cmip7_file

    def find_local_datasets(self, file_or_directory: Path) -> pd.DataFrame:
        """
        Generate a data catalog from the specified file or directory.

        Parameters
        ----------
        file_or_directory
            File or directory containing CMIP7 datasets

        Returns
        -------
        pd.DataFrame
            Data catalog containing metadata for the datasets
        """
        parsing_function = self.get_parsing_function()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            builder = Builder(
                paths=[str(file_or_directory)],
                depth=15,  # CMIP7 DRS is deeper than CMIP6
                include_patterns=["*.nc"],
                joblib_parallel_kwargs={"n_jobs": self.n_jobs},
            )

            # Check if there are any assets before building
            builder.get_assets()  # type: ignore[attr-defined]
            if not builder.assets:  # type: ignore[attr-defined]
                return pd.DataFrame(columns=self.dataset_specific_metadata + self.file_specific_metadata)

            builder.build(parsing_func=parsing_function)

        datasets: pd.DataFrame = builder.df

        if datasets.empty:
            return pd.DataFrame(columns=self.dataset_specific_metadata + self.file_specific_metadata)

        # Convert time columns
        if "start_time" in datasets.columns:
            datasets["start_time"] = _parse_datetime(datasets["start_time"])
        if "end_time" in datasets.columns:
            datasets["end_time"] = _parse_datetime(datasets["end_time"])

        # Generate instance_id
        datasets["instance_id"] = datasets.apply(lambda row: create_cmip7_instance_id(row.to_dict()), axis=1)

        # Ensure finalised column exists
        if "finalised" not in datasets.columns:
            datasets["finalised"] = True

        # Add any missing metadata columns
        missing_columns = set(self.dataset_specific_metadata + self.file_specific_metadata) - set(
            datasets.columns
        )
        for column in missing_columns:
            datasets[column] = pd.NA

        return datasets
