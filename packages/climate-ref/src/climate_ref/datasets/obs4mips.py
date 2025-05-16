from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from loguru import logger

from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.cmip6 import _parse_datetime
from climate_ref.datasets.utils import ParallelBuilder, get_version_from_filename
from climate_ref.models.dataset import Dataset, Obs4MIPsDataset

OBS4MIPS_ATTRS = (
    "activity_id",
    "frequency",
    "grid",
    "grid_label",
    "institution_id",
    "nominal_resolution",
    "realm",
    "product",
    "source_id",
    "source_type",
    "variable_id",
    "variant_label",
)

OBS4MIPS_DRS_ITEMS = (
    "activity_id",
    "institution_id",
    "source_id",
    "variable_id",
    "grid_label",
    "source_version_number",
)


def parse_obs4mips(file: Path) -> dict[str, Any | None]:
    """Parser for obs4mips"""
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(file, chunks={}, decode_times=time_coder) as ds:
        info = {key: ds.attrs.get(key) for key in OBS4MIPS_ATTRS}
        has_none_value = any(value is None for value in info.values())
        if has_none_value:
            missing_fields = [key for key in OBS4MIPS_ATTRS if info.get(key) is None]
            traceback_message = str(missing_fields) + " are missing from the file metadata"
            raise AttributeError(traceback_message)

        if info["activity_id"] != "obs4MIPs":
            traceback_message = f"{file.name} is not an obs4MIPs dataset"
            raise TypeError(traceback_message)

        variable_id = info["variable_id"]

        if variable_id:
            attrs = ds[variable_id].attrs
            for attr in ["long_name", "units"]:
                info[attr] = attrs.get(attr)

        # Set the default of # of vertical levels to 1
        vertical_levels = 1
        start_time, end_time = None, None
        try:
            vertical_levels = ds[ds.cf["vertical"].name].size
        except (KeyError, AttributeError, ValueError):
            ...
        try:
            start_time, end_time = str(ds.cf["T"][0].data), str(ds.cf["T"][-1].data)
        except (KeyError, AttributeError, ValueError):
            ...

        info["vertical_levels"] = vertical_levels
        info["start_time"] = start_time
        info["end_time"] = end_time
        if not (start_time and end_time):
            info["time_range"] = None
        else:
            info["time_range"] = f"{start_time}-{end_time}"
    info["path"] = str(file)
    info["source_version_number"] = get_version_from_filename(file) or "v0"
    return info


class Obs4MIPsDatasetAdapter(DatasetAdapter):
    """
    Adapter for obs4MIPs datasets
    """

    dataset_cls: type[Dataset] = Obs4MIPsDataset
    slug_column = "instance_id"

    dataset_specific_metadata = (
        *OBS4MIPS_ATTRS,
        "long_name",
        "units",
        "vertical_levels",
        "source_version_number",
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path")

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def pretty_subset(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Get a subset of the data_catalog to pretty print

        This is particularly useful for obs4MIPs datasets, which have a lot of metadata columns.

        Parameters
        ----------
        data_catalog
            Data catalog to subset

        Returns
        -------
        :
            Subset of the data catalog to pretty print

        """
        return data_catalog[list(OBS4MIPS_DRS_ITEMS)]

    def find_local_datasets(self, directories: str | Path | Sequence[str | Path]) -> pd.DataFrame:
        """
        Generate a data catalog from the specified file or directory

        Each dataset may contain multiple files, which are represented as rows in the data catalog.
        Each dataset has a unique identifier, which is in `slug_column`.

        Parameters
        ----------
        directories
            File or directories containing the datasets

        Returns
        -------
        :
            Data catalog containing the metadata for the dataset
        """
        builder = ParallelBuilder(
            paths=directories,
        )
        datasets = builder.get_datasets(parsing_func=parse_obs4mips)

        if datasets.empty:
            logger.error("No datasets found")
            raise ValueError("No obs4MIPs-compliant datasets found")

        # Convert the start_time and end_time columns to datetime objects
        # We don't know the calendar used in the dataset (TODO: Check what ecgtools does)
        datasets["start_time"] = _parse_datetime(datasets["start_time"])
        datasets["end_time"] = _parse_datetime(datasets["end_time"])

        datasets["instance_id"] = datasets.apply(
            lambda row: "obs4MIPs." + ".".join([row[item] for item in OBS4MIPS_DRS_ITEMS]), axis=1
        )
        return datasets
