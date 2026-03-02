from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import netCDF4
import pandas as pd
from loguru import logger

from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.catalog_builder import build_catalog
from climate_ref.datasets.netcdf_utils import (
    read_global_attrs,
    read_time_bounds,
    read_variable_attrs,
    read_vertical_levels,
)
from climate_ref.datasets.utils import parse_cftime_dates
from climate_ref.models.dataset import Dataset, Obs4MIPsDataset


def parse_obs4mips(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Parser for obs4mips

    Parameters
    ----------
    file
        File to parse
    kwargs
        Additional keyword arguments (not used, but required for protocol compatibility)
    """
    keys = sorted(
        list(
            {
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
                "source_version_number",
            }
        )
    )

    try:
        with netCDF4.Dataset(file, "r") as ds:
            if getattr(ds, "activity_id", "") != "obs4MIPs":
                traceback_message = f"{file} is not an obs4MIPs dataset"
                raise TypeError(traceback_message)

            global_attrs = read_global_attrs(ds, keys)
            missing_fields = [key for key in keys if global_attrs.get(key) is None]

            if missing_fields:
                traceback_message = str(missing_fields) + " are missing from the file metadata"
                raise AttributeError(traceback_message)
            info = {**global_attrs}

            variable_id = global_attrs["variable_id"]

            if variable_id:
                var_attrs = read_variable_attrs(ds, variable_id, ["long_name", "units"])
                info.update(var_attrs)

            vertical_levels = read_vertical_levels(ds)
            start_time, end_time = read_time_bounds(ds)

            info["vertical_levels"] = vertical_levels
            info["start_time"] = start_time
            info["end_time"] = end_time
            if not (start_time and end_time):
                info["time_range"] = None
            else:
                info["time_range"] = f"{start_time}-{end_time}"
        info["path"] = str(file)
        # Parsing the version like for CMIP6 fails because some obs4REF paths
        # do not include "v" in the version directory name.
        # TODO: fix obs4REF paths
        info["version"] = Path(file).parent.name
        if not info["version"].startswith("v"):
            info["version"] = "v{version}".format(**info)
        return info

    except (TypeError, AttributeError) as err:
        if (len(err.args)) == 1:
            logger.warning(str(err.args[0]))
        else:
            logger.warning(str(err.args))
        return {"INVALID_ASSET": file, "TRACEBACK": str(err)}
    except Exception:
        logger.warning(traceback.format_exc())
        return {"INVALID_ASSET": file, "TRACEBACK": traceback.format_exc()}


class Obs4MIPsDatasetAdapter(DatasetAdapter):
    """
    Adapter for obs4MIPs datasets
    """

    dataset_cls: type[Dataset] = Obs4MIPsDataset
    slug_column = "instance_id"

    dataset_specific_metadata = (
        "activity_id",
        "finalised",
        "frequency",
        "grid",
        "grid_label",
        "institution_id",
        "nominal_resolution",
        "product",
        "realm",
        "source_id",
        "source_type",
        "variable_id",
        "variant_label",
        "long_name",
        "units",
        "version",
        "vertical_levels",
        "source_version_number",
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path")
    version_metadata = "version"
    # See ODS2.5 at https://doi.org/10.5281/zenodo.11500474 under "Directory structure template"
    dataset_id_metadata = (
        "activity_id",
        "institution_id",
        "source_id",
        "frequency",
        "variable_id",
        "nominal_resolution",
        "grid_label",
    )

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def find_local_datasets(self, file_or_directory: Path) -> pd.DataFrame:
        """
        Generate a data catalog from the specified file or directory

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
            parsing_func=parse_obs4mips,
            include_patterns=["*.nc"],
            depth=10,
            n_jobs=self.n_jobs,
        )
        if datasets.empty:
            logger.error("No datasets found")
            raise ValueError("No obs4MIPs-compliant datasets found")

        # Convert the start_time and end_time columns to cftime objects
        datasets["start_time"] = parse_cftime_dates(datasets["start_time"])
        datasets["end_time"] = parse_cftime_dates(datasets["end_time"])

        drs_items = [
            *self.dataset_id_metadata,
            self.version_metadata,
        ]
        datasets["instance_id"] = datasets.apply(
            lambda row: (
                "obs4MIPs."
                + ".".join(
                    [
                        row[item].replace(" ", "") if item == "nominal_resolution" else row[item]
                        for item in drs_items
                    ]
                )
            ),
            axis=1,
        )
        datasets["finalised"] = True
        return datasets
