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
from climate_ref.datasets.utils import build_instance_id, parse_cftime_dates
from climate_ref.models.dataset import Dataset, Obs4MIPsDataset, Obs4REFDataset


def parse_obs4mips(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Parser for obs4MIPs and obs4REF files

    obs4REF files follow the obs4MIPs metadata conventions, so the same parser reads both.
    The adapter that called the parser decides which collection the file belongs to.

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
            activity_id = getattr(ds, "activity_id", "")
            if activity_id not in ("obs4MIPs", "obs4REF"):
                traceback_message = f"{file} is not an obs4MIPs or obs4REF dataset"
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

    activity_id = "obs4MIPs"
    """
    The collection this adapter ingests into, whatever the file itself claims.

    The obs4REF collection republishes obs4MIPs files unchanged,
    so the file attribute cannot tell the two collections apart.
    """

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
            n_jobs=self.n_jobs,
        )
        if datasets.empty:
            logger.error("No datasets found")
            raise ValueError("No obs4MIPs-compliant datasets found")

        self._warn_if_misfiled(datasets)
        datasets["activity_id"] = self.activity_id

        # Convert the start_time and end_time columns to cftime objects
        datasets["start_time"] = parse_cftime_dates(datasets["start_time"])
        datasets["end_time"] = parse_cftime_dates(datasets["end_time"])

        drs_items = [
            *self.dataset_id_metadata,
            self.version_metadata,
        ]

        def _transform(item: str, value: Any) -> str:
            return str(value).replace(" ", "") if item == "nominal_resolution" else str(value)

        datasets = build_instance_id(datasets, drs_items, prefix=self.activity_id, transform=_transform)
        datasets["finalised"] = True
        return datasets

    def _warn_if_misfiled(self, datasets: pd.DataFrame) -> None:
        """
        Warn when the files look like they belong to the other collection.

        The registry republishes obs4MIPs files unchanged, so an ``obs4REF`` directory
        or activity id is only a hint. The files are ingested either way.
        """
        other = "obs4REF" if self.activity_id == "obs4MIPs" else "obs4MIPs"
        misfiled = datasets["path"].astype(str).str.contains(f"/{other}/", regex=False)
        if other == "obs4REF":
            misfiled |= datasets["activity_id"] == other
        count = int(misfiled.sum())
        if count:
            logger.warning(
                f"{count} of {len(datasets)} files look like {other} data but are being ingested as "
                f"{self.activity_id}. Use `--source-type {other.lower()}` if that is not intended."
            )


class Obs4REFDatasetAdapter(Obs4MIPsDatasetAdapter):
    """
    Adapter for obs4REF datasets

    obs4REF is the REF-curated collection of observational data.
    It shares the obs4MIPs metadata conventions and parser,
    but is ingested as its own dataset type so it is never mistaken for published obs4MIPs data.
    """

    dataset_cls: type[Dataset] = Obs4REFDataset
    activity_id = "obs4REF"
