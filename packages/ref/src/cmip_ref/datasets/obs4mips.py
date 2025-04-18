from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
from ecgtools import Builder
from loguru import logger

from cmip_ref.datasets.base import DatasetAdapter
from cmip_ref.datasets.cmip6 import _parse_datetime
from cmip_ref.models.dataset import Dataset, DatasetFile, Obs4MIPsDataset, Obs4MIPsFile


def extract_attr_with_regex(
    input_str: str, regex: str, strip_chars: str | None, ignore_case: bool
) -> list[Any] | None:
    """
    Extract version information from attribute with regular expressions.
    """
    if ignore_case:
        pattern = re.compile(regex, re.IGNORECASE)
    else:
        pattern = re.compile(regex)
    match = re.findall(pattern, input_str)
    if match:
        matchstr = max(match, key=len)
        match = matchstr.strip(strip_chars) if strip_chars else matchstr.strip()
        return match
    else:
        return None


def parse_citation_metadata(info: dict[str, Any | None], ds: xr.Dataset) -> dict[str, Any | None]:
    """Parser for doi and tracking_id"""
    try:
        info["tracking_id"] = ds.attrs["tracking_id"]

    except (KeyError, AttributeError, ValueError):
        ...

    try:
        info["doi"] = ds.attrs["doi"]
    except (KeyError, AttributeError, ValueError):
        info["doi"] = None
    try:
        info["references"] = ds.attrs["references"]
    except (KeyError, AttributeError, ValueError):
        info["references"] = None
    if info["doi"] is None and info["references"] is not None:
        doi_pattern = r"10.\d{4,9}/[-._;()/:A-Z0-9]+"
        dois = re.findall(doi_pattern, info["references"], re.IGNORECASE)
        if dois:
            info["doi"] = dois[0]
    return info


def parse_uncertainty(info: dict[str, Any | None], ds: xr.Dataset) -> dict[str, Any | None]:
    """Parser for uncertainty information"""
    if "has_auxdata" in ds.attrs:
        info["has_auxdata"] = ds.attrs["has_auxdata"]
    else:
        info["has_auxdata"] = "False"

    if info["has_auxdata"] is True:
        try:
            info["aux_variable_id"] = ds.attrs["aux_variable_id"]
        except (KeyError, AttributeError, ValueError):
            info["has_auxdata"] = "False"
            info["aux_variable_id"] = None
    else:
        info["aux_variable_id"] = None
    return info


def parse_obs4mips(file: str) -> dict[str, Any | None]:
    """Parser for obs4mips"""
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
            }
        )
    )

    try:
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        with xr.open_dataset(file, chunks={}, decode_times=time_coder) as ds:
            has_none_value = any(ds.attrs.get(key) is None for key in keys)
            if has_none_value:
                missing_fields = [key for key in keys if ds.attrs.get(key) is None]
                traceback_message = str(missing_fields) + " are missing from the file metadata"
                raise AttributeError(traceback_message)
            info = {key: ds.attrs.get(key) for key in keys}

            if info["activity_id"] != "obs4MIPs":
                traceback_message = f"{file} is not an obs4MIPs dataset"
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
            info = parse_citation_metadata(info, ds)
            info = parse_uncertainty(info, ds)
            info["vertical_levels"] = vertical_levels
            info["start_time"] = start_time
            info["end_time"] = end_time
            if not (start_time and end_time):
                info["time_range"] = None
            else:
                info["time_range"] = f"{start_time}-{end_time}"
        info["path"] = str(file)
        info["source_version_number"] = (
            extract_attr_with_regex(
                str(file), regex=r"v\d{4}\d{2}\d{2}|v\d{1}", strip_chars=None, ignore_case=True
            )
            or "v0"
        )
        return info

    except (TypeError, AttributeError) as err:
        if (len(err.args)) == 1:
            logger.warning(str(file + " " + err.args[0]))
        else:
            logger.warning(str(err.args))
        return {"INVALID_ASSET": file, "TRACEBACK": traceback_message}
    except Exception:
        logger.warning(traceback.format_exc())
        return {"INVALID_ASSET": file, "TRACEBACK": traceback.format_exc()}


class Obs4MIPsDatasetAdapter(DatasetAdapter):
    """
    Adapter for obs4MIPs datasets
    """

    dataset_cls: type[Dataset] = Obs4MIPsDataset
    file_cls: type[DatasetFile] = Obs4MIPsFile
    slug_column = "instance_id"

    dataset_specific_metadata = (
        "activity_id",
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
        "vertical_levels",
        "source_version_number",
        "doi",
        "references",
        "tracking_id",
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
        return data_catalog[
            [
                "activity_id",
                "institution_id",
                "source_id",
                "variable_id",
                "grid_label",
                "source_version_number",
                "doi",
                "tracking_id",
                "has_auxdata",
                "aux_variable_id",
            ]
        ]

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
        builder = Builder(
            paths=[str(file_or_directory)],
            depth=10,
            include_patterns=["*.nc"],
            joblib_parallel_kwargs={"n_jobs": self.n_jobs},
        ).build(parsing_func=parse_obs4mips)  # type: ignore[arg-type]

        datasets = builder.df
        if datasets.empty:
            logger.error("No datasets found")
            raise ValueError("No obs4MIPs-compliant datasets found")

        # Convert the start_time and end_time columns to datetime objects
        # We don't know the calendar used in the dataset (TODO: Check what ecgtools does)
        datasets["start_time"] = _parse_datetime(datasets["start_time"])
        datasets["end_time"] = _parse_datetime(datasets["end_time"])

        drs_items = [
            "activity_id",
            "institution_id",
            "source_id",
            "variable_id",
            "grid_label",
            "source_version_number",
        ]
        datasets["instance_id"] = datasets.apply(
            lambda row: "obs4MIPs." + ".".join([row[item] for item in drs_items]), axis=1
        )
        return datasets
