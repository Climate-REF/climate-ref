from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from ecgtools import Builder
from loguru import logger

from climate_ref.config import Config
from climate_ref.datasets.base import DatasetAdapter, DatasetParsingFunction
from climate_ref.datasets.cmip6_parsers import parse_cmip6_complete, parse_cmip6_drs
from climate_ref.models.dataset import CMIP6Dataset


def _parse_datetime(dt_str: pd.Series[str]) -> pd.Series[datetime | Any]:
    """
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


def _apply_fixes(data_catalog: pd.DataFrame) -> pd.DataFrame:
    def _fix_parent_variant_label(group: pd.DataFrame) -> pd.DataFrame:
        if group["parent_variant_label"].nunique() == 1:
            return group
        group["parent_variant_label"] = group["parent_variant_label"].iloc[0]

        return group

    if "parent_variant_label" in data_catalog:
        data_catalog = (
            data_catalog.groupby("instance_id")
            .apply(_fix_parent_variant_label, include_groups=False)  # type: ignore[call-overload]
            .reset_index(level="instance_id")
        )

    if "branch_time_in_child" in data_catalog:
        data_catalog["branch_time_in_child"] = _clean_branch_time(data_catalog["branch_time_in_child"])
    if "branch_time_in_parent" in data_catalog:
        data_catalog["branch_time_in_parent"] = _clean_branch_time(data_catalog["branch_time_in_parent"])

    return data_catalog


def _clean_branch_time(branch_time: pd.Series[str]) -> pd.Series[float]:
    # EC-Earth3 uses "D" as a suffix for the branch_time_in_child and branch_time_in_parent columns
    # Handle missing values (these result in nan values)
    return pd.to_numeric(branch_time.astype(str).str.replace("D", ""), errors="coerce")


class CMIP6DatasetAdapter(DatasetAdapter):
    """
    Adapter for CMIP6 datasets
    """

    dataset_cls = CMIP6Dataset
    slug_column = "instance_id"

    dataset_specific_metadata = (
        "activity_id",
        "branch_method",
        "branch_time_in_child",
        "branch_time_in_parent",
        "experiment",
        "experiment_id",
        "frequency",
        "grid",
        "grid_label",
        "institution_id",
        "nominal_resolution",
        "parent_activity_id",
        "parent_experiment_id",
        "parent_source_id",
        "parent_time_units",
        "parent_variant_label",
        "product",
        "realm",
        "source_id",
        "source_type",
        "sub_experiment",
        "sub_experiment_id",
        "table_id",
        "variable_id",
        "variant_label",
        "member_id",
        "vertical_levels",
        "version",
        # Variable identifiers
        "standard_name",
        "long_name",
        "units",
        "finalised",
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path")

    version_metadata = "version"
    # See https://wcrp-cmip.github.io/WGCM_Infrastructure_Panel/Papers/CMIP6_global_attributes_filenames_CVs_v6.2.7.pdf
    # under "Directory structure template"
    dataset_id_metadata = (
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "member_id",
        "table_id",
        "variable_id",
        "grid_label",
    )

    def __init__(self, n_jobs: int = 1, config: Config | None = None):
        self.n_jobs = n_jobs
        self.config = config or Config.default()

    def get_parsing_function(self) -> DatasetParsingFunction:
        """
        Get the parsing function for CMIP6 datasets based on configuration

        The parsing function used is determined by the `cmip6_parser` configuration value:
        - "drs": Use the DRS parser (default)
        - "complete": Use the complete parser that extracts all available metadata

        Returns
        -------
        :
            The appropriate parsing function based on configuration
        """
        parser_type = self.config.cmip6_parser
        if parser_type == "complete":
            logger.info("Using complete CMIP6 parser")
            return parse_cmip6_complete
        else:
            logger.info(f"Using DRS CMIP6 parser (config value: {parser_type})")
            return parse_cmip6_drs

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
        parsing_function = self.get_parsing_function()

        with warnings.catch_warnings():
            # Ignore the DeprecationWarning from xarray
            warnings.simplefilter("ignore", DeprecationWarning)

            builder = Builder(
                paths=[str(file_or_directory)],
                depth=10,
                include_patterns=["*.nc"],
                joblib_parallel_kwargs={"n_jobs": self.n_jobs},
            ).build(parsing_func=parsing_function)

        datasets: pd.DataFrame = builder.df.drop(["init_year"], axis=1)

        # Convert the start_time and end_time columns to datetime objects
        # We don't know the calendar used in the dataset (TODO: Check what ecgtools does)
        datasets["start_time"] = _parse_datetime(datasets["start_time"])
        datasets["end_time"] = _parse_datetime(datasets["end_time"])

        drs_items = [
            *self.dataset_id_metadata,
            self.version_metadata,
        ]
        datasets["instance_id"] = datasets.apply(
            lambda row: "CMIP6." + ".".join([row[item] for item in drs_items]), axis=1
        )

        # Add in any missing metadata columns
        missing_columns = set(self.dataset_specific_metadata + self.file_specific_metadata) - set(
            datasets.columns
        )
        if missing_columns:
            for column in missing_columns:
                datasets[column] = pd.NA

        # Temporary fix for some datasets
        # TODO: Replace with a standalone package that contains metadata fixes for CMIP6 datasets
        datasets = _apply_fixes(datasets)

        return datasets
