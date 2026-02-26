"""
CMIP7 Dataset Adapter

Adapter for parsing and registering CMIP7 datasets based on CMIP7 Global Attributes v1.0.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from climate_ref.config import Config
from climate_ref.datasets.base import DatasetAdapter, DatasetParsingFunction
from climate_ref.datasets.catalog_builder import build_catalog
from climate_ref.datasets.cmip7_parsers import parse_cmip7_complete, parse_cmip7_drs
from climate_ref.datasets.mixins import FinaliseableDatasetAdapterMixin
from climate_ref.datasets.utils import clean_branch_time, parse_datetime
from climate_ref.models.dataset import CMIP7Dataset


class CMIP7DatasetAdapter(FinaliseableDatasetAdapterMixin, DatasetAdapter):
    """
    Adapter for CMIP7 datasets

    Based on CMIP7 Global Attributes v1.0 (DOI: 10.5281/zenodo.17250297).
    """

    dataset_cls = CMIP7Dataset
    slug_column = "instance_id"

    columns_requiring_finalisation = frozenset(
        {
            # Optional information
            "realm",
            "nominal_resolution",
            "license_id",
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
            "time_units",
            "calendar",
        }
    )

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
        # # Time encoding metadata
        # "time_units",
        # "calendar",
        # Finalisation status
        "finalised",
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

    def get_complete_parser(self) -> DatasetParsingFunction:
        """
        Return the complete parser that opens files to extract full CMIP7 metadata.

        Returns
        -------
        :
            Complete CMIP7 parsing function
        """
        return parse_cmip7_complete

    def _post_finalise_fixes(self, datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Apply CMIP7-specific fixes after finalisation.

        Cleans branch time values that may be stored as strings with units suffixes.

        Parameters
        ----------
        datasets
            DataFrame with finalised metadata

        Returns
        -------
        :
            DataFrame with fixes applied
        """
        if "branch_time_in_child" in datasets.columns:
            datasets["branch_time_in_child"] = clean_branch_time(datasets["branch_time_in_child"])
        if "branch_time_in_parent" in datasets.columns:
            datasets["branch_time_in_parent"] = clean_branch_time(datasets["branch_time_in_parent"])
        return datasets

    def get_parsing_function(self) -> DatasetParsingFunction:
        """
        Get the parsing function for CMIP7 datasets based on configuration

        The parsing function used is determined by the `cmip7_parser` configuration value:
        - "drs": Use the DRS parser (default)
        - "complete": Use the complete parser that extracts all available metadata

        Returns
        -------
        :
            The appropriate parsing function based on configuration
        """
        parser_type = self.config.cmip7_parser
        if parser_type == "complete":
            logger.info("Using complete CMIP7 parser")
            return parse_cmip7_complete
        else:
            logger.info(f"Using DRS CMIP7 parser (config value: {parser_type})")
            return parse_cmip7_drs

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
        parsing_function = self.get_parsing_function()

        datasets = build_catalog(
            paths=[str(file_or_directory)],
            parsing_func=parsing_function,
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
