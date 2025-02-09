from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import ecgtools.parsers
import pandas as pd
from ecgtools import Builder
from loguru import logger
from sqlalchemy.orm import joinedload

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.datasets.base import DatasetAdapter
from cmip_ref.datasets.utils import validate_path
from cmip_ref.models.dataset import CMIP6Dataset, CMIP6File, Dataset
from cmip_ref_core.exceptions import RefException


def _parse_datetime(dt_str: pd.Series[str]) -> pd.Series[datetime | Any]:
    """
    Pandas tries to coerce everything to their own datetime format, which is not what we want here.
    """

    def _inner(date_string: str | None) -> datetime | None:
        if not date_string:
            return None

        # Try to parse the date string with and without milliseconds
        try:
            dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")

        return dt

    return pd.Series(
        [_inner(dt) for dt in dt_str],
        index=dt_str.index,
        dtype="object",
    )


def _apply_fixes(data_catalog: pd.DataFrame) -> pd.DataFrame:
    def _fix_parent_variant_label(group: pd.DataFrame) -> pd.DataFrame:
        if group["parent_variant_label"].nunique() == 1:
            return group
        group["parent_variant_label"] = group["variant_label"].iloc[0]

        return group

    data_catalog = (
        data_catalog.groupby("instance_id")
        .apply(_fix_parent_variant_label, include_groups=False)
        .reset_index(level="instance_id")
    )

    data_catalog["branch_time_in_child"] = _clean_branch_time(data_catalog["branch_time_in_child"])
    data_catalog["branch_time_in_parent"] = _clean_branch_time(data_catalog["branch_time_in_parent"])

    return data_catalog


def _clean_branch_time(branch_time: pd.Series[str]) -> pd.Series[float]:
    # EC-Earth3 uses "D" as a suffix for the branch_time_in_child and branch_time_in_parent columns
    # Handle missing values (these result in nan values)
    return pd.to_numeric(branch_time.astype(str).str.replace("D", "").replace("None", ""), errors="raise")


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
        "standard_name",
        "long_name",
        "units",
        "vertical_levels",
        "init_year",
        "version",
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path")

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def pretty_subset(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Get a subset of the data_catalog to pretty print

        This is particularly useful for CMIP6 datasets, which have a lot of metadata columns.

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
                "experiment_id",
                "member_id",
                "table_id",
                "variable_id",
                "grid_label",
                "version",
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            builder = Builder(
                paths=[str(file_or_directory)],
                depth=10,
                include_patterns=["*.nc"],
                joblib_parallel_kwargs={"n_jobs": self.n_jobs},
            ).build(parsing_func=ecgtools.parsers.parse_cmip6)

        datasets = builder.df

        # Convert the start_time and end_time columns to datetime objects
        # We don't know the calendar used in the dataset (TODO: Check what ecgtools does)
        datasets["start_time"] = _parse_datetime(datasets["start_time"])
        datasets["end_time"] = _parse_datetime(datasets["end_time"])

        drs_items = [
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "version",
        ]
        datasets["instance_id"] = datasets.apply(
            lambda row: "CMIP6." + ".".join([row[item] for item in drs_items]), axis=1
        )

        # Temporary fix for some datasets
        # TODO: Replace with a standalone package that contains metadata fixes for CMIP6 datasets
        datasets = _apply_fixes(datasets)

        return datasets

    def register_dataset(
        self, config: Config, db: Database, data_catalog_dataset: pd.DataFrame
    ) -> CMIP6Dataset | None:
        """
        Register a dataset in the database using the data catalog

        Parameters
        ----------
        config
            Configuration object
        db
            Database instance
        data_catalog_dataset
            A subset of the data catalog containing the metadata for a single dataset

        Returns
        -------
        :
            Registered dataset if successful, else None
        """
        self.validate_data_catalog(data_catalog_dataset)

        unique_slugs = data_catalog_dataset[self.slug_column].unique()
        if len(unique_slugs) != 1:
            raise RefException(f"Found multiple datasets in the same directory: {unique_slugs}")
        slug = unique_slugs[0]

        dataset_metadata = data_catalog_dataset[list(self.dataset_specific_metadata)].iloc[0].to_dict()
        dataset, created = db.get_or_create(self.dataset_cls, slug=slug, **dataset_metadata)

        if not created:
            logger.warning(f"{dataset} already exists in the database. Skipping")
            return None

        db.session.flush()

        for dataset_file in data_catalog_dataset.to_dict(orient="records"):
            path = validate_path(config, dataset_file.pop("path"))

            db.session.add(
                CMIP6File(
                    path=str(path),
                    dataset_id=dataset.id,
                    start_time=dataset_file.pop("start_time"),
                    end_time=dataset_file.pop("end_time"),
                )
            )

        return dataset

    def load_catalog(
        self, db: Database, include_files: bool = True, limit: int | None = None
    ) -> pd.DataFrame:
        """
        Load the data catalog containing the currently tracked datasets/files from the database

        Iterating over different datasets within the data catalog can be done using a `groupby`
        operation for the `instance_id` column.

        The index of the data catalog is the primary key of the dataset.
        This should be maintained during any processing.

        Returns
        -------
        :
            Data catalog containing the metadata for the currently ingested datasets
        """
        # TODO: Paginate this query to avoid loading all the data at once
        if include_files:
            result = (
                db.session.query(CMIP6File)
                # The join is necessary to be able to order by the dataset columns
                .join(CMIP6File.dataset)
                # The joinedload is necessary to avoid N+1 queries (one for each dataset)
                # https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html#the-zen-of-joined-eager-loading
                .options(joinedload(CMIP6File.dataset))
                .order_by(Dataset.updated_at.desc())
                .limit(limit)
                .all()
            )

            return pd.DataFrame(
                [
                    {
                        **{k: getattr(file, k) for k in self.file_specific_metadata},
                        **{k: getattr(file.dataset, k) for k in self.dataset_specific_metadata},
                    }
                    for file in result
                ],
                index=[file.dataset.id for file in result],
            )
        else:
            result_datasets = (
                db.session.query(CMIP6Dataset).order_by(Dataset.updated_at.desc()).limit(limit).all()
            )

            return pd.DataFrame(
                [
                    {k: getattr(dataset, k) for k in self.dataset_specific_metadata}
                    for dataset in result_datasets
                ],
                index=[file.id for file in result_datasets],
            )
