from pathlib import Path
from typing import Protocol

import pandas as pd
from loguru import logger
from sqlalchemy.orm import joinedload

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.datasets.utils import validate_path
from cmip_ref.models import Base
from cmip_ref.models.dataset import Dataset
from cmip_ref_core.exceptions import RefException


def _log_duplicate_metadata(
    data_catalog: pd.DataFrame, unique_metadata: pd.DataFrame, slug_column: str
) -> None:
    # Drop out the rows where the values are the same
    invalid_datasets = unique_metadata[unique_metadata.gt(1).any(axis=1)]
    # Drop out the columns where the values are the same
    invalid_datasets = invalid_datasets[invalid_datasets.columns[invalid_datasets.gt(1).any(axis=0)]]

    for instance_id in invalid_datasets.index:
        # Get the columns where the values are different
        invalid_dataset_nunique = invalid_datasets.loc[instance_id]
        invalid_dataset_columns = invalid_dataset_nunique[invalid_dataset_nunique.gt(1)].index.tolist()

        # Include time_range in the list of invalid columns to make debugging easier
        invalid_dataset_columns.append("time_range")

        data_catalog_subset = data_catalog[data_catalog[slug_column] == instance_id]

        logger.error(
            f"Dataset {instance_id} has varying metadata:\n{data_catalog_subset[invalid_dataset_columns]}"
        )


class DatasetAdapter(Protocol):
    """
    An adapter to provide a common interface for different dataset types

    This allows the same code to work with different dataset types.
    """

    dataset_cls: type[Dataset]
    file_cls: type[Base]
    slug_column: str
    dataset_specific_metadata: tuple[str, ...]
    file_specific_metadata: tuple[str, ...] = ()

    def pretty_subset(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Get a subset of the data_catalog to pretty print
        """
        ...

    def find_local_datasets(self, file_or_directory: Path) -> pd.DataFrame:
        """
        Generate a data catalog from the specified file or directory

        This data catalog should contain all the metadata needed by the database.
        The index of the data catalog should be the dataset slug.
        """
        ...

    def validate_data_catalog(self, data_catalog: pd.DataFrame, skip_invalid: bool = False) -> pd.DataFrame:
        """
        Validate a data catalog

        Parameters
        ----------
        data_catalog
            Data catalog to validate
        skip_invalid
            If True, ignore datasets with invalid metadata and remove them from the resulting data catalog.

        Raises
        ------
        ValueError
            If `skip_invalid` is False (default) and the data catalog contains validation errors.

        Returns
        -------
        :
            Validated data catalog
        """
        # Check if the data catalog contains the required columns
        missing_columns = set(self.dataset_specific_metadata + self.file_specific_metadata) - set(
            data_catalog.columns
        )
        if missing_columns:
            raise ValueError(f"Data catalog is missing required columns: {missing_columns}")

        # Verify that the dataset specific columns don't vary by dataset by counting the unique values
        # for each dataset and checking if there are any that have more than one unique value.
        unique_metadata = (
            data_catalog[list(self.dataset_specific_metadata)].groupby(self.slug_column).nunique()
        )
        if unique_metadata.gt(1).any(axis=1).any():
            _log_duplicate_metadata(data_catalog, unique_metadata, self.slug_column)

            if skip_invalid:
                data_catalog = data_catalog[
                    ~data_catalog[self.slug_column].isin(
                        unique_metadata[unique_metadata.gt(1).any(axis=1)].index
                    )
                ]
            else:
                raise ValueError("Dataset specific metadata varies by dataset")

        return data_catalog

    def register_dataset(
        self, config: Config, db: Database, data_catalog_dataset: pd.DataFrame
    ) -> Dataset | None:
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
        DatasetModel = self.dataset_cls
        FileModel = self.file_cls

        self.validate_data_catalog(data_catalog_dataset)
        unique_slugs = data_catalog_dataset[self.slug_column].unique()
        if len(unique_slugs) != 1:
            raise RefException(f"Found multiple datasets in the same directory: {unique_slugs}")
        slug = unique_slugs[0]

        dataset_metadata = data_catalog_dataset[list(self.dataset_specific_metadata)].iloc[0].to_dict()
        dataset, created = db.get_or_create(DatasetModel, slug=slug, **dataset_metadata)
        if not created:
            logger.warning(f"{dataset} already exists in the database. Skipping")
            return None
        db.session.flush()
        for dataset_file in data_catalog_dataset.to_dict(orient="records"):
            path = validate_path(dataset_file.pop("path"))

            db.session.add(
                FileModel(
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
        DatasetModel = self.dataset_cls
        FileModel = self.file_cls
        # TODO: Paginate this query to avoid loading all the data at once
        if include_files:
            result = (
                db.session.query(FileModel)
                # The join is necessary to be able to order by the dataset columns
                .join(FileModel.dataset)  # type: ignore
                # The joinedload is necessary to avoid N+1 queries (one for each dataset)
                # https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html#the-zen-of-joined-eager-loading
                .options(joinedload(FileModel.dataset))  # type: ignore
                .order_by(Dataset.updated_at.desc())
                .limit(limit)
                .all()
            )

            return pd.DataFrame(
                [
                    {
                        **{k: getattr(file, k) for k in self.file_specific_metadata},
                        **{k: getattr(file.dataset, k) for k in self.dataset_specific_metadata},  # type: ignore
                    }
                    for file in result
                ],
                index=[file.dataset.id for file in result],  # type: ignore
            )
        else:
            result_datasets = (
                db.session.query(DatasetModel).order_by(Dataset.updated_at.desc()).limit(limit).all()
            )

            return pd.DataFrame(
                [
                    {k: getattr(dataset, k) for k in self.dataset_specific_metadata}
                    for dataset in result_datasets
                ],
                index=[file.id for file in result_datasets],
            )
