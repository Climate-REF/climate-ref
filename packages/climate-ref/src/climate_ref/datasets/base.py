from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import pandas as pd
from loguru import logger
from sqlalchemy.orm import joinedload

from climate_ref.database import Database
from climate_ref.datasets.utils import validate_path
from climate_ref.models.dataset import Dataset, DatasetFile
from climate_ref_core.exceptions import RefException


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


class DatasetParsingFunction(Protocol):
    """
    Protocol for a function that parses metadata from a file or directory
    """

    def __call__(self, file: str, **kwargs: Any) -> dict[str, Any]:
        """
        Parse a file or directory and return metadata for the dataset

        Parameters
        ----------
        file_or_directory
            File or directory to parse

        kwargs
            Additional keyword arguments to pass to the parsing function.

        Returns
        -------
        :
            Data catalog containing the metadata for the dataset
        """
        ...


@runtime_checkable
class DatasetAdapter(Protocol):
    """
    An adapter to provide a common interface for different dataset types

    This allows the same code to work with different dataset types.
    """

    dataset_cls: type[Dataset]
    slug_column: str
    """
    The column in the data catalog that contains the dataset slug.
    The dataset slug is a unique identifier for the dataset that includes the version of the dataset.
    This can be used to group files together that belong to the same dataset.
    """
    dataset_specific_metadata: tuple[str, ...]
    file_specific_metadata: tuple[str, ...] = ()

    version_metadata: str = "version"
    """
    The column in the data catalog that contains the version of the dataset.
    """
    dataset_id_metadata: tuple[str, ...] = ()
    """
    The group of metadata columns that are specific to the dataset excluding the version information.

    Each unique dataset should have the same values for these columns.

    This is generally the columns that describe the `slug` of a dataset,
    excluding the version information.
    """

    def pretty_subset(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Get a subset of the data_catalog to pretty print

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
                *self.dataset_id_metadata,
                self.version_metadata,
            ]
        ]

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

    def register_dataset(self, db: Database, data_catalog_dataset: pd.DataFrame) -> Dataset | None:
        """
        Register a dataset in the database using the data catalog

        If an existing dataset with the same slug already exists in the database,
        this will update the dataset with the new metadata and files.

        Parameters
        ----------
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

        self.validate_data_catalog(data_catalog_dataset)
        unique_slugs = data_catalog_dataset[self.slug_column].unique()
        if len(unique_slugs) != 1:
            raise RefException(f"Found multiple datasets in the same directory: {unique_slugs}")
        slug = unique_slugs[0]

        dataset_metadata = data_catalog_dataset[list(self.dataset_specific_metadata)].iloc[0].to_dict()
        dataset, created = db.create_or_update(DatasetModel, values=dataset_metadata, slug=slug)
        db.session.flush()

        if not created:
            # If the dataset already exists, then we are updating it
            # We need to check if the dataset type matches
            logger.info(f"Updating existing dataset {slug}")
        for dataset_file in data_catalog_dataset.to_dict(orient="records"):
            path = validate_path(dataset_file.pop("path"))

            db.create_or_update(
                DatasetFile,
                values={
                    "dataset_id": dataset.id,
                    **{k: dataset_file[k] for k in self.file_specific_metadata if k in dataset_file},
                },
                path=str(path),
            )
        return dataset

    def _get_dataset_files(self, db: Database, limit: int | None = None) -> pd.DataFrame:
        dataset_type = self.dataset_cls.__mapper_args__["polymorphic_identity"]

        result = (
            db.session.query(DatasetFile)
            # The join is necessary to be able to order by the dataset columns
            .join(DatasetFile.dataset)
            .where(Dataset.dataset_type == dataset_type)
            # The joinedload is necessary to avoid N+1 queries (one for each dataset)
            # https://docs.sqlalchemy.org/en/14/orm/loading_relationships.html#the-zen-of-joined-eager-loading
            .options(joinedload(DatasetFile.dataset.of_type(self.dataset_cls)))
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

    def _get_datasets(self, db: Database, limit: int | None = None) -> pd.DataFrame:
        result_datasets = (
            db.session.query(self.dataset_cls).order_by(Dataset.updated_at.desc()).limit(limit).all()
        )

        return pd.DataFrame(
            [{k: getattr(dataset, k) for k in self.dataset_specific_metadata} for dataset in result_datasets],
            index=[file.id for file in result_datasets],
        )

    def load_catalog(
        self, db: Database, include_files: bool = True, limit: int | None = None
    ) -> pd.DataFrame:
        """
        Load the data catalog containing the currently tracked datasets/files from the database

        Iterating over different datasets within the data catalog can be done using a `groupby`
        operation for the `instance_id` column.

        Only the latest version of each dataset is returned.

        The index of the data catalog is the primary key of the dataset.
        This should be maintained during any processing.

        Returns
        -------
        :
            Data catalog containing the metadata for the currently ingested datasets
        """
        with db.session.begin():
            # TODO: Paginate this query to avoid loading all the data at once
            if include_files:
                catalog = self._get_dataset_files(db, limit)
            else:
                catalog = self._get_datasets(db, limit)

        def _get_latest_version(dataset_catalog: pd.DataFrame) -> pd.DataFrame:
            """
            Get the latest version of each dataset based on the version metadata.

            This assumes that the version can be sorted lexicographically.
            """
            latest_version = dataset_catalog[self.version_metadata].max()

            return cast(
                pd.DataFrame, dataset_catalog[dataset_catalog[self.version_metadata] == latest_version]
            )

        # If there are no datasets, return an empty DataFrame
        if catalog.empty:
            return pd.DataFrame(columns=self.dataset_specific_metadata + self.file_specific_metadata)

        # Group by the dataset ID and get the latest version for each dataset
        return catalog.groupby(
            list(self.dataset_id_metadata), group_keys=False, as_index=False, sort=False
        ).apply(_get_latest_version)
