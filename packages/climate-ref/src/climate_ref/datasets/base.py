from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
from attrs import define
from loguru import logger
from sqlalchemy import Select
from sqlalchemy.orm import selectinload

from climate_ref.database import Database, ModelState
from climate_ref.datasets.utils import _is_na, _to_db_str, coerce_catalog_times, validate_path
from climate_ref.models.dataset import Dataset, DatasetFile
from climate_ref.models.dataset_query import DatasetFilter, select_datasets
from climate_ref_core.datasets import select_latest_version
from climate_ref_core.exceptions import RefException


@define
class DatasetRegistrationResult:
    """
    Result of registering a dataset, containing information about file changes
    """

    dataset: Dataset
    dataset_state: ModelState | None
    files_added: list[str]
    files_updated: list[str]
    files_removed: list[str]
    files_unchanged: list[str]

    @property
    def total_changes(self) -> int:
        """Total number of file changes (added + updated + removed)"""
        return len(self.files_added) + len(self.files_updated) + len(self.files_removed)


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
        if "time_range" in data_catalog.columns and "time_range" not in invalid_dataset_columns:
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
        file
            File or directory to parse

        kwargs
            Additional keyword arguments to pass to the parsing function.

        Returns
        -------
        :
            Data catalog containing the metadata for the dataset
        """
        ...


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
    columns_requiring_finalisation: frozenset[str] = frozenset()
    """
    Columns that are not available until the dataset has been finalised.

    For adapters that support two-phase parsing (e.g. DRS-only then complete),
    these columns will contain ``pd.NA`` until finalisation opens the files.
    Filtering or grouping on these columns before finalisation will silently
    produce incorrect results.
    """

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
    derived_metadata: tuple[str, ...] = ()
    """
    Columns that are not stored in the database but are computed from stored metadata.

    These are reconstructed by :meth:`_add_derived_columns` whenever a catalog is loaded,
    so that a DB-loaded catalog has the same columns as a freshly parsed one
    (e.g. CMIP7's ``branded_variable``).
    :meth:`load_catalog` enforces that every column listed here is present after loading.
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

    def register_dataset(  # noqa: PLR0912, PLR0915
        self, db: Database, data_catalog_dataset: pd.DataFrame
    ) -> DatasetRegistrationResult:
        """
        Register a dataset in the database using the data catalog

        This assumes that the data catalog has already been validated with `validate_data_catalog`
        to ensure that the dataset-specific metadata is consistent across all files in the dataset.

        Parameters
        ----------
        db
            Database instance
        data_catalog_dataset
            A subset of the data catalog containing the metadata for a single dataset


        Raises
        ------
        RefException
            If the data catalog contains validation errors that should have been caught by
                `validate_data_catalog` (i.e. multiple unique slugs in `slug_column`).


        Returns
        -------
        :
            Registration result with dataset and file change information
        """
        DatasetModel = self.dataset_cls

        unique_slugs = data_catalog_dataset[self.slug_column].unique()
        if len(unique_slugs) != 1:
            raise RefException(f"Found multiple datasets in the same directory: {unique_slugs}")
        slug = unique_slugs[0]

        # Callers are responsible for validating the catalog with  ``validate_data_catalog`` before invoking.
        # This is a strict subset of ``validate_data_catalog`` to catch skipping upstream validation.
        slice_meta = data_catalog_dataset[list(self.dataset_specific_metadata)]
        if (slice_meta.nunique(dropna=False) > 1).any():
            raise RefException(
                f"Dataset {slug} has inconsistent dataset-specific metadata; "
                "callers must pre-validate the catalog with validate_data_catalog."
            )

        # Check if the incoming data is unfinalised (DRS parser) and the dataset
        # already exists as finalised.  In that case, skip the entire update to
        # avoid regressing metadata that was populated during finalisation.
        incoming_finalised = data_catalog_dataset.get("finalised")
        is_unfinalised_ingest = incoming_finalised is not None and not incoming_finalised.iloc[0]

        if is_unfinalised_ingest:
            existing = db.session.query(DatasetModel).filter_by(slug=slug).first()
            if existing and existing.finalised:
                logger.debug(f"Skipping already-finalised dataset: {slug}")
                current_files = db.session.query(DatasetFile).filter_by(dataset_id=existing.id).all()
                existing_paths = {f.path for f in current_files}

                new_file_data = data_catalog_dataset.to_dict(orient="records")
                new_paths = {str(validate_path(r["path"])) for r in new_file_data}
                files_to_add = new_paths - existing_paths

                if files_to_add:
                    new_file_lookup = {}
                    for r in new_file_data:
                        p = str(validate_path(r["path"]))
                        new_file_lookup[p] = {"start_time": r["start_time"], "end_time": r["end_time"]}

                    for file_path in files_to_add:
                        ft = new_file_lookup[file_path]
                        db.session.add(
                            DatasetFile(
                                path=file_path,
                                dataset_id=existing.id,
                                start_time=ft["start_time"],
                                end_time=ft["end_time"],
                            )
                        )

                return DatasetRegistrationResult(
                    dataset=existing,
                    dataset_state=ModelState.UPDATED if files_to_add else None,
                    files_added=list(files_to_add),
                    files_updated=[],
                    files_removed=[],
                    files_unchanged=list(existing_paths & new_paths),
                )

        # Upsert the dataset (create a new dataset or update the metadata)
        dataset_metadata = cast(
            dict[str, Any], data_catalog_dataset[list(self.dataset_specific_metadata)].iloc[0].to_dict()
        )

        # Strip NA/NaN values so we never overwrite existing metadata with empty values.
        # This prevents DRS re-ingestion from regressing columns that were populated
        # during finalisation.
        dataset_metadata = {k: v for k, v in dataset_metadata.items() if not _is_na(v)}

        dataset, dataset_state = db.update_or_create(DatasetModel, defaults=dataset_metadata, slug=slug)
        if dataset_state == ModelState.CREATED:
            logger.info(f"Created new dataset: {dataset}")
        elif dataset_state == ModelState.UPDATED:
            logger.info(f"Updating existing dataset: {dataset}")
        db.session.flush()

        # Initialize result tracking
        files_added: list[str] = []
        files_updated: list[str] = []
        files_removed: list[str] = []
        files_unchanged: list[str] = []

        # Get current files for this dataset
        current_files = db.session.query(DatasetFile).filter_by(dataset_id=dataset.id).all()
        current_file_paths = {f.path: f for f in current_files}

        # Columns to store per file (indexed by path)
        file_meta_cols = [c for c in self.file_specific_metadata if c != "path"]

        # Get new file data from data catalog
        new_file_data = data_catalog_dataset.to_dict(orient="records")
        new_file_lookup = {}
        for dataset_file in new_file_data:
            file_path = str(validate_path(dataset_file["path"]))
            new_file_lookup[file_path] = {c: dataset_file.get(c) for c in file_meta_cols if c in dataset_file}

        new_file_paths = set(new_file_lookup.keys())
        existing_file_paths = set(current_file_paths.keys())

        # Files that exist in the database but are absent from the incoming catalog
        # slice. Removal isn't supported yet (we want to preserve a record of
        # diagnostics that already used the file), but raising here aborts the
        # whole ingest -- including the streaming path, where a dataset can appear
        # in two different chunks (e.g. a CMIP6 mirror that stores the same
        # netCDF file under both an "actual" activity directory and a misfiled
        # parent activity directory). Emit a warning and keep the existing rows
        # in place so subsequent register_dataset calls for the same slug just
        # add their own paths.
        files_to_remove = existing_file_paths - new_file_paths
        if files_to_remove:
            logger.warning(
                f"Dataset {slug}: {len(files_to_remove)} file(s) absent from the current ingest "
                f"are being kept (removal not yet supported): {sorted(files_to_remove)}"
            )

        # Update existing files if any file-specific metadata has changed.
        # Compare via _to_db_str on the incoming value so it matches the on-disk str form
        # (DatasetFile.@validates coerces cftime -> str).
        for file_path, existing_file in current_file_paths.items():
            if file_path in new_file_lookup:
                new_meta = new_file_lookup[file_path]
                changed = any(
                    not _is_na(new_meta.get(c))
                    and hasattr(existing_file, c)
                    and getattr(existing_file, c) != _to_db_str(new_meta[c])
                    for c in file_meta_cols
                    if c in new_meta
                )
                if changed:
                    logger.warning(f"Updating file metadata for {file_path}")
                    for c in file_meta_cols:
                        if c in new_meta and not _is_na(new_meta[c]) and hasattr(existing_file, c):
                            setattr(existing_file, c, new_meta[c])
                    files_updated.append(file_path)
                else:
                    files_unchanged.append(file_path)

        # Add new files (batch operation)
        files_to_add = new_file_paths - existing_file_paths
        if files_to_add:
            files_added = list(files_to_add)
            new_dataset_files = []
            for file_path in files_to_add:
                file_meta = new_file_lookup[file_path]
                # Filter out NA values before passing to DatasetFile constructor
                clean_meta = {c: v for c, v in file_meta.items() if not _is_na(v)}
                new_dataset_files.append(
                    DatasetFile(
                        path=file_path,
                        dataset_id=dataset.id,
                        **clean_meta,
                    )
                )
            db.session.add_all(new_dataset_files)

        # Determine final dataset state
        # If dataset metadata changed, use that state
        # If no metadata changed but files changed, consider it updated
        # If nothing changed, keep the original state (None for existing, CREATED for new)
        final_dataset_state = dataset_state
        if dataset_state is None and (files_added or files_updated or files_removed):
            final_dataset_state = ModelState.UPDATED

        result = DatasetRegistrationResult(
            dataset=dataset,
            dataset_state=final_dataset_state,
            files_added=files_added,
            files_updated=files_updated,
            files_removed=files_removed,
            files_unchanged=files_unchanged,
        )
        change_message = f": ({final_dataset_state.name})" if final_dataset_state else ""
        logger.debug(
            f"Dataset registration complete for {dataset.slug}{change_message} "
            f"{len(files_added)} files added, "
            f"{len(files_updated)} files updated, "
            f"{len(files_removed)} files removed, "
            f"{len(files_unchanged)} files unchanged"
        )

        return result

    def filter_latest_versions(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Filter a data catalog to only include the latest version of each dataset

        Delegates to :func:`select_latest_version`, which compares versions numerically
        (so ``v10`` > ``v2``, not lexicographically), grouping by ``dataset_id_metadata``.

        Parameters
        ----------
        catalog
            Data catalog to filter

        Returns
        -------
        :
            Filtered data catalog with only the latest version of each dataset
        """
        if catalog.empty or not self.dataset_id_metadata:
            return catalog

        return select_latest_version(
            catalog,
            version_column=self.version_metadata,
            group_by=self.dataset_id_metadata,
        )

    def _dataset_query(self) -> Select[Any]:
        """
        Build the ``Select`` for this adapter's latest-version datasets via the shared query builder.

        Routing through ``climate_ref.models.dataset_query.select_datasets`` keeps ONE definition of
        "the datasets query" behind both ``load_catalog`` and ``climate_ref.results``'s
        ``reader.datasets``, so the two cannot drift apart.

        Passes this adapter's ``dataset_id_metadata`` as ``latest_group_by`` so deduplication to the
        latest version happens in SQL using a ``RANK`` window keyed on ``version_key``.
        """
        source_type = self.dataset_cls.__mapper_args__["polymorphic_identity"]
        return select_datasets(
            DatasetFilter(source_type=source_type), latest_group_by=self.dataset_id_metadata
        )

    def _get_dataset_files(self, db: Database) -> pd.DataFrame:
        # Eager-load files to avoid N+1 (one query per dataset), then explode to one row per file.
        # No SQL limit here: ``limit`` bounds *files* for this path (applied after exploding, in
        # ``load_catalog``), not datasets, so it cannot be pushed into this dataset-level query.
        stmt = self._dataset_query().options(selectinload(self.dataset_cls.files))  # type: ignore[attr-defined]
        datasets = db.session.execute(stmt).scalars().unique().all()

        return pd.DataFrame(
            [
                {
                    **{k: getattr(file, k) for k in self.file_specific_metadata},
                    **{k: getattr(dataset, k) for k in self.dataset_specific_metadata},
                    "finalised": dataset.finalised,
                }
                for dataset in datasets
                for file in dataset.files
            ],
            index=[dataset.id for dataset in datasets for _file in dataset.files],
        )

    def _get_datasets(self, db: Database, limit: int | None = None) -> pd.DataFrame:
        stmt = self._dataset_query()
        if limit is not None:
            stmt = stmt.limit(limit)
        result_datasets = db.session.execute(stmt).scalars().unique().all()

        return pd.DataFrame(
            [{k: getattr(dataset, k) for k in self.dataset_specific_metadata} for dataset in result_datasets],
            index=[dataset.id for dataset in result_datasets],
        )

    def load_catalog(
        self, db: Database, include_files: bool = True, limit: int | None = None
    ) -> pd.DataFrame:
        """
        Load the data catalog containing the currently tracked datasets/files from the database

        Iterating over different datasets within the data catalog can be done using a `groupby`
        operation for the `instance_id` column.

        Only the latest version of each dataset is returned.
        Deduplication happens in SQL (``_dataset_query``), not in pandas.

        The index of the data catalog is the primary key of the dataset.
        This should be maintained during any processing.

        ``limit`` (when given) bounds the returned rows after deduplicating to the latest version,
        so up to ``limit`` datasets are returned when ``include_files=False`` (the limit is pushed
        into the SQL query). When ``include_files=True``, ``limit`` bounds *files* instead (per the
        CLI help text), so it is applied in Python after exploding datasets to one row per file --
        it is not pushed into the dataset-level query in that case.

        Returns
        -------
        :
            Data catalog containing the metadata for the currently ingested datasets
        """
        with db.session.begin():
            # TODO: Paginate this query to avoid loading all the data at once
            if include_files:
                catalog = self._get_dataset_files(db)
            else:
                catalog = self._get_datasets(db, limit=limit)

        # If there are no datasets, return an empty DataFrame
        if catalog.empty:
            empty = pd.DataFrame(columns=self.dataset_specific_metadata + self.file_specific_metadata)
            return self._finalise_loaded_catalog(empty)

        # Convert start_time/end_time strings from DB to cftime objects
        catalog = coerce_catalog_times(catalog)

        # include_files=True: the SQL query already deduplicated to the latest version; the file-count
        # limit is applied here, in Python, after exploding to one row per file (it bounds files, not
        # datasets -- see the docstring). include_files=False: dedup and limit both already happened
        # in SQL (``_get_datasets``), so this is a no-op pass-through.
        if include_files and limit is not None:
            catalog = catalog.head(limit)
        return self._finalise_loaded_catalog(catalog)

    def _finalise_loaded_catalog(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns and enforce the loaded-catalog invariant.

        Every column listed in :attr:`derived_metadata` must be present after loading,
        so downstream code (e.g. the solver applying data requirement filters) can rely on its existence.
        """
        catalog = self._add_derived_columns(catalog)

        missing = set(self.derived_metadata) - set(catalog.columns)
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} did not produce its declared derived column(s): {sorted(missing)}"
            )
        return catalog

    def _add_derived_columns(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns to a loaded data catalog.

        Derived columns are computed from stored metadata rather than persisted in
        the database (e.g. CMIP7's ``branded_variable``).
        They must be reconstructed whenever a catalog is loaded from the database.

        Adapters that declare :attr:`derived_metadata` must override this to populate
        every column listed there; the base implementation is a no-op.

        Parameters
        ----------
        catalog
            Data catalog loaded from the database

        Returns
        -------
        :
            Data catalog with any derived columns added
        """
        return catalog
