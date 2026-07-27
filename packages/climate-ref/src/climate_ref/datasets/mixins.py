"""
Mixins for dataset adapters that support lazy finalization.
"""

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

import pandas as pd
from loguru import logger

from climate_ref.database import Database
from climate_ref.datasets.base import DatasetParsingFunction
from climate_ref.datasets.catalog_builder import parse_files
from climate_ref.datasets.utils import _is_na, parse_cftime_dates

DEFAULT_FINALISE_CHUNK_SIZE = 500
"""
Number of files that are parsed and committed per finalisation chunk.

Each chunk is committed before the next one is parsed,
so cancelling a long finalisation keeps the work that has already been done.
"""


def _chunk_by_dataset(datasets: pd.DataFrame, slug_column: str, chunk_size: int) -> Iterator[pd.Index]:
    """
    Split a catalog into chunks of index labels holding roughly ``chunk_size`` files each.

    A dataset is never split across two chunks,
    so each chunk can be parsed and committed as a unit.

    Parameters
    ----------
    datasets
        Catalog to split. The index must be unique.
    slug_column
        Column holding the dataset identifier.
    chunk_size
        Soft target for the number of files per chunk.
        A chunk exceeds this when a single dataset has more files.

    Yields
    ------
    :
        Index labels for each chunk.
    """
    labels: list[Any] = []

    for _, group in datasets.groupby(slug_column, sort=False):
        labels.extend(group.index.tolist())
        if len(labels) >= chunk_size:
            yield pd.Index(labels)
            labels = []

    if labels:
        yield pd.Index(labels)


class FinaliseableDatasetAdapterMixin:
    """
    Mixin for dataset adapters that support two-phase ingestion.

    Phase 1 (bootstrap): Fast ingestion from directory/filename metadata only.
    Phase 2 (finalisation): Open files to extract full metadata for a subset.

    This requires two parsing functions: one for the initial bootstrap phase that extracts minimal metadata,
    and one for the finalisation phase that extracts full metadata.
    """

    @abstractmethod
    def get_complete_parser(self) -> DatasetParsingFunction:
        """
        Return the parsing function that opens files to extract full metadata.

        Returns
        -------
        :
            Parsing function for complete metadata extraction
        """
        ...

    def _post_finalise_fixes(self, datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Apply any adapter-specific fixes after finalisation.

        Default implementation is a no-op. Subclasses may override.

        Parameters
        ----------
        datasets
            DataFrame with finalised metadata

        Returns
        -------
        :
            DataFrame with fixes applied
        """
        return datasets

    def finalise_datasets(
        self,
        db: Database,
        datasets: pd.DataFrame,
        chunk_size: int = DEFAULT_FINALISE_CHUNK_SIZE,
    ) -> pd.DataFrame:
        """
        Finalise unfinalised datasets by opening files to extract full metadata.

        Files are parsed in parallel using ``self.n_jobs`` worker processes,
        mirroring the parallelism used during ingest.

        Work is done in chunks of whole datasets.
        Each chunk is committed before the next one is parsed,
        so progress is visible and a cancelled run keeps what it has already parsed.

        Parameters
        ----------
        db
            Database instance for persisting updated metadata
        datasets
            DataFrame containing datasets to finalise (should have finalised=False)
        chunk_size
            Soft target for the number of files parsed and committed per chunk

        Returns
        -------
        :
            Updated DataFrame with full metadata extracted from files
        """
        # The catalog is indexed by dataset id, which repeats once per file.
        # Work against a unique index so per-row updates address a single file
        # and chunks can be reassembled without ambiguity.
        original_index = datasets.index
        working = datasets.set_axis(pd.RangeIndex(len(datasets)))

        unfinalised = working[working["finalised"] == False]  # noqa: E712
        total_files = int(unfinalised["path"].notna().sum())
        if not total_files:
            return datasets

        slug_column = self.slug_column  # type: ignore[attr-defined]
        parsing_func = self.get_complete_parser()
        n_jobs = self.n_jobs if hasattr(self, "n_jobs") else 1

        logger.info(
            f"Finalising {unfinalised[slug_column].nunique()} datasets ({total_files} files) "
            f"using {n_jobs} worker(s)"
        )

        # Chunk over every file of an affected dataset, not just its unfinalised ones,
        # so per-dataset fixes and the per-dataset commit see the whole dataset.
        affected = working[working[slug_column].isin(unfinalised[slug_column].unique())]
        untouched = working.drop(index=affected.index)
        chunks: list[pd.DataFrame] = [untouched] if len(untouched) else []

        parsed_files = 0
        for labels in _chunk_by_dataset(affected, slug_column, chunk_size):
            chunk = working.loc[labels].copy()
            attempted = int(chunk.loc[chunk["finalised"] == False, "path"].notna().sum())  # noqa: E712

            chunks.append(self._finalise_chunk(db, chunk, parsing_func, n_jobs))

            parsed_files += attempted
            logger.info(f"Parsed {parsed_files}/{total_files} files")

        return pd.concat(chunks).sort_index().set_axis(original_index)

    def _finalise_chunk(
        self,
        db: Database,
        chunk: pd.DataFrame,
        parsing_func: DatasetParsingFunction,
        n_jobs: int,
    ) -> pd.DataFrame:
        """
        Parse, fix and persist a single chunk of unfinalised rows.

        Parameters
        ----------
        db
            Database instance for persisting updated metadata
        chunk
            Every row of a whole number of datasets, with a unique index.
            Rows that are already finalised are left alone.
        parsing_func
            Parser that opens each file to extract full metadata
        n_jobs
            Number of parallel workers to parse the chunk with

        Returns
        -------
        :
            The chunk with metadata extracted from the files that parsed successfully
        """
        pending = chunk.loc[chunk["finalised"] == False, "path"]  # noqa: E712
        valid = [(label, str(path)) for label, path in pending.items() if not pd.isna(path)]
        if not valid:
            return chunk

        labels, paths = zip(*valid)
        parsed_results = parse_files(list(paths), parsing_func, n_jobs=n_jobs)

        updated_labels = []
        for label, path, parsed in zip(labels, paths, parsed_results):
            if "INVALID_ASSET" in parsed:
                logger.warning(f"Failed to finalise {path}: {parsed.get('TRACEBACK', '')}")
                continue

            for key, value in parsed.items():
                if key in chunk.columns and value is not None:
                    chunk.at[label, key] = value

            chunk.at[label, "finalised"] = True
            updated_labels.append(label)

        if updated_labels:
            # Convert start_time/end_time strings from the complete parser to cftime objects
            mask = chunk.index.isin(updated_labels)
            cal = chunk.loc[mask, "calendar"] if "calendar" in chunk.columns else "standard"
            chunk.loc[mask, "start_time"] = parse_cftime_dates(chunk.loc[mask, "start_time"], cal).values
            chunk.loc[mask, "end_time"] = parse_cftime_dates(chunk.loc[mask, "end_time"], cal).values

            # Apply adapter-specific fixes.
            # A chunk holds whole datasets, so per-dataset fixes see all of their files.
            chunk = self._post_finalise_fixes(chunk)

        self._persist_finalised_metadata(db, chunk, chunk.index)

        return chunk

    def _persist_finalised_metadata(
        self, db: Database, datasets: pd.DataFrame, unfinalised_index: pd.Index
    ) -> None:
        """
        Persist finalised metadata back to the database.

        We update records directly rather than calling register_dataset,
        because the solver passes a group subset that may not contain all
        files for the dataset, which would trigger a "removing files" error.

        Parameters
        ----------
        db
            Database instance
        datasets
            DataFrame with updated metadata
        unfinalised_index
            Index of rows that were originally unfinalised
        """
        dataset_cls = self.dataset_cls  # type: ignore[attr-defined]
        slug_column = self.slug_column  # type: ignore[attr-defined]
        dataset_specific_metadata = self.dataset_specific_metadata  # type: ignore[attr-defined]
        file_specific_metadata = self.file_specific_metadata  # type: ignore[attr-defined]

        finalised_mask = datasets["finalised"] == True  # noqa: E712
        originally_unfinalised = datasets.index.isin(unfinalised_index)
        seen_slugs: set[str] = set()
        for _idx, row in datasets[finalised_mask & originally_unfinalised].iterrows():
            slug = row.get(slug_column)
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            try:
                # TODO: Should this be a session or a transaction
                with db.session.begin():
                    dataset_record = (
                        db.session.query(dataset_cls)
                        .filter(getattr(dataset_cls, slug_column) == slug)
                        .one_or_none()
                    )
                    if dataset_record is None:
                        logger.warning(
                            f"No dataset with slug {slug!r} found in database when finalising. Skipping"
                        )
                        continue

                    # Update dataset-level metadata from the first finalised row.
                    # Use _is_na to skip None, pd.NA, and np.nan — matching
                    # register_dataset's filtering — so we never overwrite
                    # real values with NA sentinels.
                    for col in dataset_specific_metadata:
                        if col in datasets.columns:
                            val = row.get(col)
                            if not _is_na(val) and hasattr(dataset_record, col):
                                setattr(dataset_record, col, val)
                    dataset_record.finalised = True

                    # Update file-level metadata for files in this subset.
                    # Use file_specific_metadata (excluding "path") so adapters
                    # like CMIP7 can persist tracking_id alongside start/end times.
                    file_metadata_cols = [
                        c for c in file_specific_metadata if c != "path" and c in datasets.columns
                    ]
                    subset = datasets[datasets[slug_column] == slug]
                    file_metadata_map = {
                        str(r["path"]): {c: r.get(c) for c in file_metadata_cols}
                        for _, r in subset.iterrows()
                    }
                    for f in dataset_record.files:
                        for col, val in file_metadata_map.get(f.path, {}).items():
                            if not _is_na(val) and hasattr(f, col):
                                setattr(f, col, val)
            except Exception:
                logger.exception(f"Error persisting finalised dataset {slug}")
                # Mark the dataset as unfinalised in the DataFrame to stay
                # consistent with the DB (where the update was not committed).
                datasets.loc[datasets[slug_column] == slug, "finalised"] = False
