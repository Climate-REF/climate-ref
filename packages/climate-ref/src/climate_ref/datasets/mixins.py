"""
Mixins for dataset adapters that support lazy finalization.
"""

from abc import abstractmethod

import pandas as pd
from loguru import logger

from climate_ref.database import Database
from climate_ref.datasets.base import DatasetParsingFunction, _is_na
from climate_ref.datasets.catalog_builder import parse_files
from climate_ref.datasets.utils import parse_cftime_dates


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

    def finalise_datasets(self, db: Database, datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Finalise unfinalised datasets by opening files to extract full metadata.

        Files are parsed in parallel using ``self.n_jobs`` threads,
        mirroring the parallelism used during ingest.

        Parameters
        ----------
        db
            Database instance for persisting updated metadata
        datasets
            DataFrame containing datasets to finalise (should have finalised=False)

        Returns
        -------
        :
            Updated DataFrame with full metadata extracted from files
        """
        unfinalised = datasets[datasets["finalised"] == False]  # noqa: E712

        # Collect (index, path) pairs for rows that have a valid path
        valid = [(idx, str(row["path"])) for idx, row in unfinalised.iterrows() if not pd.isna(row["path"])]
        if not valid:
            return datasets

        indices, paths = zip(*valid)

        n_jobs = self.n_jobs if hasattr(self, "n_jobs") else 1

        parsed_results = parse_files(list(paths), self.get_complete_parser(), n_jobs=n_jobs)

        updated_indices = []
        for idx, path, parsed in zip(indices, paths, parsed_results):
            if "INVALID_ASSET" in parsed:
                logger.warning(f"Failed to finalise {path}: {parsed.get('TRACEBACK', '')}")
                continue

            for key, value in parsed.items():
                if key in datasets.columns and value is not None:
                    datasets.at[idx, key] = value

            datasets.at[idx, "finalised"] = True
            updated_indices.append(idx)

        if updated_indices:
            # Convert start_time/end_time strings from the complete parser to cftime objects
            mask = datasets.index.isin(updated_indices)
            cal = datasets.loc[mask, "calendar"] if "calendar" in datasets.columns else "standard"
            datasets.loc[mask, "start_time"] = parse_cftime_dates(
                datasets.loc[mask, "start_time"], cal
            ).values
            datasets.loc[mask, "end_time"] = parse_cftime_dates(datasets.loc[mask, "end_time"], cal).values

            # Apply adapter-specific fixes
            datasets = self._post_finalise_fixes(datasets)

        self._persist_finalised_metadata(db, datasets, unfinalised.index)

        return datasets

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
