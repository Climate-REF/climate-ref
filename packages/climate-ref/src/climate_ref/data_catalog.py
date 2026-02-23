"""
DataCatalog wrapper for lazy loading and finalisation of dataset catalogs.

This module provides a wrapper around pandas DataFrames that supports:
- Lazy loading of dataset catalogs from the database
- Lazy finalisation of unfinalised datasets at solve time
"""

from __future__ import annotations

import pandas as pd
from attrs import define
from loguru import logger

from climate_ref.database import Database
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.mixins import FinaliseableDatasetAdapterMixin
from climate_ref_core.exceptions import RefException


@define
class DataCatalog:
    """
    Wrapper around a dataset catalog DataFrame that supports lazy loading and lazy finalisation.

    This replaces the raw pd.DataFrame in the solver's data_catalog dict,
    enabling two-phase ingestion where datasets are first bootstrapped from
    DRS metadata and only finalised when needed.

    This lowers the amount of file I/O needed.
    """

    database: Database | None
    adapter: DatasetAdapter | None
    _df: pd.DataFrame | None = None

    @staticmethod
    def from_frame(df: pd.DataFrame) -> DataCatalog:
        """
        Create a DataCatalog from an existing DataFrame, bypassing lazy loading.

        This is useful for testing or when the catalog is already loaded.

        Parameters
        ----------
        df
            The DataFrame to use as the catalog

        Returns
        -------
        :
            A DataCatalog instance with the given DataFrame as its catalog
        """
        return DataCatalog(database=None, adapter=None, df=df)

    def to_frame(self) -> pd.DataFrame:
        """
        Get the catalog as a DataFrame, lazily loading from DB on first access.
        """
        if self._df is None:
            if self.adapter is None or self.database is None:
                raise RefException("Cannot load catalog: adapter and database must be provided")

            self._df = self.adapter.load_catalog(self.database)
        return self._df

    def finalise(self, subset: pd.DataFrame) -> pd.DataFrame:
        """
        Finalise unfinalised datasets in the given subset.

        If the adapter supports finalization (implements FinaliseableDatasetAdapterMixin),
        unfinalised datasets in the subset are finalized by opening their files.
        The internal cache and database are updated accordingly.

        Parameters
        ----------
        subset
            DataFrame subset to finalize (typically after filter+group_by)

        Returns
        -------
        :
            The subset with any unfinalised datasets now finalised
        """
        if not isinstance(self.adapter, FinaliseableDatasetAdapterMixin):
            return subset

        if self.database is None:  # type: ignore[unreachable]
            raise RefException("Cannot finalise datasets: database must be provided")

        has_unfinalised = (
            "finalised" in subset.columns and (subset["finalised"] == False).any()  # noqa: E712
        )
        if not has_unfinalised:
            return subset

        logger.info(
            f"Finalising {(subset['finalised'] == False).sum()} unfinalised datasets"  # noqa: E712
        )
        result = self.adapter.finalise_datasets(self.database, subset)

        # Invalidate the cached DataFrame so the next to_frame() call
        # reloads from DB with correct finalised metadata.
        # In-place cache updates are unreliable because _apply_fixes()
        # can change the DataFrame's index structure.
        # Note: this invalidation does NOT affect the current iteration in
        # extract_covered_datasets (which operates on a local catalog_df copy).
        # It ensures the *next* DataRequirement processed against this
        # DataCatalog gets fresh data from the DB.
        self._df = None

        return result
