"""
DataCatalog wrapper for lazy loading and finalization of dataset catalogs.

This module provides a wrapper around pandas DataFrames that supports:
- Lazy loading of dataset catalogs from the database
- Lazy finalization of unfinalised datasets at solve time
"""

from __future__ import annotations

import pandas as pd
from attrs import define
from loguru import logger

from climate_ref.database import Database
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.mixins import FinaliseableDatasetAdapterMixin


@define
class DataCatalog:
    """
    Wrapper around a dataset catalog DataFrame that supports lazy loading and lazy finalization.

    This replaces the raw pd.DataFrame in the solver's data_catalog dict,
    enabling two-phase ingestion where datasets are first bootstrapped from
    DRS metadata and only finalized (full file I/O) when needed.
    """

    database: Database
    adapter: DatasetAdapter
    _df: pd.DataFrame | None = None

    def to_frame(self) -> pd.DataFrame:
        """
        Get the catalog as a DataFrame, lazily loading from DB on first access.
        """
        if self._df is None:
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

        has_unfinalised = (
            "finalised" in subset.columns and (subset["finalised"] == False).any()  # noqa: E712
        )
        if not has_unfinalised:
            return subset

        logger.info(
            f"Finalising {(subset['finalised'] == False).sum()} unfinalised datasets"  # noqa: E712
        )
        result = self.adapter.finalise_datasets(self.database, subset)

        # Update the cached DataFrame with the finalised data
        if self._df is not None:
            self._df.update(result)

        return result
