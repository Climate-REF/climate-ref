"""
Mixins for dataset adapters that support lazy finalization.
"""

import pandas as pd

from climate_ref.database import Database


class FinaliseableDatasetAdapterMixin:
    """
    Mixin for dataset adapters that support two-phase ingestion.

    Phase 1 (bootstrap): Fast ingestion from directory/filename metadata only.
    Phase 2 (finalization): Open files to extract full metadata for a subset.
    """

    def finalise_datasets(self, db: Database, datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Finalise a subset of datasets by opening files and extracting full metadata.

        Parameters
        ----------
        db
            Database instance for persisting updated metadata
        datasets
            DataFrame subset containing unfinalised datasets to process

        Returns
        -------
        :
            Updated DataFrame with full metadata extracted from files
        """
        raise NotImplementedError
