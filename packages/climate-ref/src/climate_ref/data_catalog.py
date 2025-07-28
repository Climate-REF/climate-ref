import pandas as pd
from attrs import define
from loguru import logger

from climate_ref.database import Database
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.mixins import FinaliseableDatasetAdapterMixin


@define
class DataCatalog:
    """
    Data catalog for managing datasets in the database.

    This class provides an abstraction layer for interacting with a database-backed data catalog.
    """

    database: Database
    adapter: DatasetAdapter
    _df: pd.DataFrame | None = None

    def finalise(self, subset: pd.DataFrame) -> pd.DataFrame:
        """
        Finalise the datasets in the provided subset.

        This is a no-op if the adapter does not support finalisation.
        """
        if not isinstance(self.adapter, FinaliseableDatasetAdapterMixin):
            return subset

        if "finalised" in subset.columns and not subset["finalised"].all():
            subset_to_finalise = subset[~subset["finalised"]].copy()
            logger.info(f"Finalising {len(subset_to_finalise)} datasets")
            finalised_datasets = self.adapter.finalise_datasets(subset_to_finalise)

            if len(finalised_datasets) < len(subset_to_finalise):
                logger.warning(
                    f"Finalised {len(finalised_datasets)} datasets, but expected {len(subset_to_finalise)}. "
                    "Some datasets may not have been finalised."
                )

            # Merge the finalised datasets back into the original subset/data catalog
            subset.update(finalised_datasets, overwrite=True)
            subset = subset.infer_objects()

            # Update the database with the finalised datasets
            for instance_id, data_catalog_dataset in finalised_datasets.groupby(self.adapter.slug_column):
                logger.debug(f"Processing dataset {instance_id}")
                with self.database.session.begin():
                    self.adapter.register_dataset(self.database, data_catalog_dataset)
            if self._df is not None:
                self._df.update(subset_to_finalise, overwrite=True)
                self._df = self._df.infer_objects()

        return subset

    def to_frame(self) -> pd.DataFrame:
        """
        Load the data catalog into a DataFrame.
        """
        if self._df is None:
            logger.info("Loading data catalog from database")
            self._df = self.adapter.load_catalog(self.database)
        return self._df
