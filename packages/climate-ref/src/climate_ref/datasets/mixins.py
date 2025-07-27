import pandas as pd

from climate_ref.database import Database
from climate_ref.models.dataset import Dataset, DatasetFile


class FinaliseableDatasetAdapterMixin:
    """
    Mixin for dataset adapters that support lazy finalisation.
    """

    dataset_cls: type[Dataset]
    slug_column: str

    def finalise_datasets(self, datasets: pd.DataFrame) -> pd.DataFrame:
        """
        Finalise a subset of datasets by applying the complete parser.
        """
        raise NotImplementedError

    def update_catalog(self, db: Database, datasets: pd.DataFrame) -> None:
        """
        Update the data catalog in the database.
        """
        with db.session.begin():
            Dataset = self.dataset_cls

            # Update dataset-specific metadata
            unique_slugs = datasets[self.slug_column].unique()
            model_columns = Dataset.__table__.columns.keys()
            for slug in unique_slugs:
                first_row = datasets[datasets[self.slug_column] == slug].iloc[0]
                update_data = first_row.to_dict()
                update_data_filtered = {
                    k: v for k, v in update_data.items() if k in model_columns and pd.notna(v)
                }
                db.session.query(Dataset).filter(Dataset.slug == slug).update(update_data_filtered)

            # Update file-specific metadata
            for _, row in datasets.iterrows():
                path = row["path"]
                update_data = {
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                }
                db.session.query(DatasetFile).filter(DatasetFile.path == path).update(update_data)
