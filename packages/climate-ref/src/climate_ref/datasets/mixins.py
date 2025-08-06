import pandas as pd

from climate_ref.models.dataset import Dataset


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
