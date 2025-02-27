import enum
import hashlib
from collections.abc import Iterable
from typing import Any

import pandas as pd
from attrs import field, frozen


class SourceDatasetType(enum.Enum):
    """
    Types of supported source datasets
    """

    CMIP6 = "cmip6"
    CMIP7 = "cmip7"
    obs4MIPs = "obs4mips"


def _clean_facets(raw_values: dict[str, str | tuple[str, ...] | list[str]]) -> dict[str, tuple[str, ...]]:
    """
    Clean the value of a facet filter to a tuple of strings
    """
    result = {}

    for key, value in raw_values.items():
        if isinstance(value, list):
            result[key] = tuple(value)
        elif isinstance(value, str):
            result[key] = (value,)
        elif isinstance(value, tuple):
            result[key] = value
    return result


@frozen
class FacetFilter:
    """
    A filter to apply to a data catalog of datasets.
    """

    facets: dict[str, tuple[str, ...]] = field(converter=_clean_facets)
    """
    Filters to apply to the data catalog.

    The keys are the metadata fields to filter on, and the values are the values to filter on.
    The result will only contain datasets where for all fields,
    the value of the field is one of the given values.
    """
    keep: bool = True
    """
    Whether to keep or remove datasets that match the filter.

    If true (default), datasets that match the filter will be kept else they will be removed.
    """


@frozen
class DatasetCollection:
    """
    Group of datasets required for a given metric execution for a specific source dataset type.
    """

    datasets: pd.DataFrame
    slug_column: str
    """
    Column in datasets that contains the unique identifier for the dataset
    """

    def __getattr__(self, item: str) -> Any:
        return getattr(self.datasets, item)

    def __getitem__(self, item: str | list[str]) -> Any:
        return self.datasets[item]

    def __hash__(self) -> int:
        # This hashes each item individually and sums them so order doesn't matter
        return int(pd.util.hash_pandas_object(self.datasets[self.slug_column]).sum())

    def __eq__(self, other: object) -> bool:
        return self.__hash__() == other.__hash__()


class MetricDataset:
    """
    The complete set of datasets required for a metric execution.

    This may cover multiple source dataset types.
    """

    def __init__(self, collection: dict[SourceDatasetType | str, DatasetCollection]):
        self._collection = {SourceDatasetType(k): v for k, v in collection.items()}

    def __getitem__(self, key: SourceDatasetType | str) -> DatasetCollection:
        if isinstance(key, str):
            key = SourceDatasetType(key)
        return self._collection[key]

    def __hash__(self) -> int:
        return hash(self.hash)

    def items(self) -> Iterable[tuple[SourceDatasetType, DatasetCollection]]:
        """
        Iterate over the datasets in the collection
        """
        return self._collection.items()

    @property
    def hash(self) -> str:
        """
        Unique identifier for the collection

        A SHA1 hash is calculated of the combination of the hashes of the individual collections.
        The value isn't reversible but can be used to uniquely identify the aggregate of the
        collections.

        Returns
        -------
        :
            SHA1 hash of the collections
        """
        # The dataset collection hashes are reproducible,
        # so we can use them to hash the metric dataset.
        # This isn't explicitly true for all Python hashes
        hash_sum = sum(hash(item) for item in self._collection.values())
        hash_bytes = hash_sum.to_bytes(16, "little", signed=True)
        return hashlib.sha1(hash_bytes).hexdigest()  # noqa: S324
