"""
Dataset management and filtering
"""

import hashlib
from collections.abc import Collection, Iterable, Iterator, Mapping
from typing import Any

import pandas as pd
from attrs import field, frozen

# Import and re-export from source_types for backward compatibility
from climate_ref_core.source_types import Selector, SourceDatasetType

__all__ = ["Selector", "SourceDatasetType"]


def _clean_facets(raw_values: Mapping[str, str | Collection[str]]) -> dict[str, tuple[str, ...]]:
    """
    Clean the value of a facet filter to a tuple of strings
    """
    result: dict[str, tuple[str, ...]] = {}

    for key, value in raw_values.items():
        if isinstance(value, str):
            result[key] = (value,)
        else:
            result[key] = tuple(value)
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


def sort_selector(inp: Selector) -> Selector:
    """
    Sort the selector by key

    Parameters
    ----------
    inp
        Selector to sort

    Returns
    -------
    :
        Sorted selector
    """
    return tuple(sorted(inp, key=lambda x: x[0]))


@frozen
class DatasetCollection:
    """
    Group of datasets required for a given diagnostic execution for a specific source dataset type.
    """

    datasets: pd.DataFrame
    """
    DataFrame containing the datasets that were selected for the execution.

    The columns in this dataframe depend on the source dataset type, but always include:
    * path
    * [slug_column]
    """
    slug_column: str
    """
    Column in datasets that contains the unique identifier for the dataset
    """
    selector: Selector = field(converter=sort_selector, factory=tuple)
    """
    Unique key, value pairs that were selected during the initial groupby
    """

    def selector_dict(self) -> dict[str, str]:
        """
        Convert the selector to a dictionary

        Returns
        -------
        :
            Dictionary of the selector
        """
        return {key: value for key, value in self.selector}

    def __getattr__(self, item: str) -> Any:
        return getattr(self.datasets, item)

    def __getitem__(self, item: str | list[str]) -> Any:
        return self.datasets[item]

    def __hash__(self) -> int:
        # This hashes each item individually and sums them so order doesn't matter
        return int(pd.util.hash_pandas_object(self.datasets[self.slug_column]).sum())

    def __eq__(self, other: object) -> bool:
        return self.__hash__() == other.__hash__()


class ExecutionDatasetCollection:
    """
    The complete set of datasets required for an execution of a diagnostic.

    This may cover multiple source dataset types.
    """

    def __init__(self, collection: dict[SourceDatasetType | str, DatasetCollection]):
        self._collection = {SourceDatasetType(k): v for k, v in collection.items()}

    def __repr__(self) -> str:
        return f"ExecutionDatasetCollection({self._collection})"

    def __contains__(self, key: SourceDatasetType | str) -> bool:
        if isinstance(key, str):
            key = SourceDatasetType(key)
        return key in self._collection

    def __getitem__(self, key: SourceDatasetType | str) -> DatasetCollection:
        if isinstance(key, str):
            key = SourceDatasetType(key)
        return self._collection[key]

    def __hash__(self) -> int:
        return hash(self.hash)

    def __iter__(self) -> Iterator[SourceDatasetType]:
        return iter(self._collection)

    def keys(self) -> Iterable[SourceDatasetType]:
        """
        Iterate over the source types in the collection.
        """
        return self._collection.keys()

    def values(self) -> Iterable[DatasetCollection]:
        """
        Iterate over the datasets in the collection.
        """
        return self._collection.values()

    def items(self) -> Iterable[tuple[SourceDatasetType, DatasetCollection]]:
        """
        Iterate over the items in the collection.
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
        # so we can use them to hash the diagnostic dataset.
        # This isn't explicitly true for all Python hashes
        hash_sum = sum(hash(item) for item in self._collection.values())
        hash_bytes = hash_sum.to_bytes(16, "little", signed=True)
        return hashlib.sha1(hash_bytes).hexdigest()  # noqa: S324

    @property
    def selectors(self) -> dict[str, Selector]:
        """
        Collection of selectors used to identify the datasets

        These are the key, value pairs that were selected during the initial group-by,
        for each data requirement.
        """
        # The "value" of SourceType is used here so this can be stored in the db
        s = {}
        for source_type in SourceDatasetType.ordered():
            if source_type not in self._collection:
                continue
            s[source_type.value] = self._collection[source_type].selector
        return s
