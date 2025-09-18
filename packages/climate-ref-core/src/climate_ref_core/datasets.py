"""
Dataset management and filtering
"""

from __future__ import annotations

import enum
import functools
import hashlib
from collections.abc import Collection, Iterable, Iterator
from typing import Any, Literal, Self

import pandas as pd
from attrs import field, frozen

Selector = tuple[tuple[str, str], ...]
"""
Type describing the key used to identify a group of datasets

This is a tuple of tuples, where each inner tuple contains a metadata and dimension value
that was used to group the datasets together.

This type must be hashable, as it is used as a key in a dictionary.
"""


class SourceDatasetType(enum.Enum):
    """
    Types of supported source datasets
    """

    CMIP6 = "cmip6"
    CMIP7 = "cmip7"
    obs4MIPs = "obs4mips"
    PMPClimatology = "pmp-climatology"

    @classmethod
    @functools.lru_cache(maxsize=1)
    def ordered(
        cls,
    ) -> list[Self]:
        """
        Order in alphabetical order according to their value

        Returns
        -------
        :
            Ordered list of dataset types
        """
        return sorted(cls, key=lambda x: x.value)


def _clean_facets(raw_values: dict[str, str | Collection[str]]) -> dict[str, tuple[str, ...]]:
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
    keep: bool = True
    """
    Whether to keep or remove datasets that match the filter.

    If true (default), datasets that match the filter will be kept else they will be removed.
    """

    prev_filter: tuple[Literal["and", "or"], FacetFilter] | None = None
    """
    Previous filter to apply to the data catalog.

    If provided, this filter will be applied before this filter. This is a tuple
    of the operation to apply ("and" or "or") and the filter to apply.
    """

    next_filter: tuple[Literal["and", "or"], FacetFilter] | None = None
    """
    Next filter to apply to the data catalog.

    If provided, this filter will be applied after this filter. This is a tuple
    of the operation to apply ("and" or "or") and the filter to apply.
    """

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for facet in self.facets:
            if facet not in df.columns:
                raise KeyError(f"Facet {facet!r} not in data catalog columns: {df.columns.to_list()}")

        mask = df[list(self.facets)].isin(self.facets).all(axis="columns")
        if not self.keep:
            mask = ~mask
        return df[mask]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the filter to a dataframe of datasets.

        This applies all operations in sequence.

        Parameters
        ----------
        df
            Dataframe of datasets to filter

        Returns
        -------
        :
            Filtered dataframe of datasets
        """
        for facet in self.facets:
            if facet not in df.columns:
                raise KeyError(f"Facet {facet!r} not in data catalog columns: {df.columns.to_list()}")

        if self.prev_filter:
            op, prev_filt = self.prev_filter
            if op == "and":
                result = self._apply(prev_filt.apply(df))
            elif op == "or":
                result = pd.concat([self._apply(df), prev_filt.apply(df)]).sort_index().drop_duplicates()
            else:
                raise ValueError(f'Unknown operation "{op}"')
        else:
            result = self._apply(df)

        # print(self.facets)
        # print("\n".join(p for p in result.path))
        if self.next_filter:
            op, next_filt = self.next_filter
            if op == "and":
                result = next_filt.apply(result)
            elif op == "or":
                result = pd.concat([result, next_filt.apply(df)]).sort_index().drop_duplicates()
            else:
                raise ValueError(f'Unknown operation "{op}"')
        return result

    def _combine(self, other: object, op: Literal["and", "or"]) -> Self:
        if not isinstance(other, self.__class__):
            raise TypeError(f'Cannot combine {self.__class__.__name__} with "{type(other).__name__}"')
        if other.prev_filter:
            prev_op, prev_filt = other.prev_filter
            prev_filter = (prev_op, prev_filt._combine(self, op))
        else:
            prev_filter = (op, self)
        return self.__class__(
            facets=dict(other.facets.items()),
            keep=other.keep,
            prev_filter=prev_filter,
            next_filter=other.next_filter,
        )

    def __and__(self, other: object) -> Self:
        return self._combine(other, "and")

    def __or__(self, other: object) -> Self:
        return self._combine(other, "or")


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
