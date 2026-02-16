"""
Lightweight types for dataset source identification.

This module contains core type definitions that are used across the codebase
but don't require heavy dependencies like pandas. Keeping these in a separate
module allows other modules to import them without triggering heavy imports.
"""

import enum
import functools
from typing import Self

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
        Order in alphabetical order according to their value.

        Returns
        -------
        :
            Ordered list of dataset types
        """
        return sorted(cls, key=lambda x: x.value)


__all__ = ["Selector", "SourceDatasetType"]
