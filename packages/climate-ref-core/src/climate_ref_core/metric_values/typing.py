from collections.abc import Sequence
from typing import Self

from pydantic import BaseModel, model_validator

Value = float | int


class SeriesDefinition(BaseModel):
    """
    A definition of a 1-d array with an associated index and additional dimensions.
    """

    file_pattern: str
    """A glob pattern to match files that contain the series values."""

    dimensions: dict[str, str]
    """Key, value pairs that identify the dimensions of the metric."""

    values_name: str
    """The name of the variable in the file that contains the values of the series."""

    index_name: str
    """The name of the variable in the file that contains the index of the series."""

    attributes: Sequence[str]
    """A list of attributes that should be extracted from the file and included in the series metadata."""


class SeriesMetricValue(BaseModel):
    """
    A 1-d array with an associated index and additional dimensions

    These values are typically sourced from the CMEC metrics bundle
    """

    dimensions: dict[str, str]
    """
    Key, value pairs that identify the dimensions of the metric

    These values are used for a faceted search of the metric values.
    """
    values: Sequence[Value]
    """
    A 1-d array of values
    """
    index: Sequence[str | Value]
    """
    A 1-d array of index values

    Values must be strings or numbers and have the same length as values.
    Non-unique index values are not allowed.
    """

    index_name: str
    """
    The name of the index.

    This is used for presentation purposes and is not used in the controlled vocabulary.
    """

    attributes: dict[str, str | Value] | None = None
    """
    Additional unstructured attributes associated with the metric value
    """

    @model_validator(mode="after")
    def validate_index_length(self) -> Self:
        """Validate that index has the same length as values"""
        if len(self.index) != len(self.values):
            raise ValueError(
                f"Index length ({len(self.index)}) must match values length ({len(self.values)})"
            )
        return self


class ScalarMetricValue(BaseModel):
    """
    A scalar value with an associated dimensions
    """

    dimensions: dict[str, str]
    """
    Key, value pairs that identify the dimensions of the metric

    These values are used for a faceted search of the metric values.
    """
    value: Value
    """
    A scalar value
    """
    attributes: dict[str, str | Value] | None = None
    """
    Additional unstructured attributes associated with the metric value
    """
