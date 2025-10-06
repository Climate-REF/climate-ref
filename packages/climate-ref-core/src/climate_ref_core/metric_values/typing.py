import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

Value = float | int


class SeriesDefinition(BaseModel):
    """
    A definition of a 1-d array with an associated index and additional dimensions.
    """

    file_pattern: str
    """A glob pattern to match files that contain the series values."""

    sel: dict[str, Any] | None = None
    """A dictionary of selection criteria to apply with :meth:`xarray.Dataset.sel` after loading the file."""

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

    attributes: dict[str, str | Value | None] | None = None
    """
    Additional unstructured attributes associated with the metric value
    """

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        """Validate that index has the same length as values and contains no NaNs"""
        if len(self.index) != len(self.values):
            raise ValueError(
                f"Index length ({len(self.index)}) must match values length ({len(self.values)})"
            )
        for v in self.index:
            if isinstance(v, float) and np.isnan(v):
                raise ValueError("NaN values are not allowed in the index")
        return self

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, value: Any) -> Any:
        """
        Transform None values to NaN in the values field
        """
        if not isinstance(value, (list, tuple)):
            raise ValueError("`values` must be a list or tuple.")

        # Transform None values to NaN
        return [float("nan") if v is None else v for v in value]

    @classmethod
    def dump_to_json(cls, path: Path, series: Sequence["SeriesMetricValue"]) -> None:
        """
        Dump a sequence of SeriesMetricValue to a JSON file.

        Parameters
        ----------
        path
            The path to the JSON file.

            The directory containing this file must already exist.
            This file will be overwritten if it already exists.
        series
            The series values to dump.
        """

        def handle_non_finite(value: Any) -> Any:
            """
            Convert NaN values to None for JSON serialization.

            None's translate to null when serialised.
            We can't differentiate nan's from inf here so coercing both to None,
            deserialising will coerce all non-finite values to NaN.
            """
            if isinstance(value, float) and np.isnan(value):
                return None
            return value

        with open(path, "w") as f:
            json.dump(
                [s.model_dump(mode="json") for s in series],
                f,
                indent=2,
                default=handle_non_finite,
            )

    @classmethod
    def load_from_json(
        cls,
        path: Path,
    ) -> list["SeriesMetricValue"]:
        """
        Load a sequence of SeriesMetricValue from a JSON file.

        Parameters
        ----------
        path
            The path to the JSON file.
        """
        with open(path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a list of series values, got {type(data)}")

        return [cls.model_validate(s, strict=True) for s in data]


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
