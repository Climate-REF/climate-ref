import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

Value = float | int

MetricValueKind = Literal["model", "reference"]
"""The role of a metric value: a model value or a reference (observation) value."""

# Series fields that are omitted from the serialised JSON when left at their default.
# A series that does not set them then serialises identically to one from before these
# fields were added, so existing committed regression baselines stay valid without a
# re-mint; a series that does set them serialises the value.
_OMIT_WHEN_DEFAULT = ("kind", "reference_id", "value_units", "value_long_name", "index_units", "calendar")


class FileDefinition(BaseModel):
    """
    A definition of an output file with associated additional dimensions.
    """

    file_pattern: str
    """A glob pattern to match files that contain the series values."""

    dimensions: dict[str, str]
    """Key, value pairs that identify the dimensions of the metric."""


class SeriesDefinition(FileDefinition):
    """
    A definition of a 1-d array with an associated index and additional dimensions.
    """

    sel: dict[str, Any] | None = None
    """A dictionary of selection criteria to apply with :meth:`xarray.Dataset.sel` after loading the file."""

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
    kind: MetricValueKind = "model"
    """
    Whether the series is a model value or a reference (observation) value.

    This is the first-class signal for the role of the value
    and replaces provider-specific conventions for distinguishing the two.
    """
    reference_id: str | None = None
    """
    Content hash identifying the reference payload of the series.

    Stable across executions for an identical reference payload,
    so reference series can be deduplicated deterministically.
    It is ``None`` for model series and is populated at ingest for reference series.
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

    value_units: str | None = None
    """
    Units of the series values (e.g. ``"K"``).

    Presentation metadata. Optional for now; falls back to ``attributes`` when absent.
    """
    value_long_name: str | None = None
    """
    Human-readable name of the series values (e.g. ``"Near-Surface Air Temperature"``).

    Presentation metadata. Optional for now; falls back to ``attributes`` when absent.
    """
    index_units: str | None = None
    """
    Units of the index (e.g. ``"days since 1850-01-01"``).

    Presentation metadata. Optional for now; falls back to ``attributes`` when absent.
    """
    calendar: str | None = None
    """
    Calendar of a time index (e.g. ``"360_day"``), when the index is temporal.

    Presentation metadata. Optional for now; falls back to ``attributes`` when absent.
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
            if isinstance(v, float) and not np.isfinite(v):
                raise ValueError("NaN or Inf values are not allowed in the index")
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
        # Sort the series before serialising so the order is deterministic across platforms
        # and runs. Diagnostics may emit series in an implementation-defined order (e.g. set or
        # dict iteration that differs by platform), which otherwise produces spurious diffs and
        # breaks the positional regression comparator. ``dimensions`` and ``kind`` together
        # identify a series (``kind`` lives outside ``dimensions``); ``index_name`` is a stable
        # tie-breaker.
        ordered = sorted(
            series,
            key=lambda s: (json.dumps(s.dimensions, sort_keys=True), s.kind, s.index_name),
        )
        with open(path, "w") as f:
            json.dump(
                [s._dump_for_json() for s in ordered],
                f,
                indent=2,
                allow_nan=False,
                sort_keys=True,
            )

    def _dump_for_json(self) -> dict[str, Any]:
        """
        Serialise to a JSON-ready dict, omitting the added fields left at their default.

        See [_OMIT_WHEN_DEFAULT][climate_ref_core.metric_values.typing._OMIT_WHEN_DEFAULT].
        """
        data = self.model_dump(mode="json")
        fields = type(self).model_fields
        for field in _OMIT_WHEN_DEFAULT:
            if data.get(field) == fields[field].default:
                data.pop(field, None)
        return data

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
    kind: MetricValueKind = "model"
    """
    Whether the value is a model value or a reference (observation) value.

    This is the first-class signal for the role of the value
    and replaces provider-specific conventions for distinguishing the two.
    """
    value: Value
    """
    A scalar value
    """
    attributes: dict[str, str | Value] | None = None
    """
    Additional unstructured attributes associated with the metric value
    """
