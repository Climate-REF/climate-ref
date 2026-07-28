"""
JSON wire format for the objects that cross a process boundary.

An executor may run a diagnostic in another process or on another node,
which means the execution definition and its result have to be encoded.

JSON cannot represent everything these objects hold,
so values that have no JSON equivalent are written as tagged objects carrying a `__ref_type__` key.
The types that need this are paths, cftime dates and the pandas frame of selected datasets.
"""

import math
import pathlib
import re
from collections.abc import Callable
from typing import Any

import cftime
import pandas as pd

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult

TAG = "__ref_type__"
"""Key marking an object as a tagged value rather than a plain mapping."""

_ISO = re.compile(
    r"^(-?\d+)-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)?$",
)


def _encode_cftime(value: cftime.datetime) -> dict[str, Any]:
    return {
        TAG: "cftime",
        "value": value.isoformat(),
        "calendar": value.calendar,
        "has_year_zero": value.has_year_zero,
    }


def _decode_cftime(payload: dict[str, Any]) -> cftime.datetime:
    # cftime dates are not always representable by datetime.fromisoformat: calendars admit 30 February
    # and a year zero, so the components are parsed directly.
    match = _ISO.match(payload["value"])
    if match is None:  # pragma: no cover - guards against a corrupted message
        raise ValueError(f"Could not parse cftime value {payload['value']!r}")

    year, month, day, hour, minute, second, fraction = match.groups()
    return cftime.datetime(
        int(year),
        int(month),
        int(day),
        int(hour or 0),
        int(minute or 0),
        int(second or 0),
        int(fraction.ljust(6, "0")[:6]) if fraction else 0,
        calendar=payload["calendar"],
        has_year_zero=payload["has_year_zero"],
    )


def _encode_float(value: float) -> Any:
    """
    Encode a float, tagging the values JSON has no literal for.

    `json.dumps` writes bare `NaN` and `Infinity` by default, which is not JSON.
    Postgres rejects it, and so does any strict parser.
    """
    if math.isnan(value):
        return None
    if math.isinf(value):
        return {TAG: "float", "value": "Infinity" if value > 0 else "-Infinity"}
    return value


def _decode_float(payload: dict[str, Any]) -> float:
    return float(payload["value"])


def _encode_scalar(value: Any) -> Any:
    """
    Encode a single cell of a dataset frame.

    Every flavour of missing value becomes JSON null, including NaN.
    The column's dtype is recorded alongside,
    and pandas restores the sentinel that dtype uses when the column is rebuilt.
    An infinity is a real value rather than a missing one, so it is tagged instead.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, cftime.datetime):
        return _encode_cftime(value)
    if isinstance(value, pathlib.PurePath):
        return {TAG: "path", "value": str(value)}
    if isinstance(value, float):
        return _encode_float(value)
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        # numpy scalars carry a dtype that JSON has no room for
        item = value.item()
        return _encode_float(item) if isinstance(item, float) else item
    return value


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, dict) and TAG in value:
        if value[TAG] == "cftime":
            return _decode_cftime(value)
        if value[TAG] == "path":
            return pathlib.Path(value["value"])
        if value[TAG] == "float":
            return _decode_float(value)
        raise ValueError(f"Unknown tagged value {value[TAG]!r}")
    return value


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        TAG: "dataframe",
        "index": [_encode_scalar(value) for value in frame.index.tolist()],
        "index_name": frame.index.name,
        "index_dtype": str(frame.index.dtype),
        # Column-major keeps each column's dtype with its values, and repeats the column
        # names once rather than once per row.
        "columns": [
            {
                "name": name,
                "dtype": str(frame[name].dtype),
                "values": [_encode_scalar(value) for value in frame[name].tolist()],
            }
            for name in frame.columns
        ],
    }


def _decode_frame(payload: dict[str, Any]) -> pd.DataFrame:
    index = pd.Index(
        [_decode_scalar(value) for value in payload["index"]],
        name=payload["index_name"],
        dtype=payload["index_dtype"],
    )

    frame = pd.DataFrame(index=index)
    for column in payload["columns"]:
        frame[column["name"]] = pd.Series(
            [_decode_scalar(value) for value in column["values"]],
            dtype=column["dtype"],
            index=index,
        )
    return frame


def _encode_collection(collection: DatasetCollection) -> dict[str, Any]:
    return {
        TAG: "dataset_collection",
        "datasets": _encode_frame(collection.datasets),
        "slug_column": collection.slug_column,
        "selector": [list(pair) for pair in collection.selector],
    }


def _decode_collection(payload: dict[str, Any]) -> DatasetCollection:
    return DatasetCollection(
        datasets=_decode_frame(payload["datasets"]),
        slug_column=payload["slug_column"],
        selector=tuple((key, value) for key, value in payload["selector"]),
    )


def _encode_definition(definition: ExecutionDefinition) -> dict[str, Any]:
    return {
        TAG: "execution_definition",
        "diagnostic_full_slug": definition.diagnostic_full_slug,
        "key": definition.key,
        "datasets": {
            source_type.value: _encode_collection(collection)
            for source_type, collection in definition.datasets.items()
        },
        "output_directory": str(definition.output_directory),
        "root_directory": str(definition._root_directory),
    }


def _decode_definition(payload: dict[str, Any]) -> ExecutionDefinition:
    datasets = ExecutionDatasetCollection(
        {
            SourceDatasetType(source_type): _decode_collection(collection)
            for source_type, collection in payload["datasets"].items()
        }
    )
    return ExecutionDefinition(
        diagnostic=None,
        diagnostic_full_slug=payload["diagnostic_full_slug"],
        key=payload["key"],
        datasets=datasets,
        output_directory=pathlib.Path(payload["output_directory"]),
        root_directory=pathlib.Path(payload["root_directory"]),
    )


def _encode_result(result: ExecutionResult) -> dict[str, Any]:
    return {
        TAG: "execution_result",
        "definition": _encode_definition(result.definition),
        "output_bundle_filename": _optional_path(result.output_bundle_filename),
        "metric_bundle_filename": _optional_path(result.metric_bundle_filename),
        "series_filename": _optional_path(result.series_filename),
        "successful": result.successful,
        "retryable": result.retryable,
    }


def _decode_result(payload: dict[str, Any]) -> ExecutionResult:
    return ExecutionResult(
        definition=_decode_definition(payload["definition"]),
        output_bundle_filename=_optional_path_from(payload["output_bundle_filename"]),
        metric_bundle_filename=_optional_path_from(payload["metric_bundle_filename"]),
        series_filename=_optional_path_from(payload["series_filename"]),
        successful=payload["successful"],
        retryable=payload["retryable"],
    )


def _optional_path(value: pathlib.Path | None) -> str | None:
    return None if value is None else str(value)


def _optional_path_from(value: str | None) -> pathlib.Path | None:
    return None if value is None else pathlib.Path(value)


_ENCODERS: list[tuple[type, Any]] = [
    (ExecutionResult, _encode_result),
    (ExecutionDefinition, _encode_definition),
    (DatasetCollection, _encode_collection),
    (pd.DataFrame, _encode_frame),
]


def _decode_path(payload: dict[str, Any]) -> pathlib.Path:
    return pathlib.Path(payload["value"])


_DECODERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "execution_result": _decode_result,
    "execution_definition": _decode_definition,
    "dataset_collection": _decode_collection,
    "dataframe": _decode_frame,
    "cftime": _decode_cftime,
    "path": _decode_path,
    "float": _decode_float,
}


def to_wire(value: Any) -> Any:  # noqa: PLR0911
    """
    Convert a value into its JSON representation

    Parameters
    ----------
    value
        Value to encode. Containers are walked recursively.

    Raises
    ------
    TypeError
        If the value has no JSON representation.

    Returns
    -------
    :
        A structure built only from types that `json` can serialise.
    """
    for kind, encoder in _ENCODERS:
        if isinstance(value, kind):
            return encoder(value)

    if isinstance(value, pathlib.PurePath):
        return {TAG: "path", "value": str(value)}
    if isinstance(value, cftime.datetime):
        return _encode_cftime(value)
    if isinstance(value, SourceDatasetType):
        return value.value
    if isinstance(value, dict):
        return {key: to_wire(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_wire(item) for item in value]
    if isinstance(value, float):
        return _encode_float(value)
    if value is None or isinstance(value, str | bool | int):
        return value

    encoded = _encode_scalar(value)
    if encoded is value:
        raise TypeError(f"Cannot encode {type(value).__name__} as JSON")
    return encoded


def from_wire(value: Any) -> Any:
    """
    Rebuild a value encoded by [to_wire][climate_ref_core.serialisation.to_wire]

    Parameters
    ----------
    value
        JSON structure to decode.

    Raises
    ------
    ValueError
        If the structure carries a tag this version does not understand.

    Returns
    -------
    :
        The decoded value.
    """
    if isinstance(value, dict):
        tag = value.get(TAG)
        if tag is not None:
            if tag not in _DECODERS:
                raise ValueError(f"Unknown tagged value {tag!r}")
            return _DECODERS[tag](value)
        return {key: from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [from_wire(item) for item in value]
    return value
