"""
JSON wire format for the objects that cross a process boundary.

An executor may run a diagnostic in another process or on another node,
which means the execution definition and its result have to be encoded.

JSON cannot represent everything these objects hold,
and values with no JSON equivalent are written as tagged objects carrying a `__ref_type__` key.
The types that need this are paths, cftime dates and the pandas frame of selected datasets.
"""

import math
import pathlib
import re
from collections.abc import Callable
from typing import Any

import cattrs
import cftime
import numpy as np
import pandas as pd
from cattrs.gen import make_dict_unstructure_fn, override

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
    # cftime dates are not always representable by datetime.fromisoformat:
    # calendars admit 30 February and a year zero, so the components are parsed directly.
    match = _ISO.match(payload["value"])
    if match is None:  # pragma: no cover, guards against a corrupted message
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


def _encode_path(value: pathlib.PurePath) -> dict[str, Any]:
    return {TAG: "path", "value": str(value)}


def _decode_path(payload: dict[str, Any]) -> pathlib.Path:
    return pathlib.Path(payload["value"])


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
    Encode a scalar, whether it is a cell of a dataset frame or a value on its own.

    Every flavour of missing value becomes JSON null, including NaN.
    A frame column records its dtype alongside,
    so pandas restores the sentinel that dtype uses when the column is rebuilt.
    An infinity is a real value rather than a missing one, so it is tagged instead.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if hasattr(value, "item"):
        # numpy scalars carry a dtype that JSON has no room for
        value = value.item()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, cftime.datetime):
        return _encode_cftime(value)
    if isinstance(value, pathlib.PurePath):
        return _encode_path(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")


_SCALAR_DECODERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "cftime": _decode_cftime,
    "path": _decode_path,
    "float": _decode_float,
}


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, dict) and TAG in value:
        decoder = _SCALAR_DECODERS.get(value[TAG])
        if decoder is None:
            raise ValueError(f"Unknown tagged value {value[TAG]!r}")
        return decoder(value)
    return value


def _encode_values(values: pd.Series | pd.Index, where: str) -> list[Any]:
    """
    Encode a column or index of a dataset frame.

    A numeric numpy dtype cannot hold a taggable value,
    so its cells are emitted directly rather than dispatched one at a time.
    A float column is only eligible when every value is finite.
    """
    dtype = values.dtype
    if isinstance(dtype, np.dtype) and (
        dtype.kind in "iub" or (dtype.kind == "f" and bool(np.isfinite(values.to_numpy()).all()))
    ):
        return values.tolist()
    try:
        return [_encode_scalar(value) for value in values.tolist()]
    except TypeError as exc:
        raise TypeError(f"Cannot encode {where} of the dataset frame: {exc}") from exc


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        TAG: "dataframe",
        "index": _encode_values(frame.index, "the index"),
        "index_name": frame.index.name,
        "index_dtype": str(frame.index.dtype),
        # Column-major keeps each column's dtype with its values,
        # and repeats the column names once rather than once per row.
        "columns": [
            {
                "name": name,
                "dtype": str(column.dtype),
                "values": _encode_values(column, f"column {name!r}"),
            }
            for name, column in frame.items()
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


_converter = cattrs.Converter()

_converter.register_unstructure_hook(pathlib.PurePath, str)
_converter.register_structure_hook(pathlib.Path, lambda value, _: pathlib.Path(value))
_converter.register_unstructure_hook(pd.DataFrame, _encode_frame)
_converter.register_structure_hook(pd.DataFrame, lambda value, _: _decode_frame(value))


def _unstructure_execution_datasets(collection: ExecutionDatasetCollection) -> dict[str, Any]:
    return {
        source_type.value: _converter.unstructure(datasets) for source_type, datasets in collection.items()
    }


def _structure_execution_datasets(payload: dict[str, Any], _: type) -> ExecutionDatasetCollection:
    return ExecutionDatasetCollection(
        {key: _converter.structure(value, DatasetCollection) for key, value in payload.items()}
    )


_converter.register_unstructure_hook(ExecutionDatasetCollection, _unstructure_execution_datasets)
_converter.register_structure_hook(ExecutionDatasetCollection, _structure_execution_datasets)

_unstructure_definition_fields = make_dict_unstructure_fn(
    ExecutionDefinition,
    _converter,
    _diagnostic=override(omit=True),
    _diagnostic_full_slug=override(omit=True),
    _root_directory=override(rename="root_directory"),
)


def _unstructure_definition(definition: ExecutionDefinition) -> dict[str, Any]:
    # The diagnostic is a live object owned by its provider,
    # so only its slug crosses the wire and the receiver resolves it against its own registry.
    payload = _unstructure_definition_fields(definition)
    payload["diagnostic_full_slug"] = definition.diagnostic_full_slug
    return payload


def _structure_definition(payload: dict[str, Any], _: type) -> ExecutionDefinition:
    return ExecutionDefinition(
        diagnostic=None,
        diagnostic_full_slug=payload["diagnostic_full_slug"],
        key=payload["key"],
        datasets=_converter.structure(payload["datasets"], ExecutionDatasetCollection),
        output_directory=_converter.structure(payload["output_directory"], pathlib.Path),
        root_directory=_converter.structure(payload["root_directory"], pathlib.Path),
    )


_converter.register_unstructure_hook(ExecutionDefinition, _unstructure_definition)
_converter.register_structure_hook(ExecutionDefinition, _structure_definition)

_WIRE_TYPES: list[tuple[type, str]] = [
    (ExecutionResult, "execution_result"),
    (ExecutionDefinition, "execution_definition"),
    (DatasetCollection, "dataset_collection"),
]


def _structure_as(kind: type) -> Callable[[dict[str, Any]], Any]:
    def decode(payload: dict[str, Any]) -> Any:
        return _converter.structure(payload, kind)

    return decode


_DECODERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "dataframe": _decode_frame,
    **_SCALAR_DECODERS,
    **{tag: _structure_as(kind) for kind, tag in _WIRE_TYPES},
}


def to_wire(value: Any) -> Any:
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
    for kind, tag in _WIRE_TYPES:
        if isinstance(value, kind):
            return {TAG: tag, **_converter.unstructure(value)}

    if isinstance(value, pd.DataFrame):
        return _encode_frame(value)
    if isinstance(value, SourceDatasetType):
        return value.value
    if isinstance(value, dict):
        return {key: to_wire(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_wire(item) for item in value]

    return _encode_scalar(value)


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
