"""
``attrs`` field converters for the results filters.

Each converter takes one value or a sequence of them and returns a tuple, leaving ``None`` as is.
"""

import datetime
import enum
from collections.abc import Sequence

from climate_ref.models.execution import ExecutionFailureReason
from climate_ref_core.source_types import SourceDatasetType


def _as_member[E: enum.Enum](enum_type: type[E], value: E | str) -> E:
    """Convert a member, its value or its name to a member of ``enum_type``."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        pass
    if isinstance(value, str) and value in enum_type.__members__:
        return enum_type[value]
    choices = ", ".join(repr(m.value) for m in enum_type)
    raise ValueError(f"{value!r} is not a {enum_type.__name__}. Choose from {choices}")


def _as_enum_tuple[E: enum.Enum](
    enum_type: type[E], value: E | str | Sequence[E | str] | None
) -> tuple[E, ...] | None:
    if value is None:
        return None
    if isinstance(value, enum.Enum | str):
        value = (value,)
    return tuple(_as_member(enum_type, v) for v in value)


def _as_failure_reasons(
    value: ExecutionFailureReason | str | Sequence[ExecutionFailureReason | str] | None,
) -> tuple[ExecutionFailureReason, ...] | None:
    """
    Convert one failure reason or several to a tuple.

    Examples
    --------
    >>> _as_failure_reasons("diagnostic_error")
    (<ExecutionFailureReason.DiagnosticError: 'diagnostic_error'>,)
    """
    return _as_enum_tuple(ExecutionFailureReason, value)


def _as_source_types(
    value: SourceDatasetType | str | Sequence[SourceDatasetType | str] | None,
) -> tuple[SourceDatasetType, ...] | None:
    """
    Convert one source type or several to a tuple, accepting a member, its value or its name.

    Examples
    --------
    >>> _as_source_types("cmip6")
    (<SourceDatasetType.CMIP6: 'cmip6'>,)
    >>> _as_source_types("cmip9")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: 'cmip9' is not a SourceDatasetType. Choose from ...
    """
    return _as_enum_tuple(SourceDatasetType, value)


def _as_id_tuple(value: int | Sequence[int] | None) -> tuple[int, ...] | None:
    """
    Convert one id or several to a tuple. An empty list stays empty and matches nothing.

    Raises
    ------
    TypeError
        If an id is not an ``int``, e.g. ``"12"``, which would otherwise be split into ``("1", "2")``.

    Examples
    --------
    >>> _as_id_tuple(3)
    (3,)
    >>> _as_id_tuple("12")
    Traceback (most recent call last):
    ...
    TypeError: Expected ids as ints, not '12'
    """
    if value is None:
        return None
    if isinstance(value, int):
        return (value,)
    ids = tuple(value)
    if not all(isinstance(i, int) for i in ids):
        raise TypeError(f"Expected ids as ints, not {value!r}")
    return ids


def _as_str_tuple(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    """
    Convert one string or several to a tuple.

    Examples
    --------
    >>> _as_str_tuple("enso")
    ('enso',)
    >>> _as_str_tuple(["enso", "annual-cycle"])
    ('enso', 'annual-cycle')
    """
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _as_naive_utc(value: datetime.datetime | None) -> datetime.datetime | None:
    """
    Convert an aware time to naive UTC, the form ``created_at`` is stored in.

    A naive time is kept as is.

    Examples
    --------
    >>> _as_naive_utc(
    ...     datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.timezone(datetime.timedelta(hours=13)))
    ... )
    datetime.datetime(2025, 12, 31, 23, 0)
    """
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(datetime.UTC).replace(tzinfo=None)
