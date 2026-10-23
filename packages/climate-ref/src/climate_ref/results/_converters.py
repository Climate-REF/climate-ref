"""
``attrs`` field converters for the results filters.
"""

from collections.abc import Sequence

from climate_ref.models.execution import ExecutionFailureReason
from climate_ref_core.source_types import SourceDatasetType


def _as_failure_reasons(
    value: Sequence[ExecutionFailureReason] | None,
) -> tuple[ExecutionFailureReason, ...] | None:
    """Convert failure reasons to a tuple, leaving ``None`` as is."""
    return None if value is None else tuple(value)


def _as_source_types(
    value: SourceDatasetType | Sequence[SourceDatasetType] | None,
) -> tuple[SourceDatasetType, ...] | None:
    """Convert a source type or a sequence of them to a tuple, leaving ``None`` as is."""
    if value is None:
        return None
    if isinstance(value, SourceDatasetType):
        return (value,)
    return tuple(value)


def _as_id_tuple(value: Sequence[int] | None) -> tuple[int, ...] | None:
    """Convert ids to a tuple, leaving ``None`` as is. An empty list stays empty and matches nothing."""
    return None if value is None else tuple(value)


def _as_str_tuple(value: Sequence[str] | None) -> tuple[str, ...] | None:
    """
    Convert strings to a tuple, leaving ``None`` as is and rejecting a bare ``str``.

    A ``str`` is also a ``Sequence[str]``, so ``"enso"`` would otherwise be split into
    ``("e", "n", "s", "o")`` and matched one character at a time.

    Raises
    ------
    TypeError
        If ``value`` is a ``str``.

    Examples
    --------
    >>> _as_str_tuple("enso")
    Traceback (most recent call last):
    ...
    TypeError: Expected a sequence of strings, not a bare str. Wrap it in a list/tuple.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError("Expected a sequence of strings, not a bare str. Wrap it in a list/tuple.")
    return tuple(value)
