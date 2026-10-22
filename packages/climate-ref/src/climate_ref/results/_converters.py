"""Shared ``attrs`` field converters for the results filter DTOs."""

from collections.abc import Sequence

from climate_ref.models.execution import ExecutionFailureReason
from climate_ref_core.source_types import SourceDatasetType


def _as_failure_reasons(
    value: Sequence[ExecutionFailureReason] | None,
) -> tuple[ExecutionFailureReason, ...] | None:
    """Coerce a multi-value failure reason filter to a tuple, leaving ``None`` as is."""
    return None if value is None else tuple(value)


def _as_source_types(value: Sequence[SourceDatasetType]) -> tuple[SourceDatasetType, ...]:
    """Coerce the dataset types of an input dataset filter to a tuple."""
    return tuple(value)


def _as_id_tuple(value: Sequence[int] | None) -> tuple[int, ...] | None:
    """Coerce an id filter to a tuple, leaving ``None`` as is."""
    return None if value is None else tuple(value)


def _as_str_tuple(value: Sequence[str] | None) -> tuple[str, ...] | None:
    """
    Coerce a multi-value filter field to ``tuple[str, ...]``, rejecting a bare ``str``.

    A bare ``str`` is itself a ``Sequence[str]``, so without this guard a caller passing
    ``diagnostic_contains="enso"`` would have it iterated character-by-character
    (``"enso"`` -> ``e``, ``n``, ``s``, ``o``) before it ever reaches ``ilike`` /
    the facet matcher.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError("Expected a sequence of strings, not a bare str. Wrap it in a list/tuple.")
    return tuple(value)
