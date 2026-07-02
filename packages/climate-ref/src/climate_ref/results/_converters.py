"""Shared ``attrs`` field converters for the results filter DTOs."""

from collections.abc import Sequence


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
