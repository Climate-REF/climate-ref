"""The paging shape shared by every reader collection."""

from collections.abc import Iterator

import attrs


@attrs.frozen(kw_only=True)
class Page[T]:
    """
    An immutable page of items plus the paging that produced it.

    ``total_count`` is the number of rows matching the filter before ``offset``/``limit``.
    It is ``None`` only when the caller passed ``with_count=False``.
    """

    items: tuple[T, ...]
    """The items on this page."""

    total_count: int | None
    """Total rows matching the filter before ``offset``/``limit``, or ``None`` when not counted."""

    offset: int
    """Rows skipped before this page."""

    limit: int | None
    """Page size requested, or ``None`` when the whole result was returned."""

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
