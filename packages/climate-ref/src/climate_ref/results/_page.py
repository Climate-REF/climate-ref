"""
The paging shape shared by every reader collection.

An offset and limit approach is used for all collections of results.
"""

from collections.abc import Iterator

import attrs


@attrs.frozen(kw_only=True)
class Page[T]:
    """
    An immutable page of items plus the paging that produced it.

    Every collection that a reader returns is a ``Page``.
    Loop over it, or use ``len()``, to get at its ``items``.

    ``total_count`` is how many results match your filter in all.
    This includes those left out by ``offset`` and ``limit``.
    It is ``None`` only if you passed ``with_count=False``.

    Examples
    --------
    >>> from climate_ref.results._page import Page
    >>> page = Page(items=(10, 20, 30), total_count=7, offset=0, limit=3)
    >>> len(page), list(page)
    (3, [10, 20, 30])
    >>> page.total_count > page.offset + len(page)
    True

    To get the next page, repeat the read with ``offset=page.offset + len(page)``.
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
