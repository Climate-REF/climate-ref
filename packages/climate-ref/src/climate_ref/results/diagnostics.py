"""
Find out which diagnostics exist, how their runs are going and which have results.

[DiagnosticsReader][climate_ref.results.diagnostics.DiagnosticsReader] is reached via
[Reader.diagnostics][climate_ref.results.reader.Reader.diagnostics].
It returns:

* [DiagnosticView][climate_ref.results.diagnostics.DiagnosticView]: one diagnostic, with run counts.
* [DiagnosticCollection][climate_ref.results.diagnostics.DiagnosticCollection]:
  a page of diagnostics, with ``to_pandas()``.

Examples
--------
```python
from climate_ref.results import DiagnosticFilter, Reader

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    df = reader.diagnostics.list(DiagnosticFilter(provider_slug="pmp")).to_pandas()
```
"""

from typing import Any

import attrs
import pandas as pd
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.metric_value import MetricValueType
from climate_ref.results._page import Page
from climate_ref.results.filters import DiagnosticFilter
from climate_ref.results.queries import (
    count,
    paginate,
    select_diagnostics,
    select_diagnostics_with_values,
)


@attrs.frozen(kw_only=True)
class DiagnosticView:
    """
    One diagnostic, with its current version and how many of its runs have succeeded.

    Every count covers the groups and executions the filter keeps, by default those at the promoted version.

    Get them from [DiagnosticsReader.list][climate_ref.results.diagnostics.DiagnosticsReader.list].

    Examples
    --------
    >>> from climate_ref.results.diagnostics import DiagnosticView
    >>> diagnostic = DiagnosticView(
    ...     provider_slug="pmp",
    ...     slug="enso_tel",
    ...     name="ENSO teleconnections",
    ...     promoted_version=2,
    ...     execution_group_count=24,
    ...     successful=18,
    ...     inflight=2,
    ...     total=31,
    ... )

    Groups whose newest run failed or that have not run yet, and the reruns on top of one run per group:

    >>> diagnostic.execution_group_count - diagnostic.successful - diagnostic.inflight
    4
    >>> diagnostic.total - diagnostic.execution_group_count
    7
    """

    provider_slug: str
    """Owning provider's slug."""

    slug: str
    """The diagnostic's own slug, unique within its provider."""

    name: str
    """Human-readable diagnostic name."""

    promoted_version: int
    """The promoted version of the diagnostic."""

    execution_group_count: int
    """Execution groups for the diagnostic that the filter keeps."""

    successful: int
    """Groups whose newest execution succeeded."""

    inflight: int
    """Groups whose newest execution is still running."""

    total: int
    """Executions that the filter keeps, counting every rerun of every group."""


@attrs.frozen(kw_only=True)
class DiagnosticCollection(Page[DiagnosticView]):
    """
    A page of diagnostics.

    Iterate it for [DiagnosticView][climate_ref.results.diagnostics.DiagnosticView] items,
    or use ``len()`` for the size of the page.
    ``total_count`` is the number of matching diagnostics before paging.

    Examples
    --------
    An empty page still gives a DataFrame with the right columns:

    >>> from climate_ref.results.diagnostics import DiagnosticCollection
    >>> page = DiagnosticCollection(items=(), total_count=0, offset=0, limit=None)
    >>> list(page.to_pandas().columns[:3])
    ['provider', 'diagnostic', 'name']
    """

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame with the same columns as ``ref diagnostics list``.

        The columns are ``provider``, ``diagnostic``, ``name``, ``promoted_version``,
        ``execution_group_count``, ``successful``, ``inflight`` and ``total``.
        They are present even when the collection is empty.
        """
        columns = [
            "provider",
            "diagnostic",
            "name",
            "promoted_version",
            "execution_group_count",
            "successful",
            "inflight",
            "total",
        ]
        records = [
            {
                "provider": d.provider_slug,
                "diagnostic": d.slug,
                "name": d.name,
                "promoted_version": d.promoted_version,
                "execution_group_count": d.execution_group_count,
                "successful": d.successful,
                "inflight": d.inflight,
                "total": d.total,
            }
            for d in self.items
        ]
        return pd.DataFrame.from_records(records, columns=columns)


class DiagnosticsReader:
    """
    Look up diagnostics, how many of their runs succeeded and which have results.

    Reach it as ``Reader(db).diagnostics``.

    Examples
    --------
    ```python
    from climate_ref.models.metric_value import MetricValueType
    from climate_ref.results import DiagnosticFilter, Reader

    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        page = reader.diagnostics.list(DiagnosticFilter(provider_slug="example"))
        has_series = reader.diagnostics.with_values(MetricValueType.SERIES)
    ```
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The database session the queries run in."""
        return self._session if self._session is not None else self._db.session

    def _to_view(self, row: Any) -> DiagnosticView:
        return DiagnosticView(
            provider_slug=row.provider_slug,
            slug=row.slug,
            name=row.name,
            promoted_version=row.promoted_version,
            execution_group_count=row.execution_group_count,
            successful=row.successful,
            inflight=row.running,
            total=row.execution_count,
        )

    def list(
        self,
        filters: DiagnosticFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        with_count: bool = True,
    ) -> DiagnosticCollection:
        """
        Which diagnostics are there, and how many of their runs succeeded?

        Parameters
        ----------
        filters
            Which diagnostics to return. ``None`` returns all of them.
        offset
            Diagnostics to skip, ordered by provider slug then diagnostic slug.
        limit
            Maximum diagnostics to return. ``None`` returns the rest.
        with_count
            Set ``False`` to skip counting the matches. ``total_count`` is then ``None``.

        Returns
        -------
        DiagnosticCollection
            The page of diagnostics.
            ``total_count`` is the number of matches before ``offset`` and ``limit``.
            A diagnostic with no execution groups that the filter keeps has zero counts.

        Examples
        --------
        The ENSO diagnostics and how their runs are going, as a table:

        ```python
        from climate_ref.results import DiagnosticFilter, Reader

        with Database.from_config(config, read_only=True) as db:
            page = Reader(db).diagnostics.list(DiagnosticFilter(diagnostic_contains=["enso"]))
            df = page.to_pandas()
            print(df[["provider", "diagnostic", "successful", "total"]])
        ```
        """
        filters = filters or DiagnosticFilter()

        base_stmt = select_diagnostics(filters)
        total_count = count(self.session, base_stmt) if with_count else None

        rows = self.session.execute(paginate(base_stmt, offset, limit)).all()
        items = tuple(self._to_view(row) for row in rows)

        return DiagnosticCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def with_values(
        self,
        value_type: MetricValueType,
        filters: DiagnosticFilter | None = None,
    ) -> frozenset[tuple[str, str]]:
        """
        Which diagnostics have results of a given kind?

        Use it to find diagnostics worth querying, for example those with time series
        before calling ``reader.values.series_values``.
        By default only results from each diagnostic's current (promoted) version count,
        and retracted runs do not.

        Parameters
        ----------
        value_type
            ``MetricValueType.SCALAR`` for single numbers, or ``MetricValueType.SERIES`` for time series.
        filters
            Narrow which diagnostics and runs are considered. ``None`` considers all of them.
            Use ``include_retracted=True`` to count retracted runs,
            or ``created_after`` and ``created_before`` to limit by date.

        Returns
        -------
        frozenset[tuple[str, str]]
            ``(provider_slug, diagnostic_slug)`` pairs, such as ``("pmp", "enso_tel")``.

        Examples
        --------
        PMP diagnostics with time series from runs created this year:

        ```python
        import datetime

        from climate_ref.models.metric_value import MetricValueType
        from climate_ref.results import DiagnosticFilter, Reader

        with Database.from_config(config, read_only=True) as db:
            pairs = Reader(db).diagnostics.with_values(
                MetricValueType.SERIES,
                DiagnosticFilter(provider_slug="pmp", created_after=datetime.datetime(2026, 1, 1)),
            )
        # pairs == frozenset({("pmp", "enso_tel"), ...})
        ```

        See [select_diagnostics_with_values][climate_ref.results.queries.select_diagnostics_with_values]
        for the query.
        """
        rows = self.session.execute(select_diagnostics_with_values(value_type, filters)).all()
        return frozenset((row.provider_slug, row.slug) for row in rows)
