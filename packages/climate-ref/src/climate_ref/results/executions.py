"""
Find out which diagnostic executions happened, which failed and why, and what files they produced.

[ExecutionsReader][climate_ref.results.executions.ExecutionsReader] is reached via
[Reader.executions][climate_ref.results.reader.Reader.executions].

It returns:

* [ExecutionGroupView][climate_ref.results.executions.ExecutionGroupView]:
  a group, with its newest execution in ``latest``.
* [ExecutionView][climate_ref.results.executions.ExecutionView]: one execution.
* [OutputView][climate_ref.results.executions.OutputView]: one file or plot an execution produced.
* [ExecutionGroupCollection][climate_ref.results.executions.ExecutionGroupCollection]:
  a page of groups, with ``to_pandas()``.

A group's newest execution is the one created last.
Every method and count here uses that same definition.

Code that changes groups, such as deleting or requeuing them, uses
[group_rows][climate_ref.results.executions.ExecutionsReader.group_rows].
To read results, use [groups][climate_ref.results.executions.ExecutionsReader.groups].

Examples
--------
Which executions of the ENSO diagnostic failed, and why?

```python
from climate_ref.results import ExecutionGroupFilter, Reader

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    failing = reader.executions.groups(
        ExecutionGroupFilter(provider_slug="pmp", diagnostic_slug="enso_tel", successful=False)
    )
    print(failing.to_pandas()[["id", "failure_reason", "selectors"]])
```
"""

from collections.abc import Mapping
from typing import Any

import attrs
import pandas as pd
from sqlalchemy.orm import Session

from climate_ref.database import Database
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
)
from climate_ref.query import (
    ExecutionGroupFilter,
    count,
    filter_by_selectors,
    paginate,
    select_execution_group_ids,
    select_execution_groups,
    select_execution_outputs,
    select_latest_execution,
)
from climate_ref.results._page import Page


@attrs.frozen(kw_only=True)
class ExecutionView:
    """
    One execution of a diagnostic.

    Get one from [ExecutionsReader.execution][climate_ref.results.executions.ExecutionsReader.execution],
    or from ``latest`` on an [ExecutionGroupView][climate_ref.results.executions.ExecutionGroupView].

    Examples
    --------
    >>> import datetime
    >>> from climate_ref.results.executions import ExecutionView
    >>> now = datetime.datetime(2026, 3, 1)
    >>> execution = ExecutionView(
    ...     id=42,
    ...     execution_group_id=7,
    ...     successful=False,
    ...     failure_reason="diagnostic_error",
    ...     dataset_hash="abc123",
    ...     retracted=False,
    ...     output_fragment="pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42",
    ...     path=None,
    ...     provider_version="v1.0",
    ...     created_at=now,
    ...     updated_at=now,
    ... )
    >>> execution.successful
    False
    """

    id: int
    """Primary key of the underlying ``Execution`` row."""

    execution_group_id: int
    """The execution group this execution belongs to."""

    successful: bool | None
    """``True`` or ``False`` once the execution has finished, ``None`` while it is still running."""

    failure_reason: str | None = None
    """
    Why a failed execution failed, or ``None``.

    This is the value of a [ExecutionFailureReason][climate_ref.models.execution.ExecutionFailureReason].
    """

    dataset_hash: str
    """Hash of the input datasets used for this execution."""

    retracted: bool
    """Whether this execution has been retracted."""

    output_fragment: str
    """
    Directory holding this execution's outputs, relative to the results root.

    Pass it to ``reader.artifacts`` to get a path on disk.
    """

    path: str | None
    """Path to the output bundle, relative to ``output_fragment``, or ``None`` if there is none."""

    provider_version: str | None
    """Version of the diagnostic provider that produced this execution."""

    created_at: Any
    """Timestamp the execution was created."""

    updated_at: Any
    """Timestamp the execution was last updated."""


@attrs.frozen(kw_only=True)
class ExecutionGroupView:
    """
    One diagnostic applied to one set of input datasets, with its newest execution if it has one.

    For example, the ENSO diagnostic run on ACCESS-CM2 historical data is one group.
    Get groups from [ExecutionsReader.groups][climate_ref.results.executions.ExecutionsReader.groups]
    or [ExecutionsReader.group][climate_ref.results.executions.ExecutionsReader.group].
    ``successful`` and ``failure_reason`` report on the newest execution.

    Examples
    --------
    An execution group that has never been run has no newest execution:

    >>> import datetime
    >>> from climate_ref.results.executions import ExecutionGroupView
    >>> group = ExecutionGroupView(
    ...     id=7,
    ...     key="cmip6_ACCESS-CM2_r1i1p1f1",
    ...     diagnostic_slug="enso_tel",
    ...     provider_slug="pmp",
    ...     dirty=False,
    ...     diagnostic_version=1,
    ...     selectors={"cmip6": [["source_id", "ACCESS-CM2"]]},
    ...     created_at=datetime.datetime(2026, 3, 1),
    ...     updated_at=datetime.datetime(2026, 3, 1),
    ...     latest=None,
    ... )
    >>> group.successful is None
    True
    """

    id: int
    """Primary key of the underlying ``ExecutionGroup`` row."""

    key: str
    """Stable key identifying the group's input-dataset selection."""

    diagnostic_slug: str
    """Owning diagnostic's slug."""

    provider_slug: str
    """Owning provider's slug."""

    dirty: bool
    """Whether the group's inputs have changed, so it needs to be run again."""

    diagnostic_version: int
    """Diagnostic version this group was executed against."""

    selectors: Mapping[str, Any]
    """The facet values that selected this group's input datasets, keyed by dataset type."""

    created_at: Any
    """Timestamp the group was created."""

    updated_at: Any
    """Timestamp the group was last updated."""

    latest: ExecutionView | None
    """
    The group's newest execution, or ``None`` if it has never been run.

    This is the execution that
    [ExecutionsReader.latest_execution][climate_ref.results.executions.ExecutionsReader.latest_execution]
    returns.
    With ``latest_successful=True`` on the filter it is the newest successful execution instead.
    """

    @property
    def successful(self) -> bool | None:
        """Whether the newest execution succeeded, or ``None`` if it is still running or there is none."""
        return self.latest.successful if self.latest else None

    @property
    def failure_reason(self) -> str | None:
        """Why the newest execution failed, or ``None``. See ``ExecutionView.failure_reason``."""
        return self.latest.failure_reason if self.latest else None


@attrs.frozen(kw_only=True)
class OutputView:
    """
    A file an execution produced, such as a plot or a data file.

    Get them from [ExecutionsReader.outputs][climate_ref.results.executions.ExecutionsReader.outputs].
    Join ``filename`` onto the execution's ``output_fragment`` to find the file.

    Examples
    --------
    >>> from climate_ref.results.executions import OutputView
    >>> output = OutputView(
    ...     execution_id=42,
    ...     output_type="plot",
    ...     filename="plots/enso_tel.png",
    ...     short_name="enso_tel_plot",
    ...     long_name="ENSO teleconnection",
    ...     description=None,
    ...     dimensions={"source_id": "ACCESS-CM2"},
    ... )
    >>> output.output_type
    'plot'
    """

    execution_id: int
    """The execution that registered this output."""

    output_type: str
    """The kind of output, such as ``"plot"``, ``"data"`` or ``"html"``."""

    filename: str | None
    """Path to the output, relative to the execution's ``output_fragment``."""

    short_name: str | None
    """Short key of the output, unique for a given execution and output type."""

    long_name: str | None
    """Human-readable name for the output."""

    description: str | None
    """Free-text description of the output."""

    dimensions: Mapping[str, str]
    """CV dimension values for this output, such as ``{"source_id": "ACCESS-CM2"}``."""


@attrs.frozen(kw_only=True)
class ExecutionGroupCollection(Page[ExecutionGroupView]):
    """
    A page of execution groups.

    Iterate it for [ExecutionGroupView][climate_ref.results.executions.ExecutionGroupView] items,
    call ``to_pandas()`` for a table, or use ``len()`` for the size of the page.
    ``total_count`` is the number of matching groups before paging.

    Examples
    --------
    An empty page still gives a DataFrame with the right columns:

    >>> from climate_ref.results.executions import ExecutionGroupCollection
    >>> page = ExecutionGroupCollection(items=(), total_count=0, offset=0, limit=None)
    >>> len(page)
    0
    >>> list(page.to_pandas().columns[:4])
    ['id', 'key', 'provider', 'diagnostic']
    """

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to a DataFrame with the same columns as ``ref executions list-groups``.

        The columns are ``id``, ``key``, ``provider``, ``diagnostic``, ``dirty``, ``successful``,
        ``failure_reason``, ``created_at``, ``updated_at`` and ``selectors``.
        They are present even when the collection is empty.
        """
        columns = [
            "id",
            "key",
            "provider",
            "diagnostic",
            "dirty",
            "successful",
            "failure_reason",
            "created_at",
            "updated_at",
            "selectors",
        ]
        records = [
            {
                "id": eg.id,
                "key": eg.key,
                "provider": eg.provider_slug,
                "diagnostic": eg.diagnostic_slug,
                "dirty": eg.dirty,
                "successful": eg.successful,
                "failure_reason": eg.failure_reason,
                "created_at": eg.created_at,
                "updated_at": eg.updated_at,
                "selectors": dict(eg.selectors),
            }
            for eg in self.items
        ]
        return pd.DataFrame.from_records(records, columns=columns)


class ExecutionsReader:
    """
    Look up executions, their outcomes and the files they produced.

    Reach it as ``Reader(db).executions``.
    See the [module overview][climate_ref.results.executions] for what an execution and a group are.

    Examples
    --------
    ```python
    from climate_ref.results import ExecutionGroupFilter, Reader

    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)

        # Groups whose newest execution failed, 20 at a time
        page = reader.executions.groups(ExecutionGroupFilter(successful=False), limit=20)
        print(page.total_count)

        # One group, its newest execution and the files that execution produced
        group = reader.executions.group(page.items[0].id)
        for output in reader.executions.outputs(group.latest.id):
            print(output.output_type, output.filename)
    ```
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The database session the queries run in."""
        return self._session if self._session is not None else self._db.session

    def _to_execution_view(self, execution: Execution) -> ExecutionView:
        return ExecutionView(
            id=execution.id,
            execution_group_id=execution.execution_group_id,
            successful=execution.successful,
            failure_reason=execution.failure_reason,
            dataset_hash=execution.dataset_hash,
            retracted=execution.retracted,
            output_fragment=execution.output_fragment,
            path=execution.path,
            provider_version=execution.provider_version,
            created_at=execution.created_at,
            updated_at=execution.updated_at,
        )

    def _to_group_view(self, eg: ExecutionGroup, latest: Execution | None) -> ExecutionGroupView:
        return ExecutionGroupView(
            id=eg.id,
            key=eg.key,
            diagnostic_slug=eg.diagnostic.slug,
            provider_slug=eg.diagnostic.provider.slug,
            dirty=eg.dirty,
            diagnostic_version=eg.diagnostic_version,
            selectors=dict(eg.selectors),
            created_at=eg.created_at,
            updated_at=eg.updated_at,
            latest=self._to_execution_view(latest) if latest is not None else None,
        )

    def group_rows(
        self, filters: ExecutionGroupFilter | None = None
    ) -> list[tuple[ExecutionGroup, Execution | None]]:
        """
        Get matching groups and their newest executions as database rows that code can change.

        This is for code that deletes or requeues groups.
        To read results, use [groups][climate_ref.results.executions.ExecutionsReader.groups] instead.

        Parameters
        ----------
        filters
            Which groups to return. ``None`` returns all of them.

        Returns
        -------
        list[tuple[ExecutionGroup, Execution | None]]
            One ``(group, newest execution)`` pair per group, ordered by group id.
            The execution is ``None`` for a group that has never been run.
            The rows are only valid while the database is open.

        Examples
        --------
        ```python
        from climate_ref.results import ExecutionGroupFilter, Reader

        with Database.from_config(config) as db:
            for group, latest in Reader(db).executions.group_rows(ExecutionGroupFilter(dirty=True)):
                print(group.id, group.diagnostic.slug, latest.successful if latest else None)
        ```
        """
        filters = filters or ExecutionGroupFilter()
        stmt = select_execution_groups(attrs.evolve(filters, selectors=None))
        rows = list(self.session.execute(stmt).tuples().all())
        return filter_by_selectors(rows, filters.selectors) if filters.selectors else rows

    def groups(
        self,
        filters: ExecutionGroupFilter | None = None,
        *,
        offset: int = 0,
        limit: int | None = None,
        with_count: bool = True,
    ) -> ExecutionGroupCollection:
        """
        Which execution groups match, and how did their newest execution go?

        Each group comes with its newest execution, so you can see whether it succeeded and why not.
        The results are plain objects that stay usable after the database is closed.
        Code that changes groups uses
        [group_rows][climate_ref.results.executions.ExecutionsReader.group_rows].

        ``successful`` and ``latest_successful`` ask different questions.
        Say a model's execution failed last week but succeeded the week before.
        With ``successful=False`` the group matches, because its newest execution failed.
        With ``latest_successful=True`` the group matches too, but its ``latest`` is now
        the execution from the week before, because that is the newest one that succeeded.
        See [ExecutionGroupFilter][climate_ref.query.filters.ExecutionGroupFilter].

        Parameters
        ----------
        filters
            Which groups to return. ``None`` returns all of them at their promoted diagnostic version.
        offset
            Groups to skip, in id order.
        limit
            Maximum groups to return. ``None`` returns the rest.
        with_count
            Set ``False`` to skip counting the matches. ``total_count`` is then ``None``.

        Returns
        -------
        ExecutionGroupCollection
            The page of groups, ordered by group id so paging is stable.
            ``total_count`` is the number of matches before ``offset`` and ``limit``.

        Examples
        --------
        The second page of PMP groups whose newest execution failed inside the diagnostic:

        ```python
        from climate_ref.models.execution import ExecutionFailureReason
        from climate_ref.results import ExecutionGroupFilter, Reader

        filters = ExecutionGroupFilter(
            provider_slug="pmp",
            successful=False,
            failure_reasons=[ExecutionFailureReason.DiagnosticError],
        )
        with Database.from_config(config, read_only=True) as db:
            page = Reader(db).executions.groups(filters, offset=20, limit=20)
            print(page.total_count, len(page))
            df = page.to_pandas()
        ```
        """
        filters = filters or ExecutionGroupFilter()
        if filters.selectors:
            all_results = self.group_rows(filters)
            total_count = len(all_results) if with_count else None
            page = all_results[offset : offset + limit] if limit is not None else all_results[offset:]
        else:
            total_count = count(self.session, select_execution_group_ids(filters)) if with_count else None
            stmt = paginate(select_execution_groups(filters), offset, limit)
            page = list(self.session.execute(stmt).tuples().all())
        items = tuple(self._to_group_view(eg, latest) for eg, latest in page)

        return ExecutionGroupCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def latest_execution(self, execution_group_id: int) -> ExecutionView | None:
        """
        Find the newest execution of this group, whether or not it succeeded.

        The newest execution is the one created last.
        To get the newest successful execution instead, filter ``groups`` with ``latest_successful=True``.

        Returns
        -------
        ExecutionView | None
            ``None`` if the group has never been run or does not exist.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            latest = Reader(db).executions.latest_execution(7)
            if latest is not None and latest.successful is False:
                print(latest.failure_reason)
        ```
        """
        execution = self.session.execute(select_latest_execution(execution_group_id)).scalar_one_or_none()
        return self._to_execution_view(execution) if execution is not None else None

    def group(self, execution_group_id: int) -> ExecutionGroupView | None:
        """
        Look up one execution group by id, with its newest execution.

        Returns
        -------
        ExecutionGroupView | None
            ``None`` when no group has that id.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            group = Reader(db).executions.group(7)
            if group is not None:
                print(group.provider_slug, group.diagnostic_slug, group.successful)
        ```
        """
        eg = self.session.get(ExecutionGroup, execution_group_id)
        if eg is None:
            return None
        latest = self.session.execute(select_latest_execution(execution_group_id)).scalar_one_or_none()
        return self._to_group_view(eg, latest)

    def execution(self, execution_id: int) -> ExecutionView | None:
        """
        Look up one execution by id, or ``None`` when no execution has that id.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            execution = Reader(db).executions.execution(42)
            print(execution.output_fragment)
        ```
        """
        execution = self.session.get(Execution, execution_id)
        return self._to_execution_view(execution) if execution is not None else None

    def outputs(self, execution_id: int) -> tuple[OutputView, ...]:
        """
        Which plots and data files did an execution produce?

        The outputs are ordered by type, then id.
        Pass ``output_fragment`` and ``filename`` to ``reader.artifacts.output_file`` to get a path on disk.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            for output in Reader(db).executions.outputs(42):
                print(output.output_type, output.short_name, output.filename)
        ```
        """
        stmt = select_execution_outputs(execution_id)
        rows = self.session.execute(stmt).scalars().all()
        return tuple(
            OutputView(
                execution_id=row.execution_id,
                output_type=row.output_type.value,
                filename=row.filename,
                short_name=row.short_name,
                long_name=row.long_name,
                description=row.description,
                dimensions=dict(row.dimensions),
            )
            for row in rows
        )
