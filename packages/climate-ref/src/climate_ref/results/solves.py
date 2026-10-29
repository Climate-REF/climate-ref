"""
What did each solve run, and how did it go?

Every ``ref solve`` that is not a dry run is recorded.
Use this to find the executions a solve created, which of them failed,
and which datasets were new or retracted since the previous solve.
[SolvesReader][climate_ref.results.solves.SolvesReader] is reached via
[Reader.solves][climate_ref.results.reader.Reader.solves].
Execution outcomes are counted when read, through ``Execution.solve_id``, rather than stored on the solve.

A solve's window runs from the start of the previous solve, of any status, up to its own start.
The datasets created or retracted in that window are the ones the solve saw first.
The first solve's window has no start.

It returns:

* [SolveCollection][climate_ref.results.solves.SolveCollection]: a list of solves,
  with ``to_pandas()`` for a table.
* [SolveView][climate_ref.results.solves.SolveView]: one solve.
* [SolveOutcomes][climate_ref.results.solves.SolveOutcomes]: how many of a solve's executions
  succeeded, failed or are still pending.
* [SolveExecutionView][climate_ref.results.solves.SolveExecutionView]: one execution a solve created,
  with its diagnostic and group.
* [SolveWindow][climate_ref.results.solves.SolveWindow]: the span of dataset changes a solve saw first.

Examples
--------
The most recent solve and why its executions failed:

```python
from climate_ref.results import Reader

with Database.from_config(config, read_only=True) as db:
    reader = Reader(db)
    latest = reader.solves.list(limit=1).items[0]
    outcomes = reader.solves.outcomes(latest.id)
    print(latest.status, outcomes.failed, dict(outcomes.failure_reasons))
```
"""

import datetime
from collections.abc import Mapping
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.orm import InstrumentedAttribute, Session, joinedload, selectin_polymorphic

from climate_ref.database import Database
from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.provider import Provider
from climate_ref.models.solve import Solve, SolveStage, SolveStatus
from climate_ref.query import count, latest_execution_id, paginate
from climate_ref.results._page import Page, check_paging
from climate_ref.results.datasets import DatasetCollection, DatasetView, dataset_subtypes
from climate_ref.results.executions import ExecutionGroupView, ExecutionView


@attrs.frozen(kw_only=True)
class SolveView:
    """A single solve, detached from the ORM."""

    id: int
    """Primary key of the underlying ``Solve`` row."""

    status: SolveStatus
    """Outcome of the solve, ``running`` until it finishes."""

    stage: SolveStage
    """Stage the solve is in, or the stage it stopped at."""

    started_at: datetime.datetime
    """When the solve started, as a naive UTC timestamp."""

    finished_at: datetime.datetime | None
    """When the solve finished or failed, or ``None`` while it is running."""

    error: str | None
    """Why the solve failed, or ``None``."""

    options: Mapping[str, Any]
    """Options the solve was run with, including its filters."""

    skipped_providers: tuple[str, ...]
    """Providers left out because they failed their health check."""

    ref_version: str
    """Version of climate-ref that ran the solve."""

    hostname: str
    """Host that ran the solve."""

    @classmethod
    def from_model(cls, solve: Solve) -> "SolveView":
        """Build a view of a ``Solve`` row."""
        return cls(
            id=solve.id,
            status=solve.status,
            stage=solve.stage,
            started_at=solve.started_at,
            finished_at=solve.finished_at,
            error=solve.error,
            options=dict(solve.options),
            skipped_providers=tuple(solve.skipped_providers),
            ref_version=solve.ref_version,
            hostname=solve.hostname,
        )


@attrs.frozen(kw_only=True)
class SolveCollection(Page[SolveView]):
    """
    The solves that ``reader.solves.list`` found, most recent first.

    Loop over it, or use ``len()``, to get the [SolveView][climate_ref.results.solves.SolveView]
    objects on this page.
    Call ``to_pandas()`` to get a table.
    ``total_count`` is the number of solves before ``offset`` and ``limit``.
    """

    def to_pandas(self) -> pd.DataFrame:
        """DataFrame with one row per solve, with the columns emitted even when the collection is empty."""
        columns = [
            "id",
            "status",
            "stage",
            "started_at",
            "finished_at",
            "skipped_providers",
            "error",
            "hostname",
            "ref_version",
        ]
        records = [
            {
                "id": s.id,
                "status": s.status.value,
                "stage": s.stage.value,
                "started_at": s.started_at,
                "finished_at": s.finished_at,
                "skipped_providers": list(s.skipped_providers),
                "error": s.error,
                "hostname": s.hostname,
                "ref_version": s.ref_version,
            }
            for s in self.items
        ]
        return pd.DataFrame.from_records(records, columns=columns)


@attrs.frozen(kw_only=True)
class SolveOutcomes:
    """Outcomes of the executions a solve created, counted when read."""

    succeeded: int
    """Executions that succeeded."""

    failed: int
    """Executions that failed."""

    pending: int
    """Executions with no result yet."""

    failure_reasons: Mapping[str, int]
    """Failed executions by ``ExecutionFailureReason`` value, with ``unknown`` for those without one."""

    @property
    def total(self) -> int:
        """Every execution the solve created."""
        return self.succeeded + self.failed + self.pending


@attrs.frozen(kw_only=True)
class SolveExecutionView:
    """An execution with the diagnostic and group it ran for."""

    execution: ExecutionView
    """The execution itself."""

    provider_slug: str
    """Slug of the provider the diagnostic belongs to."""

    diagnostic_slug: str
    """Slug of the diagnostic that was executed."""

    group_key: str
    """Key of the execution group, which identifies the datasets selected."""


@attrs.frozen(kw_only=True)
class SolveWindow:
    """The span of dataset changes a solve saw first."""

    start: datetime.datetime | None
    """Start of the previous solve, or ``None`` for the first solve."""

    end: datetime.datetime
    """Start of this solve, which is excluded from the window."""


class SolvesReader:
    """
    Which solves have run, what did they create and which datasets did they see first?

    Reach it as ``Reader(db).solves``.
    Use ``list`` to page through solves and ``get`` to fetch one by id.
    The other methods take a solve id and read what that solve created or saw.
    The results can be used after the database is closed.

    Examples
    --------
    ```python
    with Database.from_config(config, read_only=True) as db:
        reader = Reader(db)
        solve = reader.solves.get(12)
        failed = [e for e in reader.solves.executions(12) if e.execution.successful is False]
        new = reader.solves.new_datasets(12, limit=20).to_pandas()
    ```
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._session if self._session is not None else self._db.session

    def list(
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        with_count: bool = True,
    ) -> SolveCollection:
        """
        Which solves have run?

        Parameters
        ----------
        offset
            Solves to skip, most recent first.
        limit
            Maximum solves to return. ``None`` returns the rest.
        with_count
            Set ``False`` to skip counting the solves. ``total_count`` is then ``None``.

        Returns
        -------
        SolveCollection
            The page of solves, most recent first.
            ``total_count`` is the number of solves before ``offset`` and ``limit``.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            df = Reader(db).solves.list(limit=20).to_pandas()
        ```
        """
        check_paging(offset, limit)
        stmt = select(Solve)
        total_count = count(self.session, stmt) if with_count else None
        stmt = paginate(stmt.order_by(Solve.started_at.desc(), Solve.id.desc()), offset, limit)
        items = tuple(SolveView.from_model(s) for s in self.session.execute(stmt).scalars())
        return SolveCollection(items=items, total_count=total_count, offset=offset, limit=limit)

    def get(self, solve_id: int) -> SolveView | None:
        """
        Fetch one solve by id.

        Parameters
        ----------
        solve_id
            Primary key of the solve.

        Returns
        -------
        SolveView or None
            The solve, or ``None`` if no solve has that id.
        """
        solve = self.session.get(Solve, solve_id)
        return SolveView.from_model(solve) if solve is not None else None

    def outcomes(self, solve_id: int) -> SolveOutcomes:
        """
        How many of the executions a solve created succeeded, failed or are still pending?

        Parameters
        ----------
        solve_id
            Primary key of the solve.

        Returns
        -------
        SolveOutcomes
            The counts, with failures broken down by reason.
            Every count is zero when no solve has that id.
        """
        rows = self.session.execute(
            select(Execution.successful, Execution.failure_reason, func.count())
            .where(Execution.solve_id == solve_id)
            .group_by(Execution.successful, Execution.failure_reason)
        ).all()
        counts = {True: 0, False: 0, None: 0}
        failure_reasons: dict[str, int] = {}
        for successful, failure_reason, n in rows:
            counts[successful] += n
            if successful is False:
                key = failure_reason or "unknown"
                failure_reasons[key] = failure_reasons.get(key, 0) + n
        return SolveOutcomes(
            succeeded=counts[True],
            failed=counts[False],
            pending=counts[None],
            failure_reasons=failure_reasons,
        )

    def executions(self, solve_id: int, *, limit: int | None = None) -> tuple[SolveExecutionView, ...]:
        """
        Which executions did the solve create?

        Parameters
        ----------
        solve_id
            Primary key of the solve.
        limit
            Maximum executions to return. ``None`` returns them all.

        Returns
        -------
        tuple of SolveExecutionView
            The executions, oldest first.
            Empty when no solve has that id.
        """
        return self._execution_views(Execution.solve_id == solve_id, limit=limit)

    def reaped_executions(self, solve_id: int, *, limit: int | None = None) -> tuple[SolveExecutionView, ...]:
        """
        Which executions did the solve's preflight mark as abandoned or never started?

        These were created by earlier solves, not by this one.

        Parameters
        ----------
        solve_id
            Primary key of the solve.
        limit
            Maximum executions to return. ``None`` returns them all.

        Returns
        -------
        tuple of SolveExecutionView
            The executions, oldest first.
            Empty when no solve has that id.
        """
        return self._execution_views(Execution.reaped_by_solve_id == solve_id, limit=limit)

    def _execution_views(
        self, condition: ColumnElement[bool], *, limit: int | None
    ) -> tuple[SolveExecutionView, ...]:
        stmt = (
            select(Execution, ExecutionGroup.key, Diagnostic.slug, Provider.slug)
            .join(Execution.execution_group)
            .join(ExecutionGroup.diagnostic)
            .join(Diagnostic.provider)
            .where(condition)
            .order_by(Execution.created_at, Execution.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return tuple(
            SolveExecutionView(
                execution=ExecutionView.from_model(execution),
                provider_slug=provider_slug,
                diagnostic_slug=diagnostic_slug,
                group_key=group_key,
            )
            for execution, group_key, diagnostic_slug, provider_slug in self.session.execute(stmt)
        )

    def groups(self, solve_id: int) -> tuple[ExecutionGroupView, ...]:
        """
        Which execution groups did the solve create?

        Groups that already existed are left out, even if the solve ran them again.

        Parameters
        ----------
        solve_id
            Primary key of the solve.

        Returns
        -------
        tuple of ExecutionGroupView
            The groups, ordered by id, each with its latest execution.
            Empty when no solve has that id.
        """
        stmt = (
            select(ExecutionGroup, Execution)
            .options(joinedload(ExecutionGroup.diagnostic).joinedload(Diagnostic.provider))
            .outerjoin(Execution, Execution.id == latest_execution_id())
            .where(ExecutionGroup.created_by_solve_id == solve_id)
            .order_by(ExecutionGroup.id)
        )
        return tuple(ExecutionGroupView.from_model(eg, latest) for eg, latest in self.session.execute(stmt))

    def window(self, solve_id: int) -> SolveWindow | None:
        """
        When does the span of dataset changes a solve saw first start and end?

        Pass the result to ``new_datasets`` and ``retracted_datasets`` to avoid looking it up twice.

        Parameters
        ----------
        solve_id
            Primary key of the solve.

        Returns
        -------
        SolveWindow or None
            The window, or ``None`` if no solve has that id.
        """
        end = self.session.execute(select(Solve.started_at).where(Solve.id == solve_id)).scalar_one_or_none()
        if end is None:
            return None
        start = self.session.execute(
            select(func.max(Solve.started_at)).where(Solve.started_at < end)
        ).scalar_one()
        return SolveWindow(start=start, end=end)

    def new_datasets(
        self, solve_id: int, *, limit: int | None = None, window: SolveWindow | None = None
    ) -> DatasetCollection:
        """
        Which datasets were ingested since the previous solve?

        The first solve counts every dataset as new.

        Parameters
        ----------
        solve_id
            Primary key of the solve.
        limit
            Maximum datasets to return. ``None`` returns them all.
        window
            The solve's window, if you already have it from ``window``.
            ``None`` looks it up.

        Returns
        -------
        DatasetCollection
            The datasets, ordered by slug, without their files.
            ``total_count`` is the number of datasets before ``limit``.
            Empty when no solve has that id.

        Examples
        --------
        ```python
        with Database.from_config(config, read_only=True) as db:
            solves = Reader(db).solves
            window = solves.window(12)
            new = solves.new_datasets(12, window=window)
            retracted = solves.retracted_datasets(12, window=window)
        ```
        """
        return self._window_datasets(window or self.window(solve_id), Dataset.created_at, limit=limit)

    def retracted_datasets(
        self, solve_id: int, *, limit: int | None = None, window: SolveWindow | None = None
    ) -> DatasetCollection:
        """
        Which datasets were retracted since the previous solve?

        Parameters
        ----------
        solve_id
            Primary key of the solve.
        limit
            Maximum datasets to return. ``None`` returns them all.
        window
            The solve's window, if you already have it from ``window``.
            ``None`` looks it up.

        Returns
        -------
        DatasetCollection
            The datasets, ordered by slug, without their files.
            ``total_count`` is the number of datasets before ``limit``.
            Empty when no solve has that id.
        """
        return self._window_datasets(window or self.window(solve_id), Dataset.retracted_at, limit=limit)

    def _window_datasets(
        self,
        window: SolveWindow | None,
        column: InstrumentedAttribute[Any],
        *,
        limit: int | None,
    ) -> DatasetCollection:
        if window is None:
            return DatasetCollection(items=(), total_count=0, offset=0, limit=limit)

        stmt: Select[Any] = select(Dataset).where(column < window.end)
        if window.start is not None:
            stmt = stmt.where(column >= window.start)

        total_count = count(self.session, stmt)
        stmt = stmt.order_by(Dataset.slug, Dataset.id).options(
            selectin_polymorphic(Dataset, dataset_subtypes())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        items = tuple(
            DatasetView.from_model(d, include_files=False) for d in self.session.execute(stmt).scalars()
        )
        return DatasetCollection(items=items, total_count=total_count, offset=0, limit=limit)
