"""
Typed, detached read surface for solves.

[SolvesReader][climate_ref.results.solves.SolvesReader] is reached via
[Reader.solves][climate_ref.results.values.Reader.solves].
Execution outcomes are joined live through ``Execution.solve_id`` rather than stored on the solve.

A solve's window runs from the start of the previous solve, of any status, up to its own start.
The datasets created or retracted in that window are the ones the solve saw first.
The first solve's window has no start.
"""

import datetime
from collections.abc import Iterator, Mapping
from typing import Any

import attrs
import pandas as pd
from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.orm import InstrumentedAttribute, Session, selectin_polymorphic

from climate_ref.database import Database
from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup, get_execution_group_and_latest
from climate_ref.models.provider import Provider
from climate_ref.models.solve import Solve, SolveStage, SolveStatus
from climate_ref.results._query import count_values, dataset_subtypes
from climate_ref.results.datasets import DatasetCollection, DatasetView
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
class SolveCollection:
    """An immutable page of solves, most recent first."""

    items: tuple[SolveView, ...]
    """The solves on this page."""

    total_count: int
    """Total solves before ``limit``."""

    limit: int | None
    """Page size requested, or ``None`` when the whole result was returned."""

    def __iter__(self) -> Iterator[SolveView]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

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
    Solve read domain.

    Constructed from a [Database][climate_ref.database.Database], which owns the session.
    All read methods return detached DTOs that outlive the session.
    """

    def __init__(self, database: Database, *, session: Session | None = None) -> None:
        self._db = database
        self._session = session

    @property
    def session(self) -> Session:
        """The underlying database session."""
        return self._session if self._session is not None else self._db.session

    def list(self, *, limit: int | None = None) -> SolveCollection:
        """List solves, most recent first."""
        stmt = select(Solve)
        total_count = count_values(self.session, stmt)
        stmt = stmt.order_by(Solve.started_at.desc(), Solve.id.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        items = tuple(SolveView.from_model(s) for s in self.session.execute(stmt).scalars())
        return SolveCollection(items=items, total_count=total_count, limit=limit)

    def get(self, solve_id: int) -> SolveView | None:
        """Fetch one solve by id, or ``None`` when no solve has that id."""
        solve = self.session.get(Solve, solve_id)
        return SolveView.from_model(solve) if solve is not None else None

    def outcomes(self, solve_id: int) -> SolveOutcomes:
        """Count the outcomes of the executions a solve created."""
        rows = self.session.execute(
            select(Execution.successful, Execution.failure_reason, func.count())
            .where(Execution.solve_id == solve_id)
            .group_by(Execution.successful, Execution.failure_reason)
        ).all()
        counts = {True: 0, False: 0, None: 0}
        failure_reasons: dict[str, int] = {}
        for successful, failure_reason, count in rows:
            counts[successful] += count
            if successful is False:
                key = failure_reason or "unknown"
                failure_reasons[key] = failure_reasons.get(key, 0) + count
        return SolveOutcomes(
            succeeded=counts[True],
            failed=counts[False],
            pending=counts[None],
            failure_reasons=failure_reasons,
        )

    def executions(self, solve_id: int, *, limit: int | None = None) -> tuple[SolveExecutionView, ...]:
        """List the executions the solve created, oldest first."""
        return self._execution_views(Execution.solve_id == solve_id, limit=limit)

    def reaped_executions(self, solve_id: int, *, limit: int | None = None) -> tuple[SolveExecutionView, ...]:
        """List the executions the solve's preflight marked as abandoned or never started, oldest first."""
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
        """List the execution groups the solve created, with their latest execution."""
        rows = (
            get_execution_group_and_latest(self.session)
            .filter(ExecutionGroup.created_by_solve_id == solve_id)
            .order_by(ExecutionGroup.id)
        )
        return tuple(ExecutionGroupView.from_model(eg, latest) for eg, latest in rows)

    def window(self, solve_id: int) -> SolveWindow | None:
        """Find the window of a solve, or ``None`` when no solve has that id."""
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
        """List the datasets created in the solve's window, ordered by slug."""
        return self._window_datasets(window or self.window(solve_id), Dataset.created_at, limit=limit)

    def retracted_datasets(
        self, solve_id: int, *, limit: int | None = None, window: SolveWindow | None = None
    ) -> DatasetCollection:
        """List the datasets retracted in the solve's window, ordered by slug."""
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

        total_count = count_values(self.session, stmt)
        stmt = stmt.order_by(Dataset.slug, Dataset.id).options(
            selectin_polymorphic(Dataset, dataset_subtypes())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        items = tuple(
            DatasetView.from_model(d, include_files=False) for d in self.session.execute(stmt).scalars()
        )
        return DatasetCollection(items=items, total_count=total_count, offset=0, limit=limit)
