import datetime
import enum
from typing import Any

from sqlalchemy import JSON, Enum
from sqlalchemy.orm import Mapped, mapped_column

from climate_ref.models.base import Base
from climate_ref.models.mixins import CreatedUpdatedMixin


class SolveStatus(enum.Enum):
    """
    Outcome of a solve
    """

    Running = "running"
    """The solve has not finished yet"""

    Succeeded = "succeeded"
    """The solve finished with every provider in scope"""

    Degraded = "degraded"
    """The solve finished, but left out providers that failed their health check"""

    Failed = "failed"
    """The solve stopped early because of an error, or was abandoned"""


class SolveStage(enum.Enum):
    """
    Stage a solve is in, or stopped at
    """

    Preflight = "preflight"
    """Reaping stale solves and executions, and health checking providers"""

    Solving = "solving"
    """Finding the executions to run and recording them"""

    Queuing = "queuing"
    """Finished submitting executions to the executor"""

    Waiting = "waiting"
    """Waiting for the submitted executions to finish"""


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_cls]


class Solve(CreatedUpdatedMixin, Base):
    """
    A single run of the solver that was not a dry run

    Executions and execution groups record the solve that created them.
    Their outcomes are joined from those rows rather than stored here.
    """

    __tablename__ = "solve"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Stored as plain strings rather than native enums, matching ``Execution.failure_reason``.
    status: Mapped[SolveStatus] = mapped_column(
        Enum(SolveStatus, native_enum=False, length=16, values_callable=_enum_values), index=True
    )
    """
    Outcome of the solve
    """

    stage: Mapped[SolveStage] = mapped_column(
        Enum(SolveStage, native_enum=False, length=16, values_callable=_enum_values)
    )
    """
    Stage the solve is in while running, or the stage it stopped at once finished
    """

    started_at: Mapped[datetime.datetime] = mapped_column(index=True)
    """
    When the solve started, as a naive UTC timestamp

    The datasets created or retracted since the previous solve started are the ones this solve saw first.
    """

    finished_at: Mapped[datetime.datetime | None] = mapped_column(nullable=True)
    """
    When the solve finished or failed, or ``None`` while it is running
    """

    error: Mapped[str | None] = mapped_column(nullable=True)
    """
    Why the solve failed, or ``None`` if it did not
    """

    options: Mapped[dict[str, Any]] = mapped_column(default=dict)
    """
    Options the solve was run with, including its filters
    """

    skipped_providers: Mapped[list[str]] = mapped_column(JSON, default=list)
    """
    Slugs of the providers left out because they failed their health check
    """

    ref_version: Mapped[str] = mapped_column()
    """
    Version of climate-ref that ran the solve
    """

    hostname: Mapped[str] = mapped_column()
    """
    Host that ran the solve
    """

    def __repr__(self) -> str:
        return f"<Solve id={self.id} status={self.status.value} stage={self.stage.value}>"
