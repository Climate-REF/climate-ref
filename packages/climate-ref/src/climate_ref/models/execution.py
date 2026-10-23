import datetime
import enum
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from sqlalchemy import (
    BigInteger,
    Column,
    ColumnElement,
    Enum,
    Float,
    ForeignKey,
    Select,
    String,
    Table,
    UniqueConstraint,
    func,
    inspect,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from climate_ref.models.base import Base
from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.mixins import CreatedUpdatedMixin, DimensionMixin
from climate_ref_core.source_types import SourceDatasetType

if TYPE_CHECKING:
    from climate_ref.database import Database
    from climate_ref.models.metric_value import MetricValue
    from climate_ref_core.datasets import ExecutionDatasetCollection
    from climate_ref_core.resources import ResourceUsage


class SkipReason(enum.Enum):
    """
    Why a solve does not run an execution group
    """

    InProgress = "in progress"
    """The latest execution has not finished"""

    FailedNotDirty = "failed and not dirty (use --rerun-failed)"
    """The latest execution failed and no new data has arrived since"""

    UpToDate = "up to date"
    """The latest execution succeeded and no new data has arrived since"""


class ExecutionGroupSourceType(Base):
    """
    A source dataset type an execution group draws on.

    This allows execution groups to be queried by source type using plain SQL.
    """

    __tablename__ = "execution_group_source_type"

    execution_group_id: Mapped[int] = mapped_column(
        ForeignKey("execution_group.id", ondelete="CASCADE"), primary_key=True
    )
    # Non-native so a new source type needs no Postgres enum migration.
    # No index of its own: without table statistics SQLite then reads every row of a type.
    source_type: Mapped[SourceDatasetType] = mapped_column(
        Enum(SourceDatasetType, native_enum=False, create_constraint=False, length=64),
        primary_key=True,
    )

    def __repr__(self) -> str:
        values = inspect(self).dict
        return (
            f"<ExecutionGroupSourceType execution_group_id={values.get('execution_group_id')} "
            f"source_type={values.get('source_type')}>"
        )


class ExecutionGroup(CreatedUpdatedMixin, Base):
    """
    Represents a group of executions with a shared set of input datasets.

    When solving, the `ExecutionGroup`s are derived from the available datasets,
    the defined diagnostics and their data requirements. From the information in the
    group an execution can be triggered, which is an actual run of a diagnostic calculation
    with a specific set of input datasets.

    When the `ExecutionGroup` is created, it is marked dirty, meaning there are no
    current executions available. When an Execution was run successfully for a
    ExecutionGroup, the dirty mark is removed. After ingesting new data and
    solving again and if new versions of the input datasets are available, the
    ExecutionGroup will be marked dirty again.

    The diagnostic_id and key form a unique identifier for `ExecutionGroup`s.
    """

    __tablename__ = "execution_group"
    __table_args__ = (UniqueConstraint("diagnostic_id", "key", "diagnostic_version", name="execution_ident"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    diagnostic_id: Mapped[int] = mapped_column(ForeignKey("diagnostic.id"), index=True)
    """
    The diagnostic that this execution group belongs to
    """

    key: Mapped[str] = mapped_column(index=True)
    """
    Key for the datasets in this Execution group.
    """

    diagnostic_version: Mapped[int] = mapped_column(default=1, server_default="1")
    """
    Diagnostic version that produced this group.

    Read from the live ``Diagnostic.version`` class attribute at solve time.
    Combined with ``diagnostic_id`` and ``key`` to form the unique identifier,
    so v1 and v2 groups for the same key coexist as separate rows.
    """

    dirty: Mapped[bool] = mapped_column(default=False)
    """
    Whether the execution group should be rerun

    An execution group is dirty if the diagnostic or any of the input datasets has been
    updated since the last execution.
    """

    selectors: Mapped[dict[str, Any]] = mapped_column(default=dict)
    """
    Collection of selectors that define the group

    These selectors are the unique key, value pairs that were selected during the initial groupby
    operation.
    These are also used to define the dataset key.
    """

    def __repr__(self) -> str:
        # Read loaded state only, so a detached or expired instance cannot raise while being logged
        values = inspect(self).dict
        return (
            f"<ExecutionGroup id={values.get('id')} diagnostic_id={values.get('diagnostic_id')} "
            f"key={values.get('key')}>"
        )

    diagnostic: Mapped["Diagnostic"] = relationship(back_populates="execution_groups")
    executions: Mapped[list["Execution"]] = relationship(
        back_populates="execution_group", order_by="Execution.created_at, Execution.id"
    )
    source_types: Mapped[list[ExecutionGroupSourceType]] = relationship(cascade="all, delete-orphan")
    """
    Source dataset types this group draws on, kept in step with ``selectors``.
    """

    @validates("selectors")
    def _sync_source_types(self, _key: str, selectors: dict[str, Any]) -> dict[str, Any]:
        wanted = {SourceDatasetType(name) for name in selectors}
        kept = [link for link in self.source_types if link.source_type in wanted]
        missing = wanted - {link.source_type for link in kept}
        added = [ExecutionGroupSourceType(source_type=t) for t in SourceDatasetType.ordered() if t in missing]
        self.source_types = kept + added
        return selectors

    def skip_reason(  # noqa: PLR0911
        self,
        dataset_hash: str,
        rerun_failed: bool = False,
        stale_cutoff: "datetime.datetime | None" = None,
    ) -> SkipReason | None:
        """
        Explain why the diagnostic execution group does not need to be executed.

        The dirty flag is the primary signal for whether an execution group needs to be rerun.
        It is set when the group is created or when new data is available,
        and cleared when an execution completes (whether successful or not).
        Manual intervention (``flag-dirty``, ``fail-running``) can set it back to True.

        The execution group should be run if:

        * no executions have been performed ever
        * the dataset hash is different from the last run
        * the execution group is marked as dirty
        * ``rerun_failed=True`` is passed and the last execution failed

        The execution group should NOT be run if:

        * an execution with the same dataset hash is already in progress
        * the last execution failed and the group is not dirty
          (use ``rerun_failed=True`` or ``flag-dirty`` to retry)

        Parameters
        ----------
        dataset_hash
            Hash of the candidate datasets for this run.
        rerun_failed
            Re-run the group even if the last execution failed and the group is not dirty.
        stale_cutoff
            When provided,
            an in-progress execution created before this timestamp is treated as already failed.
            A real solve reaps such abandoned executions (via ``fail_stale_in_progress_executions``)
            before evaluating this method.

        Returns
        -------
        :
            ``None`` if the group should run, otherwise the reason it should not
        """
        if not self.executions:
            logger.debug(f"Execution group {self.diagnostic.slug}/{self.key} was never executed")
            return None

        last_execution = self.executions[-1]

        if last_execution.dataset_hash != dataset_hash:
            logger.debug(
                f"Execution group {self.diagnostic.slug}/{self.key} hash mismatch:"
                f" {last_execution.dataset_hash} != {dataset_hash}"
            )
            return None

        treat_as_failed = (
            last_execution.successful is None
            and stale_cutoff is not None
            and last_execution.created_at < stale_cutoff
        )

        # Don't submit duplicate tasks for an execution that is already in progress
        # Stuck tasks can be cleaned up with the `fail-running` command
        if last_execution.successful is None and not treat_as_failed:
            logger.debug(
                f"Execution group {self.diagnostic.slug}/{self.key} "
                f"already has an in-progress execution with hash {dataset_hash}"
            )
            return SkipReason.InProgress

        # Dirty flag is the primary signal for rerunning existing jobs
        if self.dirty:
            logger.debug(f"Execution group {self.diagnostic.slug}/{self.key} is dirty")
            return None

        if last_execution.successful is not False and not treat_as_failed:
            return SkipReason.UpToDate

        # Re-run all failed executions if explicitly requested
        if rerun_failed:
            logger.debug(
                f"Execution group {self.diagnostic.slug}/{self.key} "
                f"last execution failed, rerunning (rerun_failed=True)"
            )
            return None
        return SkipReason.FailedNotDirty

    def should_run(
        self,
        dataset_hash: str,
        rerun_failed: bool = False,
        stale_cutoff: "datetime.datetime | None" = None,
    ) -> bool:
        """
        Check if the diagnostic execution group needs to be executed.

        See [skip_reason][climate_ref.models.execution.ExecutionGroup.skip_reason] for the rules.
        """
        return ExecutionGroup.skip_reason(self, dataset_hash, rerun_failed, stale_cutoff) is None


execution_datasets = Table(
    "execution_dataset",
    Base.metadata,
    Column("execution_id", ForeignKey("execution.id"), index=True),
    Column("dataset_id", ForeignKey("dataset.id"), index=True),
)


class ExecutionFailureReason(enum.Enum):
    """
    Why an execution failed
    """

    DiagnosticError = "diagnostic_error"
    """The diagnostic ran and reported an error that a retry with the same inputs would repeat"""

    SystemError = "system_error"
    """The diagnostic could not run to completion for a reason outside it, e.g. a broken environment"""

    ResultError = "result_error"
    """The diagnostic succeeded but its outputs could not be copied or ingested"""

    Abandoned = "abandoned"
    """The execution started but no result was recorded, e.g. its worker was killed"""

    NotStarted = "not_started"
    """The execution was queued but never started, e.g. the solve was killed first"""

    Cancelled = "cancelled"
    """The execution was marked failed by hand"""

    Unknown = "unknown"
    """The caller did not say why the execution failed"""


class Execution(CreatedUpdatedMixin, Base):
    """
    Represents a single execution of a diagnostic

    Each result is part of a group of executions that share similar input datasets.

    An execution group might be run multiple times as new data becomes available,
    each run will create a `Execution`.
    """

    __tablename__ = "execution"

    id: Mapped[int] = mapped_column(primary_key=True)

    output_fragment: Mapped[str] = mapped_column()
    """
    Relative directory to store the output of the execution.

    During execution this directory is relative to the temporary directory.
    If the diagnostic execution is successful, the executions will be moved to the final output directory
    and the temporary directory will be cleaned up.
    This directory may contain multiple input and output files.
    """

    execution_group_id: Mapped[int] = mapped_column(
        ForeignKey(
            "execution_group.id",
            name="fk_execution_id",
        ),
        index=True,
    )
    """
    The execution group that this execution belongs to
    """

    dataset_hash: Mapped[str] = mapped_column(index=True)
    """
    Hash of the datasets used to calculate the diagnostic

    This is used to verify if an existing diagnostic execution has been run with the same datasets.
    """

    successful: Mapped[bool | None] = mapped_column(nullable=True, index=True)
    """
    Was the run successful
    """

    path: Mapped[str] = mapped_column(nullable=True)
    """
    Path to the output bundle

    Relative to the diagnostic execution result output directory
    """

    retracted: Mapped[bool] = mapped_column(default=False)
    """
    Whether the diagnostic execution result has been retracted or not

    This may happen if a dataset has been retracted, or if the diagnostic execution was incorrect.
    Rather than delete the values, they are marked as retracted.
    These data may still be visible in the UI, but should be marked as retracted.
    """

    provider_version: Mapped[str | None] = mapped_column(nullable=True)
    """
    Provider version recorded by the worker at run time.

    Snapshot of the worker-installed ``provider.version`` when the execution ran.
    Purely informational for audit; not used for validation or recomputation triggers.
    Rows that predate the column stay NULL.
    """

    wall_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    """
    Wall clock time taken by the execution, in seconds.

    Measured by the worker as a monotonic delta around the diagnostic run.
    """

    cpu_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    """
    CPU time consumed by the execution, in seconds.

    Sums user and system time for the worker process and its children.
    """

    peak_memory_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """
    Peak resident memory observed during the execution, in bytes.

    Stored as a big integer because a plain integer overflows at 4 GiB on Postgres,
    and executions routinely peak well above that.
    """

    # Plain text rather than an enum, so a release still reads reasons added after it.
    failure_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    """
    Why the execution failed, as an `ExecutionFailureReason` value.

    NULL for successful or in-progress executions.
    Failures recorded before the column existed are ``unknown``.
    """

    memory_source: Mapped[str | None] = mapped_column(nullable=True)
    """
    Provenance of ``peak_memory_bytes``.

    One of ``cgroup``, ``proc_tree``, ``rusage`` or ``unavailable``.
    The sources measure different things and are not interchangeable.
    """

    memory_limit_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    """
    Memory limit in force while the execution ran, in bytes.

    Read from the container cgroup at run time.
    A column rather than a context key because headroom is a comparison against
    ``peak_memory_bytes`` that has to happen in SQL.
    """

    cpu_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    """
    CPU cores available to the execution.

    Read from the container cgroup quota at run time.
    """

    resources_exclusive: Mapped[bool | None] = mapped_column(nullable=True)
    """
    Whether this execution had the worker's cgroup to itself.

    Declared by the executor, which is the only party that knows its own concurrency.
    A cgroup reading covers the whole container,
    so it is meaningless when a container runs several executions at once.
    Non-exclusive measurements are excluded from aggregation by default.
    """

    queue_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    """
    Time between submission and the start of the execution, in seconds.

    Recorded by the submitting executor rather than the worker.
    """

    resource_context: Mapped[dict[str, Any] | None] = mapped_column(nullable=True)
    """
    Supporting detail for the resource measurements.

    Host, CPU count, sampler settings and any cross-checks the worker made.
    Both sampled peaks are kept here, under ``cgroup_peak_bytes`` and ``proc_tree_peak_bytes``,
    whichever of them ``peak_memory_bytes`` reports,
    so which one to trust stays a read-time decision.
    A large gap between the two says the cgroup was shared.
    Purely informational, so nothing here is a query predicate.
    """

    execution_group: Mapped["ExecutionGroup"] = relationship(back_populates="executions")
    outputs: Mapped[list["ExecutionOutput"]] = relationship(back_populates="execution")
    values: Mapped[list["MetricValue"]] = relationship(back_populates="execution")

    datasets: Mapped[list[Dataset]] = relationship(secondary=execution_datasets)
    """
    The datasets used in this execution
    """

    def __repr__(self) -> str:
        # Read loaded state only, so a detached or expired instance cannot raise while being logged
        values = inspect(self).dict
        return (
            f"<Execution id={values.get('id')} group_id={values.get('execution_group_id')} "
            f"output_fragment={values.get('output_fragment')}>"
        )

    def register_datasets(self, db: "Database", execution_dataset: "ExecutionDatasetCollection") -> None:
        """
        Register the datasets used in the diagnostic calculation with the execution
        """
        for _, dataset in execution_dataset.items():
            db.session.execute(
                execution_datasets.insert(),
                [{"execution_id": self.id, "dataset_id": idx} for idx in dataset.index],
            )

    def mark_successful(self, path: pathlib.Path | str) -> None:
        """
        Mark the diagnostic execution as successful
        """
        # TODO: this needs to accept both a diagnostic and output bundle
        self.successful = True
        self.failure_reason = None
        self.path = str(path)

    def apply_resource_usage(
        self, usage: "ResourceUsage | None", *, queue_seconds: float | None = None
    ) -> None:
        """
        Copy a worker's resource measurements onto this row.

        Every value is coerced before any of them is assigned,
        so a value a column cannot store raises here rather than at flush,
        where the offending execution is no longer identifiable.

        Parameters
        ----------
        usage
            The measurements taken by the worker, or None when nothing was measured.
        queue_seconds
            Submit-to-start latency observed by the submitting executor,
            or None when it is unknown.
        """
        if queue_seconds is not None:
            self.queue_seconds = float(queue_seconds)

        if usage is None:
            return

        wall_seconds = float(usage.wall_seconds)
        cpu_seconds = None if usage.cpu_seconds is None else float(usage.cpu_seconds)
        peak_memory_bytes = None if usage.peak_memory_bytes is None else int(usage.peak_memory_bytes)
        memory_source = str(usage.memory_source)
        memory_limit_bytes = None if usage.memory_limit_bytes is None else int(usage.memory_limit_bytes)
        cpu_limit = None if usage.cpu_limit is None else float(usage.cpu_limit)
        exclusive = bool(usage.exclusive)
        context = dict(usage.context)

        self.wall_seconds = wall_seconds
        self.cpu_seconds = cpu_seconds
        self.peak_memory_bytes = peak_memory_bytes
        self.memory_source = memory_source
        self.memory_limit_bytes = memory_limit_bytes
        self.cpu_limit = cpu_limit
        self.resources_exclusive = exclusive
        self.resource_context = context

    def mark_failed(self, reason: ExecutionFailureReason = ExecutionFailureReason.Unknown) -> None:
        """
        Mark the diagnostic execution as unsuccessful

        Parameters
        ----------
        reason
            Why the execution failed
        """
        self.successful = False
        self.failure_reason = reason.value


class ResultOutputType(enum.Enum):
    """
    Types of supported outputs

    These map to the categories of output in the CMEC output bundle
    """

    Plot = "plot"
    Data = "data"
    HTML = "html"


class ExecutionOutput(DimensionMixin, CreatedUpdatedMixin, Base):
    """
    An output generated as part of an execution.

    This output may be a plot, data file or HTML file.
    These outputs are defined in the CMEC output bundle.

    Outputs can be tagged with dimensions from the controlled vocabulary
    to enable filtering and organization.
    """

    __tablename__ = "execution_output"

    _cv_dimensions: ClassVar[list[str]] = []

    id: Mapped[int] = mapped_column(primary_key=True)

    execution_id: Mapped[int] = mapped_column(ForeignKey("execution.id"), index=True)

    output_type: Mapped[ResultOutputType] = mapped_column(index=True)
    """
    Type of the output

    This will determine how the output is displayed
    """

    filename: Mapped[str] = mapped_column(nullable=True)
    """
    Path to the output

    Relative to the diagnostic execution result output directory
    """

    short_name: Mapped[str] = mapped_column(nullable=True)
    """
    Short key of the output

    This is unique for a given result and output type
    """

    long_name: Mapped[str] = mapped_column(nullable=True)
    """
    Human readable name describing the plot
    """

    description: Mapped[str] = mapped_column(nullable=True)
    """
    Long description describing the plot
    """

    execution: Mapped["Execution"] = relationship(back_populates="outputs")

    @classmethod
    def build(  # noqa: PLR0913
        cls,
        *,
        execution_id: int,
        output_type: ResultOutputType,
        dimensions: dict[str, str],
        filename: str | None = None,
        short_name: str | None = None,
        long_name: str | None = None,
        description: str | None = None,
    ) -> "ExecutionOutput":
        """
        Build an ExecutionOutput from dimensions and metadata

        This is a helper method that validates the dimensions supplied.

        Parameters
        ----------
        execution_id
            Execution that created the output
        output_type
            Type of the output
        dimensions
            Dimensions that describe the output
        filename
            Path to the output
        short_name
            Short key of the output
        long_name
            Human readable name
        description
            Long description

        Raises
        ------
        KeyError
            If an unknown dimension was supplied.

            Dimensions must exist in the controlled vocabulary.

        Returns
        -------
            Newly created ExecutionOutput
        """
        for k in dimensions:
            if k not in cls._cv_dimensions:
                raise KeyError(f"Unknown dimension column '{k}'")

        return ExecutionOutput(
            execution_id=execution_id,
            output_type=output_type,
            filename=filename,
            short_name=short_name,
            long_name=long_name,
            description=description,
            **dimensions,
        )


# There are two forms to select for the "latest" execution.
# These selectors are outside of climate_ref.reader as they are used by the models.
# The grouped max is fast across many groups, the ordered limit-1 lookup is fast for one group or a page.
def _select_latest_execution(*where: ColumnElement[bool]) -> Select[tuple[Execution]]:
    """Select the latest ``Execution`` matching ``where``."""
    return select(Execution).where(*where).order_by(Execution.created_at.desc(), Execution.id.desc()).limit(1)


def select_latest_execution(execution_group_id: int) -> Select[tuple[Execution]]:
    """
    Build the ``Select`` for one group's latest ``Execution``, which returns no row if it has none.

    The latest is chosen by ``created_at``, then id, and retracted executions count.
    """
    return _select_latest_execution(Execution.execution_group_id == execution_group_id)


def latest_execution_id(among_executions: Sequence[ColumnElement[bool]] = ()) -> Any:
    """
    Correlated subquery of the outer ``ExecutionGroup``'s latest execution id, or ``NULL`` if it has none.

    ``among_executions`` works as in
    [select_latest_executions][climate_ref.results.queries.select_latest_executions].
    """
    return (
        _select_latest_execution(Execution.execution_group_id == ExecutionGroup.id, *among_executions)
        .with_only_columns(Execution.id)
        .correlate(ExecutionGroup)
        .scalar_subquery()
    )


def select_latest_executions(
    among_executions: Sequence[ColumnElement[bool]] = (),
) -> Select[tuple[int, int]]:
    """
    Build a ``Select`` of the latest execution in each execution group.

    "Latest" is the greatest ``created_at``, with the greatest ``id`` breaking a tie,
    the same ordering as [select_latest_execution][climate_ref.results.queries.select_latest_execution].
    Retracted executions are not skipped, so a retracted latest execution still wins its group.

    Parameters
    ----------
    among_executions
        Predicates on ``Execution`` applied *before* picking the latest,
        so the winner is the newest execution *among those matching*.
        For example ``[Execution.successful.is_(True)]`` picks each group's latest successful execution.
        Group-level filters (diagnostic, provider, dirty, version) belong on the outer query instead,
        where they narrow which groups are returned without changing which execution wins.

    Returns
    -------
        A ``Select`` with ``execution_group_id`` and ``execution_id`` columns,
        one row per group that has at least one matching execution.
        Call ``.subquery()`` to join or filter against it.
    """
    # Grouped max rather than row_number: with a predicate, SQLite 3.40 rescans a filtered window
    # subquery for every outer row when it is left-joined to execution_group.
    newest = (
        select(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("created_at"),
        )
        .where(*among_executions)
        .group_by(Execution.execution_group_id)
        .subquery()
    )
    return (
        select(
            Execution.execution_group_id.label("execution_group_id"),
            func.max(Execution.id).label("execution_id"),
        )
        .join(
            newest,
            (Execution.execution_group_id == newest.c.execution_group_id)
            & (Execution.created_at == newest.c.created_at),
        )
        .where(*among_executions)
        .group_by(Execution.execution_group_id)
    )
