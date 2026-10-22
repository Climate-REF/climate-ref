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
    Float,
    ForeignKey,
    Select,
    String,
    Table,
    UniqueConstraint,
    func,
    inspect,
    or_,
    select,
)
from sqlalchemy.orm import Mapped, Session, joinedload, mapped_column, relationship
from sqlalchemy.orm.query import Query, RowReturningQuery

from climate_ref.models.base import Base
from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.mixins import CreatedUpdatedMixin, DimensionMixin
from climate_ref.models.provider import Provider

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


def select_latest_executions(
    among_executions: Sequence[ColumnElement[bool]] = (),
) -> Select[tuple[int, int]]:
    """
    Build a ``Select`` of the latest execution in each execution group.

    "Latest" is the greatest ``created_at``, with the greatest ``id`` breaking a tie,
    the same ordering as [latest_execution_for_group][climate_ref.results._query.latest_execution_for_group].
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


def _latest_execution_id(among_executions: Sequence[Any] | None = None) -> Any:
    """
    Correlated scalar subquery of the outer ``ExecutionGroup``'s latest execution id, or ``NULL`` if none.

    It picks the same execution as
    [select_latest_executions][climate_ref.models.execution.select_latest_executions],
    and ``among_executions`` means the same there.
    """
    return (
        select(Execution.id)
        .where(Execution.execution_group_id == ExecutionGroup.id, *(among_executions or ()))
        .order_by(Execution.created_at.desc(), Execution.id.desc())
        .limit(1)
        .correlate(ExecutionGroup)
        .scalar_subquery()
    )


def _successful_predicate(successful: bool) -> Any:
    """
    Build a predicate on ``Execution.successful``.

    ``True`` matches successful executions.
    ``False`` matches unsuccessful *or* in-progress (NULL) executions and,
    when applied to an outer-joined ``Execution``, also matches groups with no execution at all (NULL row).

    Shared by the pre-rank population filter and the post-rank ``successful`` filter.
    """
    if successful:
        return Execution.successful.is_(True)
    return or_(Execution.successful.is_(False), Execution.successful.is_(None))


def get_execution_group_and_latest(
    session: Session,
    among_executions: Sequence[Any] | None = None,
) -> RowReturningQuery[tuple[ExecutionGroup, Execution | None]]:
    """
    Query to get the most recent result for each execution group

    Parameters
    ----------
    session
        The database session to use for the query.
    among_executions
        Optional predicates on ``Execution`` applied *before* the latest-per-group ranking,
        so "latest" is chosen from that filtered population (see :func:`_latest_execution_id`).
        Defaults to ranking over all executions.

    Returns
    -------
        Query to get the most recent result for each execution group.
        The result is a tuple of the execution group and the most recent result,
        which can be None.
    """
    # Groups with no matching execution still appear (outer join, Execution=None).
    # Every known caller (`cli/executions.py::delete_groups`, `ExecutionsReader.groups()`,
    # `executor/reingest.py`) reads `eg.diagnostic.slug`/`eg.diagnostic.provider.slug` off the
    # returned groups, so eager-load both relationships here to avoid an N+1 lazy-load per row.
    query = (
        session.query(ExecutionGroup, Execution)
        .options(joinedload(ExecutionGroup.diagnostic).joinedload(Diagnostic.provider))
        .outerjoin(Execution, Execution.id == _latest_execution_id(among_executions))
    )

    return query  # type: ignore


def _selectors_match_facet(
    selectors: dict[str, list[list[str]]],
    facet_key: str,
    facet_values: list[str],
) -> bool:
    """
    Check if an execution group's selectors match a single facet filter.

    Parameters
    ----------
    selectors
        The execution group's selectors dict (dataset_type -> list of [key, value] pairs)
    facet_key
        Facet key, optionally prefixed with ``dataset_type.`` to scope to one type
    facet_values
        Allowed values (OR logic -- any match is sufficient)
    """
    if "." in facet_key:
        dataset_type, key = facet_key.split(".", 1)
        if dataset_type in selectors:
            return any([key, fv] in selectors[dataset_type] for fv in facet_values)
        return False

    # Bare key: search across all dataset types
    for ds_type_selectors in selectors.values():
        if any([facet_key, fv] in ds_type_selectors for fv in facet_values):
            return True
    return False


def _filter_executions_by_selectors(
    results: Sequence[tuple[ExecutionGroup, Execution | None]],
    selector_filters: dict[str, list[str]],
) -> list[tuple[ExecutionGroup, Execution | None]]:
    """
    Filter execution groups by facet key-value pairs in their ``selectors``.

    This is a relatively expensive operation as it requires iterating over all results.
    This should be replaced once we have normalised the selectors into a separate table.

    Parameters
    ----------
    results
        List of tuples containing ExecutionGroup and its latest Execution (or None)
    selector_filters
        Dictionary mapping facet keys to lists of allowed values.
        Different keys are ANDed; multiple values for the same key are ORed.

    Returns
    -------
        Filtered list of tuples containing ExecutionGroup and its latest Execution (or None)

    Notes
    -----
    - Facet filters can either be key=value (searches all dataset types)
      or dataset_type.key=value (searches specific dataset type)
    - Key=value filters search across all dataset types
    - dataset_type.key=value filters only search within the specified dataset type
    - Multiple values for the same key use OR logic
    - All specified keys must match for an execution group to be included (AND logic)
    """
    return [
        (eg, execution)
        for eg, execution in results
        if all(
            _selectors_match_facet(eg.selectors, facet_key, facet_values)
            for facet_key, facet_values in selector_filters.items()
        )
    ]


def query_execution_group_and_latest_filtered(  # noqa: PLR0913
    session: Session,
    diagnostic_filters: list[str] | None = None,
    provider_filters: list[str] | None = None,
    dirty: bool | None = None,
    successful: bool | None = None,
    latest_successful: bool | None = None,
    include_superseded: bool = False,
    failure_reasons: Sequence[ExecutionFailureReason] | None = None,
    group_conditions: Sequence[ColumnElement[bool]] | None = None,
    latest_conditions: Sequence[ColumnElement[bool]] | None = None,
) -> RowReturningQuery[tuple[ExecutionGroup, Execution | None]]:
    """
    Build the unmaterialised query behind ``get_execution_group_and_latest_filtered``, ordered by group id.

    Every filter except ``selector_filters`` is applied in SQL,
    so callers can add pagination before running it.
    The parameters match that function.
    """
    # Pre-rank population filter: restrict which executions "latest" is chosen from.
    among_executions = None if latest_successful is None else [_successful_predicate(latest_successful)]

    query = get_execution_group_and_latest(session, among_executions=among_executions)
    query = _filter_execution_groups(
        query, diagnostic_filters, provider_filters, dirty, include_superseded, group_conditions
    )

    if successful is not None:
        query = query.filter(_successful_predicate(successful))

    if failure_reasons:
        query = query.filter(Execution.failure_reason.in_([reason.value for reason in failure_reasons]))

    for condition in latest_conditions or ():
        query = query.filter(condition)

    return query.order_by(ExecutionGroup.id)


def count_execution_group_and_latest_filtered(  # noqa: PLR0913
    session: Session,
    diagnostic_filters: list[str] | None = None,
    provider_filters: list[str] | None = None,
    dirty: bool | None = None,
    successful: bool | None = None,
    latest_successful: bool | None = None,
    include_superseded: bool = False,
    failure_reasons: Sequence[ExecutionFailureReason] | None = None,
    group_conditions: Sequence[ColumnElement[bool]] | None = None,
    latest_conditions: Sequence[ColumnElement[bool]] | None = None,
) -> int:
    """
    Count the rows ``query_execution_group_and_latest_filtered`` returns.

    Each group yields exactly one row, so the latest execution is only resolved
    when a filter on it (``successful``, ``failure_reasons`` or ``latest_conditions``) can drop the group.
    """
    if successful is not None or failure_reasons or latest_conditions:
        query = query_execution_group_and_latest_filtered(
            session,
            diagnostic_filters=diagnostic_filters,
            provider_filters=provider_filters,
            dirty=dirty,
            successful=successful,
            latest_successful=latest_successful,
            include_superseded=include_superseded,
            failure_reasons=failure_reasons,
            group_conditions=group_conditions,
            latest_conditions=latest_conditions,
        )
        return query.order_by(None).count()
    groups = _filter_execution_groups(
        session.query(func.count(ExecutionGroup.id)).select_from(ExecutionGroup),
        diagnostic_filters,
        provider_filters,
        dirty,
        include_superseded,
        group_conditions,
    )
    return groups.scalar() or 0


def _filter_execution_groups[GroupQuery: Query[Any]](  # noqa: PLR0913
    query: GroupQuery,
    diagnostic_filters: list[str] | None,
    provider_filters: list[str] | None,
    dirty: bool | None,
    include_superseded: bool,
    group_conditions: Sequence[ColumnElement[bool]] | None,
) -> GroupQuery:
    """Apply the group-level filters, which never depend on which execution is latest."""
    if not include_superseded:
        query = query.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
        query = query.filter(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)

    # Matched against diagnostics first, so SQLite reaches the groups by index rather than scanning them.
    if diagnostic_filters or provider_filters:
        diagnostics = select(Diagnostic.id)
        # Apply diagnostic filter (OR logic for multiple values)
        if diagnostic_filters:
            diagnostic_conditions = [
                Diagnostic.slug.ilike(f"%{filter_value.lower()}%") for filter_value in diagnostic_filters
            ]
            diagnostics = diagnostics.where(or_(*diagnostic_conditions))

        # Apply provider filter (OR logic for multiple values)
        if provider_filters:
            provider_conditions = [
                Provider.slug.ilike(f"%{filter_value.lower()}%") for filter_value in provider_filters
            ]
            diagnostics = diagnostics.join(Provider, Diagnostic.provider_id == Provider.id).where(
                or_(*provider_conditions)
            )
        query = query.filter(ExecutionGroup.diagnostic_id.in_(diagnostics))

    if dirty is not None:
        if dirty:
            query = query.filter(ExecutionGroup.dirty.is_(True))
        else:
            query = query.filter(or_(ExecutionGroup.dirty.is_(False), ExecutionGroup.dirty.is_(None)))

    for condition in group_conditions or ():
        query = query.filter(condition)

    return query


def get_execution_group_and_latest_filtered(  # noqa: PLR0913
    session: Session,
    diagnostic_filters: list[str] | None = None,
    provider_filters: list[str] | None = None,
    selector_filters: dict[str, list[str]] | None = None,
    dirty: bool | None = None,
    successful: bool | None = None,
    latest_successful: bool | None = None,
    include_superseded: bool = False,
    failure_reasons: Sequence[ExecutionFailureReason] | None = None,
    group_conditions: Sequence[ColumnElement[bool]] | None = None,
    latest_conditions: Sequence[ColumnElement[bool]] | None = None,
) -> list[tuple[ExecutionGroup, Execution | None]]:
    """
    Query execution groups with filtering capabilities.

    New callers should build an ``ExecutionGroupFilter`` and use ``ExecutionsReader.group_rows`` instead.

    By default, returns only execution groups whose ``diagnostic_version`` matches
    the parent diagnostic's ``promoted_version`` so consumers see exactly one
    version's worth of results.
    Pass ``include_superseded=True`` to bypass the version filter and see the full history.

    Success can be filtered in two different ways: ``successful`` and ``latest_successful``.
    ``successful=True`` keeps a group only if its newest run happened to succeed,
    whereas ``latest_successful=True`` changes which run is chosen as newest.
    The two compose but answer different questions.

    Parameters
    ----------
    session
        Database session
    diagnostic_filters
        List of diagnostic slug substrings (OR logic, case-insensitive)
    provider_filters
        List of provider slug substrings (OR logic, case-insensitive)
    selector_filters
        Dictionary mapping facet keys to lists of allowed values.
        Different keys are ANDed; multiple values for the same key are ORed.
        These match each group's ``ExecutionGroup.selectors`` in Python.
        To match the datasets the latest execution ran against in SQL, pass an
        ``InputDatasetFilter`` clause through ``latest_conditions`` instead.
    dirty
        If True, only return dirty execution groups.
        If False, only return clean execution groups.
        If None, do not filter by dirty status.
    successful
        Post-rank filter on the *winning* execution -- asks "is the latest execution successful?".
        If True, only return execution groups whose latest execution was successful.
        If False, only return execution groups whose latest execution was unsuccessful or has no executions.
        If None, do not filter by execution success.
    latest_successful
        Pre-rank population filter -- asks "what is the latest *successful* execution?".
        If True, rank only over successful executions,
        so the returned execution is each group's latest successful run (if any).
        If False, rank only over unsuccessful / in-progress executions.
        If None (default), rank over all executions.
    include_superseded
        If True, include execution groups for diagnostic versions older than the
        currently promoted version.
        If False (default), join ``Diagnostic`` and filter to ``ExecutionGroup.diagnostic_version
        == Diagnostic.promoted_version``.
        Set this for recovery / audit callers that need the full version history
        (e.g. ``executor/reingest.py``).
    failure_reasons
        Post-rank filter on the *winning* execution.
        If provided, only return execution groups whose latest execution failed for one of these reasons.
        If None, do not filter by failure reason.
    group_conditions
        Extra SQL conditions over ``ExecutionGroup``, ANDed with the other filters.
    latest_conditions
        Extra SQL conditions over the latest ``Execution`` (as ranked under ``latest_successful``),
        ANDed with the other filters. A group with no execution never matches one.
        For example, to keep only groups whose latest execution ran against ACCESS-ESM1-5 CMIP6 data:

        ```python
        scope = InputDatasetFilter(
            source_types=(SourceDatasetType.CMIP6,),
            facets={"source_id": ("ACCESS-ESM1-5",)},
        )
        get_execution_group_and_latest_filtered(session, latest_conditions=[scope.execution_clause()])
        ```

    Returns
    -------
        List of (ExecutionGroup, latest Execution or None) tuples, ordered by group id

    Notes
    -----
    - Diagnostic and provider filters use substring matching (case-insensitive)
    - Multiple values within same filter type use OR logic
    - Different filter types use AND logic
    - Selector filters can either be key=value (searches all dataset types)
      or dataset_type.key=value (searches specific dataset type)
    - This helper is the only sanctioned path for new callers that should respect the promoted-version filter.
      The one acknowledged exception is the ``cli/executions.py::stats`` aggregation,
      which inlines
      ``.join(Diagnostic).filter(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)``
      because it returns aggregate rows rather than a list of tuples and so cannot reuse this helper.
      Operational queries that must remain version-agnostic
      (e.g. ``mark_failed_running`` in the same module) intentionally do not use this helper at all.
    """
    query = query_execution_group_and_latest_filtered(
        session,
        diagnostic_filters=diagnostic_filters,
        provider_filters=provider_filters,
        dirty=dirty,
        successful=successful,
        latest_successful=latest_successful,
        include_superseded=include_superseded,
        failure_reasons=failure_reasons,
        group_conditions=group_conditions,
        latest_conditions=latest_conditions,
    )

    if selector_filters:
        # Load all results into memory for Python-based filtering
        # TODO: Update once we have normalised the selector
        results = [r._tuple() for r in query.all()]
        return _filter_executions_by_selectors(results, selector_filters)
    else:
        return [r._tuple() for r in query.all()]
