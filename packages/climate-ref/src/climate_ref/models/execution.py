import datetime
import enum
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from sqlalchemy import Column, ForeignKey, Table, UniqueConstraint, func, or_
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from sqlalchemy.orm.query import RowReturningQuery

from climate_ref.models.base import Base
from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.mixins import CreatedUpdatedMixin, DimensionMixin
from climate_ref.models.provider import Provider

if TYPE_CHECKING:
    from climate_ref.database import Database
    from climate_ref.models.metric_value import MetricValue
    from climate_ref_core.datasets import ExecutionDatasetCollection


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

    diagnostic: Mapped["Diagnostic"] = relationship(back_populates="execution_groups")
    executions: Mapped[list["Execution"]] = relationship(
        back_populates="execution_group", order_by="Execution.created_at"
    )

    def should_run(
        self,
        dataset_hash: str,
        rerun_failed: bool = False,
        stale_cutoff: "datetime.datetime | None" = None,
    ) -> bool:
        """
        Check if the diagnostic execution group needs to be executed.

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
        """
        if not self.executions:
            logger.debug(f"Execution group {self.diagnostic.slug}/{self.key} was never executed")
            return True

        last_execution = self.executions[-1]

        if last_execution.dataset_hash != dataset_hash:
            logger.debug(
                f"Execution group {self.diagnostic.slug}/{self.key} hash mismatch:"
                f" {last_execution.dataset_hash} != {dataset_hash}"
            )
            return True

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
            return False

        # Dirty flag is the primary signal for rerunning existing jobs
        if self.dirty:
            logger.debug(f"Execution group {self.diagnostic.slug}/{self.key} is dirty")
            return True

        # Re-run all failed executions if explicitly requested
        if (last_execution.successful is False or treat_as_failed) and rerun_failed:
            logger.debug(
                f"Execution group {self.diagnostic.slug}/{self.key} "
                f"last execution failed, rerunning (rerun_failed=True)"
            )
            return True

        return False


execution_datasets = Table(
    "execution_dataset",
    Base.metadata,
    Column("execution_id", ForeignKey("execution.id"), index=True),
    Column("dataset_id", ForeignKey("dataset.id"), index=True),
)


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

    execution_group: Mapped["ExecutionGroup"] = relationship(back_populates="executions")
    outputs: Mapped[list["ExecutionOutput"]] = relationship(back_populates="execution")
    values: Mapped[list["MetricValue"]] = relationship(back_populates="execution")

    datasets: Mapped[list[Dataset]] = relationship(secondary=execution_datasets)
    """
    The datasets used in this execution
    """

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
        self.path = str(path)

    def mark_failed(self) -> None:
        """
        Mark the diagnostic execution as unsuccessful
        """
        self.successful = False


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


def _latest_execution_ids(
    session: Session,
    among_executions: Sequence[Any] | None = None,
) -> Any:
    """
    Subquery selecting the id of the latest execution in each group.

    "Latest" is the execution with the greatest ``created_at``, breaking ties
    deterministically by greatest ``id`` so that exactly one execution wins per
    group. This is stricter than a bare ``max(created_at)`` equijoin, which
    duplicates a group row when two executions share the same timestamp.

    Parameters
    ----------
    session
        The database session to use for the query.
    among_executions
        Optional predicates on ``Execution`` applied *before* ranking. These
        change which execution is considered "latest" for each group -- the
        winner is the newest execution *among those matching the predicates*.
        For example ``[Execution.successful.is_(True)]`` selects each group's
        latest **successful** execution rather than its latest execution.
        Group-level filters (diagnostic, provider, dirty, version) do not belong
        here -- apply them to the outer query, where they narrow which groups are
        returned without changing which execution wins.

    Returns
    -------
        A subquery exposing ``execution_id`` and ``group_id`` columns, one row
        per group that has at least one matching execution.
    """
    ranked = session.query(
        Execution.id.label("execution_id"),
        Execution.execution_group_id.label("group_id"),
        func.row_number()
        .over(
            partition_by=Execution.execution_group_id,
            order_by=(Execution.created_at.desc(), Execution.id.desc()),
        )
        .label("rn"),
    )
    for predicate in among_executions or ():
        ranked = ranked.filter(predicate)
    ranked_sq = ranked.subquery()

    return (
        session.query(
            ranked_sq.c.execution_id.label("execution_id"),
            ranked_sq.c.group_id.label("group_id"),
        )
        .filter(ranked_sq.c.rn == 1)
        .subquery()
    )


def _successful_predicate(successful: bool) -> Any:
    """
    Build a predicate on ``Execution.successful``.

    ``True`` matches successful executions; ``False`` matches unsuccessful *or*
    in-progress (NULL) executions -- and, when applied to an outer-joined
    ``Execution``, also matches groups with no execution at all (NULL row).
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
        Optional predicates on ``Execution`` applied *before* the latest-per-group
        ranking, so "latest" is chosen from that filtered population (see
        :func:`_latest_execution_ids`). Defaults to ranking over all executions.

    Returns
    -------
        Query to get the most recent result for each execution group.
        The result is a tuple of the execution group and the most recent result,
        which can be None.
    """
    latest = _latest_execution_ids(session, among_executions)

    # Groups with no matching execution still appear (outer join, Execution=None).
    query = (
        session.query(ExecutionGroup, Execution)
        .outerjoin(latest, latest.c.group_id == ExecutionGroup.id)
        .outerjoin(Execution, Execution.id == latest.c.execution_id)
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


def _filter_executions_by_facets(
    results: Sequence[tuple[ExecutionGroup, Execution | None]],
    facet_filters: dict[str, list[str]],
) -> list[tuple[ExecutionGroup, Execution | None]]:
    """
    Filter execution groups and their latest executions based on facet key-value pairs.

    This is a relatively expensive operation as it requires iterating over all results.
    This should be replaced once we have normalised the selectors into a separate table.

    Parameters
    ----------
    results
        List of tuples containing ExecutionGroup and its latest Execution (or None)
    facet_filters
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
            for facet_key, facet_values in facet_filters.items()
        )
    ]


def get_execution_group_and_latest_filtered(  # noqa: PLR0913
    session: Session,
    diagnostic_filters: list[str] | None = None,
    provider_filters: list[str] | None = None,
    facet_filters: dict[str, list[str]] | None = None,
    dirty: bool | None = None,
    successful: bool | None = None,
    latest_successful: bool | None = None,
    include_superseded: bool = False,
) -> list[tuple[ExecutionGroup, Execution | None]]:
    """
    Query execution groups with filtering capabilities.

    By default, returns only execution groups whose ``diagnostic_version`` matches
    the parent diagnostic's ``promoted_version`` so consumers see exactly one
    version's worth of results.
    Pass ``include_superseded=True`` to bypass the version filter and see the full history.

    Parameters
    ----------
    session
        Database session
    diagnostic_filters
        List of diagnostic slug substrings (OR logic, case-insensitive)
    provider_filters
        List of provider slug substrings (OR logic, case-insensitive)
    facet_filters
        Dictionary mapping facet keys to lists of allowed values.
        Different keys are ANDed; multiple values for the same key are ORed.
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
        If True, rank only over successful executions, so the returned execution is each group's
        latest successful run (a group whose newest run failed but succeeded earlier is still
        included, showing that earlier success).
        If False, rank only over unsuccessful / in-progress executions.
        If None (default), rank over all executions.
        This differs from ``successful``: ``successful=True`` keeps a group only if its newest run
        happened to succeed, whereas ``latest_successful=True`` changes which run is chosen as
        newest. The two compose but answer different questions.
    include_superseded
        If True, include execution groups for diagnostic versions older than the
        currently promoted version.
        If False (default), join ``Diagnostic`` and filter to ``ExecutionGroup.diagnostic_version
        == Diagnostic.promoted_version``.
        Set this for recovery / audit callers that need the full version history
        (e.g. ``executor/reingest.py``).

    Returns
    -------
        Query returning tuples of (ExecutionGroup, latest Execution or None)

    Notes
    -----
    - Diagnostic and provider filters use substring matching (case-insensitive)
    - Multiple values within same filter type use OR logic
    - Different filter types use AND logic
    - Facet filters can either be key=value (searches all dataset types)
      or dataset_type.key=value (searches specific dataset type)
    - This helper is the only sanctioned path for new callers that should respect the promoted-version filter.
      The one acknowledged exception is the ``cli/executions.py::stats`` aggregation,
      which inlines
      ``.join(Diagnostic).filter(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)``
      because it returns aggregate rows rather than a list of tuples and so cannot reuse this helper.
      Operational queries that must remain version-agnostic
      (e.g. ``mark_failed_running`` in the same module) intentionally do not use this helper at all.
    """
    # Pre-rank population filter: restrict which executions "latest" is chosen from.
    among_executions = None if latest_successful is None else [_successful_predicate(latest_successful)]

    # Start with base query
    query = get_execution_group_and_latest(session, among_executions=among_executions)

    # Join Diagnostic when needed for filtering (by name or by promoted version).
    needs_diagnostic_join = bool(diagnostic_filters or provider_filters) or not include_superseded
    if needs_diagnostic_join:
        query = query.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)

    if not include_superseded:
        query = query.filter(ExecutionGroup.diagnostic_version == Diagnostic.promoted_version)

    # Apply diagnostic filter (OR logic for multiple values)
    if diagnostic_filters:
        diagnostic_conditions = [
            Diagnostic.slug.ilike(f"%{filter_value.lower()}%") for filter_value in diagnostic_filters
        ]
        query = query.filter(or_(*diagnostic_conditions))

    # Apply provider filter (OR logic for multiple values)
    if provider_filters:
        # Need to join through Diagnostic to Provider
        query = query.join(Provider, Diagnostic.provider_id == Provider.id)

        provider_conditions = [
            Provider.slug.ilike(f"%{filter_value.lower()}%") for filter_value in provider_filters
        ]
        query = query.filter(or_(*provider_conditions))

    if successful is not None:
        query = query.filter(_successful_predicate(successful))

    if dirty is not None:
        if dirty:
            query = query.filter(ExecutionGroup.dirty.is_(True))
        else:
            query = query.filter(or_(ExecutionGroup.dirty.is_(False), ExecutionGroup.dirty.is_(None)))

    if facet_filters:
        # Load all results into memory for Python-based filtering
        # TODO: Update once we have normalised the selector
        results = [r._tuple() for r in query.all()]
        return _filter_executions_by_facets(results, facet_filters)
    else:
        return [r._tuple() for r in query.all()]
