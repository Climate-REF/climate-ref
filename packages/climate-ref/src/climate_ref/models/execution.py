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
    __table_args__ = (UniqueConstraint("diagnostic_id", "key", name="execution_ident"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    diagnostic_id: Mapped[int] = mapped_column(ForeignKey("diagnostic.id"), index=True)
    """
    The diagnostic that this execution group belongs to
    """

    key: Mapped[str] = mapped_column(index=True)
    """
    Key for the datasets in this Execution group.
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

    def should_run(self, dataset_hash: str) -> bool:
        """
        Check if the diagnostic execution group needs to be executed.

        The diagnostic execution group should be run if:

        * the execution group is marked as dirty
        * no executions have been performed ever
        * the dataset hash is different from the last run
        """
        if not self.executions:
            logger.debug(f"Execution group {self.diagnostic.slug}/{self.key} was never executed")
            return True

        if self.executions[-1].dataset_hash != dataset_hash:
            logger.debug(
                f"Execution group {self.diagnostic.slug}/{self.key} hash mismatch:"
                f" {self.executions[-1].dataset_hash} != {dataset_hash}"
            )
            return True

        if self.dirty:
            logger.debug(f"Execution group {self.diagnostic.slug}/{self.key} is dirty")
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

    successful: Mapped[bool] = mapped_column(nullable=True, index=True)
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


def get_execution_group_and_latest(
    session: Session,
) -> RowReturningQuery[tuple[ExecutionGroup, Execution | None]]:
    """
    Query to get the most recent result for each execution group

    Parameters
    ----------
    session
        The database session to use for the query.

    Returns
    -------
        Query to get the most recent result for each execution group.
        The result is a tuple of the execution group and the most recent result,
        which can be None.
    """
    # Find the most recent result for each execution group
    # This uses an aggregate function because it is more efficient than order by
    subquery = (
        session.query(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("latest_created_at"),
        )
        .group_by(Execution.execution_group_id)
        .subquery()
    )

    # Join the diagnostic execution with the latest result
    query = (
        session.query(ExecutionGroup, Execution)
        .outerjoin(subquery, ExecutionGroup.id == subquery.c.execution_group_id)
        .outerjoin(
            Execution,
            (Execution.execution_group_id == ExecutionGroup.id)
            & (Execution.created_at == subquery.c.latest_created_at),
        )
    )

    return query  # type: ignore


def _filter_executions_by_facets(
    results: Sequence[tuple[ExecutionGroup, Execution | None]],
    facet_filters: dict[str, str],
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
        Dictionary of facet key-value pairs to filter by (AND logic, exact match)

    Returns
    -------
        Filtered list of tuples containing ExecutionGroup and its latest Execution (or None)

    Notes
    -----
    - Facet filters can either be key=value (searches all dataset types)
      or dataset_type.key=value (searches specific dataset type)
    - Key=value filters search across all dataset types
    - dataset_type.key=value filters only search within the specified dataset type
    - Multiple values within same filter type use OR logic
    - All specified facets must match for an execution group to be included (AND logic)
    """
    filtered_results = []
    for eg, execution in results:
        all_filters_match = True
        for facet_key, facet_value in facet_filters.items():
            filter_match = False
            if "." in facet_key:
                # Handle dataset_type.key=value format
                dataset_type, key = facet_key.split(".", 1)
                if dataset_type in eg.selectors:
                    if [key, facet_value] in eg.selectors[dataset_type]:
                        filter_match = True
                        break
            else:
                # Handle key=value format (search across all dataset types)
                for ds_type_selectors in eg.selectors.values():
                    if [facet_key, facet_value] in ds_type_selectors:
                        filter_match = True
                        break

            if not filter_match:
                all_filters_match = False
                break
        if all_filters_match:
            filtered_results.append((eg, execution))
    return filtered_results


def get_execution_group_and_latest_filtered(  # noqa: PLR0913
    session: Session,
    diagnostic_filters: list[str] | None = None,
    provider_filters: list[str] | None = None,
    facet_filters: dict[str, str] | None = None,
    dirty: bool | None = None,
    successful: bool | None = None,
) -> list[tuple[ExecutionGroup, Execution | None]]:
    """
    Query execution groups with filtering capabilities.

    Parameters
    ----------
    session
        Database session
    diagnostic_filters
        List of diagnostic slug substrings (OR logic, case-insensitive)
    provider_filters
        List of provider slug substrings (OR logic, case-insensitive)
    facet_filters
        Dictionary of facet key-value pairs (AND logic, exact match)
    dirty
        If True, only return dirty execution groups.
        If False, only return clean execution groups.
        If None, do not filter by dirty status.
    successful
        If True, only return execution groups whose latest execution was successful.
        If False, only return execution groups whose latest execution was unsuccessful or has no executions.
        If None, do not filter by execution success.

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
    """
    # Start with base query
    query = get_execution_group_and_latest(session)

    if diagnostic_filters or provider_filters:
        # Join through to the Diagnostic table
        query = query.join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)

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
        if successful:
            query = query.filter(Execution.successful.is_(True))
        else:
            query = query.filter(or_(Execution.successful.is_(False), Execution.successful.is_(None)))

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
