import enum
import pathlib
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import Column, ForeignKey, Table, UniqueConstraint, func
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from sqlalchemy.orm.query import RowReturningQuery

from cmip_ref.models import Dataset
from cmip_ref.models.base import Base, CreatedUpdatedMixin
from cmip_ref_core.datasets import MetricDataset

if TYPE_CHECKING:
    from cmip_ref.database import Database
    from cmip_ref.models.metric import Metric


class MetricExecutionGroup(CreatedUpdatedMixin, Base):
    """
    Represents a group of executions of a metric with a set of input datasets.

    When solving, the MetricExecutionGroups are derived from the available datasets,
    the defined metrics and their data requirements. From the information in the
    group an execution can be triggered, which is an actual run of a metric calculation
    with a specific set of input datasets.

    When the MetricExecutionGroup is created, it is marked dirty, meaning there are no
    current results available. When an Execution was run successfully for a
    MetricExecutionGroup, the dirty mark is removed. After ingesting new data and
    solving again and if new versions of the input datasets are available, the
    MetricExecutionGroup will be marked dirty again.

    The metric_id and dataset_key form a unique identifier for MetricExecutionGroups.
    """

    __tablename__ = "metric_execution_group"
    __table_args__ = (UniqueConstraint("metric_id", "dataset_key", name="metric_execution_group_ident"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    metric_id: Mapped[int] = mapped_column(ForeignKey("metric.id"))
    """
    The target metric
    """

    dataset_key: Mapped[str] = mapped_column(index=True)
    """
    Key for the datasets in this Execution group.
    """

    dirty: Mapped[bool] = mapped_column(default=False)
    """
    Whether the execution group should be rerun

    An execution group is dirty if the metric or any of the input datasets has been
    updated since the last execution.
    """

    metric: Mapped["Metric"] = relationship(back_populates="execution_groups")
    results: Mapped[list["MetricExecutionResult"]] = relationship(
        back_populates="metric_execution_group", order_by="MetricExecutionResult.created_at"
    )

    def should_run(self, dataset_hash: str) -> bool:
        """
        Check if the metric execution group needs to be executed.

        The metric execution group should be run if:

        * the execution group is marked as dirty
        * no executions have been performed ever
        * the dataset hash is different from the last run
        """
        if not self.results:
            logger.debug(f"Execution group {self.metric.slug}/{self.dataset_key} was never executed")
            return True

        if self.results[-1].dataset_hash != dataset_hash:
            logger.debug(
                f"Execution group {self.metric.slug}/{self.dataset_key} hash mismatch:"
                f" {self.results[-1].dataset_hash} != {dataset_hash}"
            )
            return True

        if self.dirty:
            logger.debug(f"Execution group {self.metric.slug}/{self.dataset_key} is dirty")
            return True

        return False


metric_datasets = Table(
    "metric_execution_result_dataset",
    Base.metadata,
    Column("metric_execution_result_id", ForeignKey("metric_execution_result.id")),
    Column("dataset_id", ForeignKey("dataset.id")),
)


class MetricExecutionResult(CreatedUpdatedMixin, Base):
    """
    Represents a run of a metric calculation

    An execution group might be run multiple times as new data becomes available,
    each run will create a MetricExecutionResult.
    """

    __tablename__ = "metric_execution_result"

    id: Mapped[int] = mapped_column(primary_key=True)

    output_fragment: Mapped[str] = mapped_column()
    """
    Relative directory to store the output of the metric execution.

    During execution this directory is relative to the temporary directory.
    If the metric execution is successful, the results will be moved to the final output directory
    and the temporary directory will be cleaned up.
    This directory may contain multiple input and output files.
    """

    metric_execution_group_id: Mapped[int] = mapped_column(ForeignKey("metric_execution_group.id"))
    """
    The target metric execution group
    """

    dataset_hash: Mapped[str] = mapped_column(index=True)
    """
    Hash of the datasets used to calculate the metric

    This is used to verify if an existing metric execution has been run with the same datasets.
    """

    successful: Mapped[bool] = mapped_column(nullable=True)
    """
    Was the run successful
    """

    path: Mapped[str] = mapped_column(nullable=True)
    """
    Path to the output bundle
    """

    retracted: Mapped[bool] = mapped_column(default=False)
    """
    Whether the metric execution result has been retracted or not

    This may happen if a dataset has been retracted, or if the metric execution was incorrect.
    Rather than delete the values, they are marked as retracted.
    These data may still be visible in the UI, but should be marked as retracted.
    """

    metric_execution_group: Mapped["MetricExecutionGroup"] = relationship(back_populates="results")
    outputs: Mapped[list["ResultOutput"]] = relationship(back_populates="metric_execution_result")

    datasets: Mapped[list[Dataset]] = relationship(secondary=metric_datasets)

    def register_datasets(self, db: "Database", metric_dataset: MetricDataset) -> None:
        """
        Register the datasets used in the metric calculation
        """
        for _, dataset in metric_dataset.items():
            db.session.execute(
                metric_datasets.insert(),
                [{"metric_execution_result_id": self.id, "dataset_id": idx} for idx in dataset.index],
            )

    def mark_successful(self, path: pathlib.Path | str) -> None:
        """
        Mark the metric execution as successful
        """
        # TODO: this needs to accept both a metric and output bundle
        self.successful = True
        self.path = str(path)

    def mark_failed(self) -> None:
        """
        Mark the metric execution as unsuccessful
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


class ResultOutput(CreatedUpdatedMixin, Base):
    """
    An output generated as part of a metric execution

    These outputs are defined in the CMEC output bundle
    """

    __tablename__ = "metric_execution_result_output"

    id: Mapped[int] = mapped_column(primary_key=True)

    metric_execution_result_id: Mapped[int] = mapped_column(
        ForeignKey("metric_execution_result.id"), index=True
    )

    output_type: Mapped[ResultOutputType] = mapped_column(index=True)
    """
    Type of the output

    This will determine how the output is displayed
    """

    filename: Mapped[str] = mapped_column(nullable=True)
    """
    Path to the output

    Relative to the metric execution result output directory
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

    metric_execution_result: Mapped["MetricExecutionResult"] = relationship(back_populates="outputs")


def get_execution_group_and_latest_result(
    session: Session,
) -> RowReturningQuery[tuple[MetricExecutionGroup, MetricExecutionResult | None]]:
    """
    Query to get the most recent result for each metric execution group

    Parameters
    ----------
    session
        The database session to use for the query.

    Returns
    -------
        Query to get the most recent result for each metric execution group.
        The result is a tuple of the metric execution group and the most recent result,
        which can be None.
    """
    # Find the most recent result for each metric execution group
    # This uses an aggregate function because it is more efficient than order by
    subquery = (
        session.query(
            MetricExecutionResult.metric_execution_group_id,
            func.max(MetricExecutionResult.created_at).label("latest_created_at"),
        )
        .group_by(MetricExecutionResult.metric_execution_group_id)
        .subquery()
    )

    # Join the metric execution with the latest result
    query = (
        session.query(MetricExecutionGroup, MetricExecutionResult)
        .outerjoin(subquery, MetricExecutionGroup.id == subquery.c.metric_execution_group_id)
        .outerjoin(
            MetricExecutionResult,
            (MetricExecutionResult.metric_execution_group_id == MetricExecutionGroup.id)
            & (MetricExecutionResult.created_at == subquery.c.latest_created_at),
        )
    )

    return query  # type: ignore
