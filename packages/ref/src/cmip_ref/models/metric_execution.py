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


class MetricExecution(CreatedUpdatedMixin, Base):
    """
    Represents an execution of a metric calculation

    Each execution is a run of a metric calculation with a specific set of input datasets.
    The metric_id, key form an identifier of a unique group.
    """

    __tablename__ = "metric_execution"
    __table_args__ = (UniqueConstraint("metric_id", "key", name="metric_execution_ident"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    retracted: Mapped[bool] = mapped_column(default=False)
    """
    Whether the metric execution has been retracted or not

    This may happen if a dataset has been retracted, or if the metric execution was incorrect.
    Rather than delete the values, they are marked as retracted.
    These data may still be visible in the UI, but should be marked as retracted.
    """

    metric_id: Mapped[int] = mapped_column(ForeignKey("metric.id"))
    """
    The target metric
    """

    key: Mapped[str] = mapped_column(index=True)
    """
    Key for the metric execution

    This should be unique for each run of a metric.
    """

    dirty: Mapped[bool] = mapped_column(default=False)
    """
    Whether the execution should be rerun

    An execution is dirty if the metric or any of the input datasets has been updated since the last run.
    """

    metric: Mapped["Metric"] = relationship(back_populates="executions")
    results: Mapped[list["MetricExecutionResult"]] = relationship(
        back_populates="metric_execution", order_by="MetricExecutionResult.created_at"
    )

    def should_run(self, dataset_hash: str) -> bool:
        """
        Check if a new run of the metric execution should be performed

        The metric execution should be run if:

        * the execution if marked as dirty
        * no runs have been performed
        * the dataset hash is different from the last run
        """
        if not self.results:
            logger.debug(f"Execution {self.key} no previous results")
            return True

        if self.results[-1].dataset_hash != dataset_hash:
            logger.debug(
                f"Execution {self.key} hash mismatch: {self.results[-1].dataset_hash} != {dataset_hash}"
            )
            return True

        if self.dirty:
            logger.debug(f"Execution {self.key} is dirty")
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

    An execution might be run multiple times as new data becomes available.
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

    metric_execution_id: Mapped[int] = mapped_column(ForeignKey("metric_execution.id"))
    """
    The target metric execution
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

    Relative to the metric execution result output directory
    """

    metric_execution: Mapped["MetricExecution"] = relationship(back_populates="results")
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
        Mark the metric execution as successful
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


def get_execution_and_latest_result(
    session: Session,
) -> RowReturningQuery[tuple[MetricExecution, MetricExecutionResult | None]]:
    """
    Query to get the most recent result for each metric execution

    Parameters
    ----------
    session
        The database session to use for the query.

    Returns
    -------
        Query to get the most recent result for each metric execution.
        The result is a tuple of the metric execution and the most recent result,
        which can be None.
    """
    # Find the most recent result for each metric execution
    # This uses an aggregate function because it is more efficient than order by
    subquery = (
        session.query(
            MetricExecutionResult.metric_execution_id,
            func.max(MetricExecutionResult.created_at).label("latest_created_at"),
        )
        .group_by(MetricExecutionResult.metric_execution_id)
        .subquery()
    )

    # Join the metric execution with the latest result
    query = (
        session.query(MetricExecution, MetricExecutionResult)
        .outerjoin(subquery, MetricExecution.id == subquery.c.metric_execution_id)
        .outerjoin(
            MetricExecutionResult,
            (MetricExecutionResult.metric_execution_id == MetricExecution.id)
            & (MetricExecutionResult.created_at == subquery.c.latest_created_at),
        )
    )

    return query  # type: ignore
