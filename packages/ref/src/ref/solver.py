"""
Solver to determine which metrics need to be calculated

This module provides a solver to determine which metrics need to be calculated.

This module is still a work in progress and is not yet fully implemented.
"""

import itertools
import pathlib
import typing

import pandas as pd
from attrs import define, frozen
from loguru import logger
from ref_core.constraints import apply_constraint
from ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType
from ref_core.exceptions import InvalidMetricException
from ref_core.executor import get_executor
from ref_core.metrics import DataRequirement, Metric, MetricExecutionDefinition
from ref_core.providers import MetricsProvider

from ref.database import Database
from ref.datasets import get_dataset_adapter
from ref.datasets.cmip6 import CMIP6DatasetAdapter
from ref.env import env
from ref.provider_registry import ProviderRegistry


@frozen
class MetricExecution:
    """
    Class to hold information about the execution of a metric
    """

    provider: MetricsProvider
    metric: Metric
    metric_dataset: MetricDataset

    def build_metric_execution_info(self) -> MetricExecutionDefinition:
        """
        Build the metric execution info for the current metric execution
        """
        # TODO: We might want to pretty print the dataset slug
        slug = "_".join([self.provider.slug, self.metric.slug, self.metric_dataset.slug])

        return MetricExecutionDefinition(
            output_fragment=pathlib.Path(self.provider.slug) / self.metric.slug / self.metric_dataset.slug,
            slug=slug,
            metric_dataset=self.metric_dataset,
        )


def extract_covered_datasets(data_catalog: pd.DataFrame, requirement: DataRequirement) -> list[pd.DataFrame]:
    """
    Determine the different metric executions that should be performed with the current data catalog
    """
    subset = requirement.apply_filters(data_catalog)

    if len(subset) == 0:
        logger.debug(f"No datasets found for requirement {requirement}")
        return []

    if requirement.group_by is None:
        # Use a single group
        groups = [(None, subset)]
    else:
        groups = subset.groupby(list(requirement.group_by))  # type: ignore

    results = []

    for name, group in groups:
        constrained_group = _process_group_constraints(data_catalog, group, requirement)

        if constrained_group is not None:
            results.append(constrained_group)

    return results


def _process_group_constraints(
    data_catalog: pd.DataFrame, group: pd.DataFrame, requirement: DataRequirement
) -> pd.DataFrame | None:
    for constraint in requirement.constraints or []:
        constrained_group = apply_constraint(group, constraint, data_catalog)
        if constrained_group is None:
            return None

        group = constrained_group
    return group


@define
class MetricSolver:
    """
    A solver to determine which metrics need to be calculated
    """

    provider_registry: ProviderRegistry
    data_catalog: dict[SourceDatasetType, pd.DataFrame]

    @staticmethod
    def build_from_db(db: Database) -> "MetricSolver":
        """
        Initialise the solver using information from the database

        Parameters
        ----------
        db
            Database instance

        Returns
        -------
        :
            A new MetricSolver instance
        """
        return MetricSolver(
            provider_registry=ProviderRegistry.build_from_db(db),
            data_catalog={
                SourceDatasetType.CMIP6: CMIP6DatasetAdapter().load_catalog(db),
            },
        )

    def solve(self) -> typing.Generator[MetricExecution, None, None]:
        """
        Solve which metrics need to be calculated for a dataset

        The solving scheme is iterative,
        for each iteration we find all metrics that can be solved and calculate them.
        After each iteration we check if there are any more metrics to solve.
        This may not be the most efficient way to solve the metrics, but it's a start.

        Yields
        ------
        MetricExecution
            A class containing the information related to the execution of a metric
        """
        for provider in self.provider_registry.providers:
            for metric in provider.metrics():
                # Collect up the different data groups that can be used to calculate the metric
                dataset_groups = {}

                for requirement in metric.data_requirements:
                    if requirement.source_type not in self.data_catalog:
                        raise InvalidMetricException(
                            metric, f"No data catalog for source type {requirement.source_type}"
                        )

                    dataset_groups[requirement.source_type] = extract_covered_datasets(
                        self.data_catalog[requirement.source_type], requirement
                    )

                # I'm not sure if the right approach here is a product of the groups
                for items in itertools.product(*dataset_groups.values()):
                    yield MetricExecution(
                        provider=provider,
                        metric=metric,
                        metric_dataset=MetricDataset(
                            {
                                key: DatasetCollection(
                                    datasets=value, slug_column=get_dataset_adapter(key.value).slug_column
                                )
                                for key, value in zip(dataset_groups.keys(), items)
                            }
                        ),
                    )


def solve_metrics(db: Database, dry_run: bool = False, solver: MetricSolver | None = None) -> None:
    """
    Solve for metrics that require recalculation

    This may trigger a number of additional calculations depending on what data has been ingested
    since the last solve.
    """
    if solver is None:
        solver = MetricSolver.build_from_db(db)

    logger.info("Solving for metrics that require recalculation...")

    executor = get_executor(env.str("REF_EXECUTOR", "local"))

    for metric_execution in solver.solve():
        info = metric_execution.build_metric_execution_info()

        logger.info(f"Calculating metric {info.slug}")

        if not dry_run:
            executor.run_metric(metric=metric_execution.metric, definition=info)
