"""
Solver to determine which diagnostics need to be calculated

This module provides a solver to determine which diagnostics need to be calculated.
"""

import itertools
import pathlib
import typing
from collections.abc import Sequence

import pandas as pd
from attrs import define, frozen
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.datasets import (
    CMIP6DatasetAdapter,
    CMIP7DatasetAdapter,
    Obs4MIPsDatasetAdapter,
    PMPClimatologyDatasetAdapter,
    get_dataset_adapter,
)
from climate_ref.models import Diagnostic as DiagnosticModel
from climate_ref.models import ExecutionGroup
from climate_ref.models import Provider as ProviderModel
from climate_ref.models.execution import Execution
from climate_ref.provider_registry import ProviderRegistry
from climate_ref_core.constraints import apply_constraint
from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    Selector,
    SourceDatasetType,
)
from climate_ref_core.diagnostics import DataRequirement, Diagnostic, ExecutionDefinition
from climate_ref_core.exceptions import InvalidDiagnosticException
from climate_ref_core.providers import DiagnosticProvider


@frozen
class DiagnosticExecution:
    """
    Class to hold information about the execution of a diagnostic

    This is a temporary class used by the solver to hold information about an execution that might
    be required.
    """

    provider: DiagnosticProvider
    diagnostic: Diagnostic
    datasets: ExecutionDatasetCollection

    def execution_slug(self) -> str:
        """
        Get a slug for the execution
        """
        return f"{self.diagnostic.full_slug()}/{self.dataset_key}"

    @property
    def dataset_key(self) -> str:
        """
        Key used to uniquely identify the execution group

        This key is unique to an execution group and uses unique set of metadata (selectors)
         that defines the group.
        This key is combines the selectors from each source dataset type into a single key
        and should be stable if new datasets are added or removed.
        """
        key_values = []

        for source_type in SourceDatasetType.ordered():
            # Ensure the selector is sorted using the dimension names
            # This will ensure a stable key even if the groupby order changes
            if source_type not in self.datasets:
                continue

            selector = self.datasets[source_type].selector
            selector_sorted = sorted(selector, key=lambda item: item[0])

            source_key = f"{source_type.value}_" + "_".join(value for _, value in selector_sorted)
            key_values.append(source_key)

        return "__".join(key_values)

    @property
    def selectors(self) -> dict[str, Selector]:
        """
        Collection of selectors used to identify the datasets

        These are the key, value pairs that were selected during the initial group-by,
        for each data requirement.
        """
        return self.datasets.selectors

    def build_execution_definition(self, output_root: pathlib.Path) -> ExecutionDefinition:
        """
        Build the execution definition for the current diagnostic execution
        """
        # Ensure that the output root is always an absolute path
        output_root = output_root.resolve()

        # This is the desired path relative to the output directory
        fragment = pathlib.Path() / self.provider.slug / self.diagnostic.slug / self.datasets.hash

        return ExecutionDefinition(
            diagnostic=self.diagnostic,
            root_directory=output_root,
            output_directory=output_root / fragment,
            key=self.dataset_key,
            datasets=self.datasets,
        )


def extract_covered_datasets(
    data_catalog: pd.DataFrame, requirement: DataRequirement
) -> dict[Selector, pd.DataFrame]:
    """
    Determine the different diagnostic executions that should be performed with the current data catalog
    """
    if len(data_catalog) == 0:
        logger.error(f"No datasets found in the data catalog: {requirement.source_type.value}")
        return {}

    subset = requirement.apply_filters(data_catalog)

    if len(subset) == 0:
        logger.debug(f"No datasets found for requirement {requirement}")
        return {}

    if requirement.group_by is None:
        # Use a single group
        groups = [((), subset)]
    else:
        groups = list(subset.groupby(list(requirement.group_by)))  # type: ignore[arg-type]

    results = {}

    for name, group in groups:
        if requirement.group_by is None:
            assert len(groups) == 1
            group_keys: Selector = ()
        else:
            group_keys = tuple(zip(requirement.group_by, name))
        constrained_group = _process_group_constraints(data_catalog, group, requirement)

        if constrained_group is not None:
            results[group_keys] = constrained_group

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


def solve_executions(
    data_catalog: dict[SourceDatasetType, pd.DataFrame], diagnostic: Diagnostic, provider: DiagnosticProvider
) -> typing.Generator["DiagnosticExecution", None, None]:
    """
    Calculate the diagnostic executions that need to be performed for a given diagnostic

    Parameters
    ----------
    data_catalog
        Data catalogs for each source dataset type
    diagnostic
        Diagnostic of interest
    provider
        Provider of the diagnostic

    Returns
    -------
    :
        A generator that yields the diagnostic executions that need to be performed

    """
    if not diagnostic.data_requirements:
        raise ValueError(f"Diagnostic {diagnostic.slug!r} has no data requirements")

    first_item = next(iter(diagnostic.data_requirements))

    if isinstance(first_item, DataRequirement):
        # We have a single collection of data requirements
        yield from _solve_from_data_requirements(
            data_catalog,
            diagnostic,
            typing.cast(Sequence[DataRequirement], diagnostic.data_requirements),
            provider,
        )
    elif isinstance(first_item, Sequence):
        # We have a sequence of collections of data requirements (OR logic)
        # Try each requirement collection and yield from those that have matching data
        any_matched = False
        for requirement_collection in diagnostic.data_requirements:
            if not isinstance(requirement_collection, Sequence):
                raise TypeError(f"Expected a sequence of DataRequirement, got {type(requirement_collection)}")
            try:
                # Check if all required source types are available
                for req in requirement_collection:
                    if isinstance(req, DataRequirement) and req.source_type not in data_catalog:
                        raise InvalidDiagnosticException(
                            diagnostic, f"No data catalog for source type {req.source_type}"
                        )
                yield from _solve_from_data_requirements(
                    data_catalog, diagnostic, requirement_collection, provider
                )
                any_matched = True
            except InvalidDiagnosticException:
                # This requirement set doesn't match available data, try the next one
                continue
        if not any_matched:
            available = ", ".join(str(s) for s in data_catalog.keys())
            raise InvalidDiagnosticException(
                diagnostic,
                f"No data catalog matches any of the diagnostic's data requirements. "
                f"Available source types: {available}",
            )
    else:
        raise TypeError(f"Expected a DataRequirement, got {type(first_item)}")


def _solve_from_data_requirements(
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
    diagnostic: Diagnostic,
    data_requirements: Sequence[DataRequirement],
    provider: DiagnosticProvider,
) -> typing.Generator["DiagnosticExecution", None, None]:
    # Collect up the different data groups that can be used to calculate the diagnostic
    dataset_groups = {}

    for requirement in data_requirements:
        if not isinstance(requirement, DataRequirement):
            raise TypeError(f"Expected a DataRequirement, got {type(requirement)}")
        if requirement.source_type not in data_catalog:
            raise InvalidDiagnosticException(
                diagnostic, f"No data catalog for source type {requirement.source_type}"
            )

        dataset_groups[requirement.source_type] = extract_covered_datasets(
            data_catalog[requirement.source_type], requirement
        )

    # Calculate the product across each of the source types
    for items in itertools.product(*dataset_groups.values()):
        yield DiagnosticExecution(
            provider=provider,
            diagnostic=diagnostic,
            datasets=ExecutionDatasetCollection(
                {
                    source_type: DatasetCollection(
                        datasets=dataset_groups[source_type][selector],
                        slug_column=get_dataset_adapter(source_type.value).slug_column,
                        selector=selector,
                    )
                    for source_type, selector in zip(dataset_groups.keys(), items)
                }
            ),
        )


@define
class SolveFilterOptions:
    """
    Options to filter the diagnostics that are solved
    """

    diagnostic: list[str] | None = None
    """
    Check if the diagnostic slug contains any of the provided values
    """
    provider: list[str] | None = None
    """
    Check if the provider slug contains any of the provided values
    """


def matches_filter(diagnostic: Diagnostic, filters: SolveFilterOptions | None) -> bool:
    """
    Check if a diagnostic matches the provided filters

    Each filter is optional and a diagnostic will match if it satisfies all the provided filters.
    i.e. the filters are ANDed together.

    Parameters
    ----------
    diagnostic
        Diagnostic to check against the filters
    filters
        Collection of filters to apply to the diagnostic

        If no filters are provided, the diagnostic is considered to match

    Returns
    -------
        True if the diagnostic matches the filters, False otherwise
    """
    if filters is None:
        return True

    diagnostic_slug = diagnostic.slug
    provider_slug = diagnostic.provider.slug

    if filters.provider and not any([f.lower() in provider_slug for f in filters.provider]):
        return False

    if filters.diagnostic and not any([f.lower() in diagnostic_slug for f in filters.diagnostic]):
        return False

    return True


@define
class ExecutionSolver:
    """
    A solver to determine which executions need to be calculated.
    """

    provider_registry: ProviderRegistry
    data_catalog: dict[SourceDatasetType, pd.DataFrame]

    @staticmethod
    def build_from_db(config: Config, db: Database) -> "ExecutionSolver":
        """
        Initialise the solver using information from the database

        Parameters
        ----------
        db
            Database instance

        Returns
        -------
        :
            A new ExecutionSolver instance
        """
        return ExecutionSolver(
            provider_registry=ProviderRegistry.build_from_config(config, db),
            data_catalog={
                SourceDatasetType.CMIP6: CMIP6DatasetAdapter().load_catalog(db),
                SourceDatasetType.CMIP7: CMIP7DatasetAdapter().load_catalog(db),
                SourceDatasetType.obs4MIPs: Obs4MIPsDatasetAdapter().load_catalog(db),
                SourceDatasetType.PMPClimatology: PMPClimatologyDatasetAdapter().load_catalog(db),
            },
        )

    def solve(
        self, filters: SolveFilterOptions | None = None
    ) -> typing.Generator[DiagnosticExecution, None, None]:
        """
        Solve which executions need to be calculated for a dataset

        The solving scheme is iterative,
        for each iteration we find all diagnostics that can be solved and calculate them.
        After each iteration we check if there are any more diagnostics to solve.

        Yields
        ------
        DiagnosticExecution
            A class containing the information related to the execution of a diagnostic
        """
        for provider in self.provider_registry.providers:
            for diagnostic in provider.diagnostics():
                # Filter the diagnostic based on the provided filters
                if not matches_filter(diagnostic, filters):
                    logger.debug(f"Skipping {diagnostic.full_slug()} due to filter")
                    continue
                yield from solve_executions(self.data_catalog, diagnostic, provider)


def solve_required_executions(  # noqa: PLR0912, PLR0913
    db: Database,
    dry_run: bool = False,
    execute: bool = True,
    solver: ExecutionSolver | None = None,
    config: Config | None = None,
    timeout: int = 60,
    one_per_provider: bool = False,
    one_per_diagnostic: bool = False,
    filters: SolveFilterOptions | None = None,
) -> None:
    """
    Solve for executions that require recalculation

    This may trigger a number of additional calculations depending on what data has been ingested
    since the last solve.

    Raises
    ------
    TimeoutError
        If the execution isn't completed within the specified timeout
    """
    if config is None:
        config = Config.default()
    if solver is None:
        solver = ExecutionSolver.build_from_db(config, db)

    logger.info("Solving for diagnostics that require recalculation...")

    executor = config.executor.build(config, db)

    diagnostic_count = {}
    provider_count = {}

    for potential_execution in solver.solve(filters):
        # The diagnostic output is first written to the scratch directory
        definition = potential_execution.build_execution_definition(output_root=config.paths.scratch)

        logger.debug(
            f"Identified candidate execution {definition.key} "
            f"for {potential_execution.diagnostic.full_slug()}"
        )

        if potential_execution.provider.slug not in provider_count:
            provider_count[potential_execution.provider.slug] = 0
        if potential_execution.diagnostic.full_slug() not in diagnostic_count:
            diagnostic_count[potential_execution.diagnostic.full_slug()] = 0

        if dry_run:
            provider_count[potential_execution.provider.slug] += 1
            diagnostic_count[potential_execution.diagnostic.full_slug()] += 1
            continue

        # Use a transaction to make sure that the models
        # are created correctly before potentially executing out of process
        with db.session.begin():
            diagnostic = (
                db.session.query(DiagnosticModel)
                .join(DiagnosticModel.provider)
                .filter(
                    ProviderModel.slug == potential_execution.provider.slug,
                    DiagnosticModel.slug == potential_execution.diagnostic.slug,
                )
                .one()
            )
            execution_group, created = db.get_or_create(
                ExecutionGroup,
                key=definition.key,
                diagnostic_id=diagnostic.id,
                defaults={
                    "selectors": potential_execution.selectors,
                    "dirty": True,
                },
            )

            if created:
                logger.info(f"Created new execution group: {potential_execution.execution_slug()!r}")
                db.session.flush()

            # TODO: Move this logic to the solver
            # Check if we should run given the one_per_provider or one_per_diagnostic flags
            one_of_check_failed = (
                one_per_provider and provider_count.get(diagnostic.provider.slug, 0) > 0
            ) or (one_per_diagnostic and diagnostic_count.get(diagnostic.full_slug(), 0) > 0)

            logger.debug(
                f"one_per_provider={one_per_provider}, one_per_diagnostic={one_per_diagnostic}, "
                f"one_of_check_failed={one_of_check_failed}, diagnostic_count={diagnostic_count}, "
                f"provider_count={provider_count}"
            )

            if execution_group.should_run(definition.datasets.hash):
                if (one_per_provider or one_per_diagnostic) and one_of_check_failed:
                    logger.info(
                        f"Skipping execution due to one-of check: {potential_execution.execution_slug()!r}"
                    )
                    continue

                logger.info(
                    f"Running new execution for execution group: {potential_execution.execution_slug()!r}"
                )
                execution = Execution(
                    execution_group=execution_group,
                    dataset_hash=definition.datasets.hash,
                    output_fragment=str(definition.output_fragment()),
                )
                db.session.add(execution)
                db.session.flush()

                # Add links to the datasets used in the execution
                execution.register_datasets(db, definition.datasets)

                if execute:
                    executor.run(
                        definition=definition,
                        execution=execution,
                    )

                provider_count[diagnostic.provider.slug] += 1
                diagnostic_count[diagnostic.full_slug()] += 1

    logger.info("Solve complete")
    logger.info(f"Found {sum(diagnostic_count.values())} new executions")
    for diag, count in diagnostic_count.items():
        logger.info(f"  {diag}: {count} new executions")
    for prov, count in provider_count.items():
        logger.info(f"  {prov}: {count} new executions")

    if timeout > 0:
        executor.join(timeout=timeout)
        logger.info("All executions complete")
