"""
Solver to determine which diagnostics need to be calculated

This module provides a solver to determine which diagnostics need to be calculated.
"""

import datetime
import itertools
import pathlib
import typing
from collections.abc import Mapping, Sequence

import attrs
import pandas as pd
from attrs import define, frozen
from loguru import logger

from climate_ref.config import Config
from climate_ref.data_catalog import DataCatalog
from climate_ref.database import Database
from climate_ref.datasets import (
    CMIP6DatasetAdapter,
    CMIP7DatasetAdapter,
    Obs4MIPsDatasetAdapter,
    PMPClimatologyDatasetAdapter,
    get_slug_column,
)
from climate_ref.executor.fragment import PLACEHOLDER_FRAGMENT, assign_execution_fragment
from climate_ref.models import Diagnostic as DiagnosticModel
from climate_ref.models import ExecutionGroup
from climate_ref.models import Provider as ProviderModel
from climate_ref.models.diagnostic import recompute_promoted_version
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
        Build the execution definition for the current diagnostic execution.

        The returned definition uses a placeholder fragment for the output directory.
        ``solve_required_executions`` rewrites ``output_directory`` via
        :func:`attrs.evolve` once the new ``Execution.id`` is known.
        """
        # Ensure that the output root is always an absolute path
        output_root = output_root.resolve()

        fragment = pathlib.Path() / self.provider.slug / self.diagnostic.slug / PLACEHOLDER_FRAGMENT

        return ExecutionDefinition(
            diagnostic=self.diagnostic,
            root_directory=output_root,
            output_directory=output_root / fragment,
            key=self.dataset_key,
            datasets=self.datasets,
        )


def _validate_requirement_columns(
    requirement: DataRequirement, columns_requiring_finalisation: frozenset[str]
) -> None:
    """
    Validate that a DataRequirement does not filter or group on unfinalised columns.

    Parameters
    ----------
    requirement
        The data requirement to validate
    columns_requiring_finalisation
        Set of column names that are unavailable before finalisation

    Raises
    ------
    ValueError
        If any filter facet key or group_by column requires finalisation
    """
    if not columns_requiring_finalisation:
        return

    # Check filter facet keys
    if requirement.filters:
        for facet_filter in requirement.filters:
            conflicting = set(facet_filter.facets.keys()) & columns_requiring_finalisation
            if conflicting:
                raise ValueError(
                    f"DataRequirement for {requirement.source_type.value} filters on columns "
                    f"that require finalisation: {sorted(conflicting)}. "
                    f"These columns are unavailable before datasets are finalised."
                )

    # Check group_by columns
    if requirement.group_by:
        conflicting = set(requirement.group_by) & columns_requiring_finalisation
        if conflicting:
            raise ValueError(
                f"DataRequirement for {requirement.source_type.value} groups by columns "
                f"that require finalisation: {sorted(conflicting)}. "
                f"These columns are unavailable before datasets are finalised."
            )


def extract_covered_datasets(
    data_catalog: pd.DataFrame | DataCatalog, requirement: DataRequirement
) -> dict[Selector, pd.DataFrame]:
    """
    Determine the different diagnostic executions that should be performed with the current data catalog
    """
    # Resolve the DataCatalog to a DataFrame for filtering, but keep the
    # original reference for finalisation calls later.
    if isinstance(data_catalog, DataCatalog):
        data_catalog_source: DataCatalog | None = data_catalog
        _validate_requirement_columns(requirement, data_catalog.columns_requiring_finalisation)
        catalog_df: pd.DataFrame = data_catalog.to_frame()
    else:
        data_catalog_source = None
        catalog_df = data_catalog

    if len(catalog_df) == 0:
        logger.error(f"No datasets found in the data catalog: {requirement.source_type.value}")
        return {}

    subset = requirement.apply_filters(catalog_df)

    if len(subset) == 0:
        logger.debug(f"No datasets found for requirement {requirement}")
        return {}

    # Finalise all unfinalised datasets in the filtered subset upfront,
    if data_catalog_source is not None:
        subset = data_catalog_source.finalise(subset)
        # Refresh catalog_df so constraints that reference the full catalog
        # (e.g. AddParentDataset, AddSupplementaryDataset) see finalised data.
        catalog_df = data_catalog_source.to_frame()

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

        constrained_group = _process_group_constraints(catalog_df, group, requirement, data_catalog_source)

        if constrained_group is not None:
            results[group_keys] = constrained_group

    return results


def _process_group_constraints(
    data_catalog: pd.DataFrame,
    group: pd.DataFrame,
    requirement: DataRequirement,
    data_catalog_source: DataCatalog | None = None,
) -> pd.DataFrame | None:
    for constraint in requirement.constraints or []:
        constrained_group = apply_constraint(group, constraint, data_catalog)
        if constrained_group is None:
            return None

        # Re-finalise after constraints that add rows from the unfinalised
        # catalog (e.g. AddParentDataset, AddSupplementaryDataset).  Without
        # this, columns like ``calendar`` remain NaN and downstream
        # constraints such as RequireContiguousTimerange silently skip checks.
        if data_catalog_source is not None:
            constrained_group = data_catalog_source.finalise(constrained_group)

        group = constrained_group
    return group


def solve_executions(
    data_catalog: Mapping[SourceDatasetType, pd.DataFrame | DataCatalog],
    diagnostic: Diagnostic,
    provider: DiagnosticProvider,
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
            # Buffer executions to check if any were actually produced
            # _solve_from_data_requirements returns empty if source types are missing
            executions = list(
                _solve_from_data_requirements(data_catalog, diagnostic, requirement_collection, provider)
            )
            if executions:
                any_matched = True
                yield from executions
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
    data_catalog: Mapping[SourceDatasetType, pd.DataFrame | DataCatalog],
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
            logger.debug(
                f"No data catalog for source type {requirement.source_type} of "
                f"{provider.slug} diagnostic {diagnostic.slug}"
            )
            return

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
                        slug_column=get_slug_column(source_type),
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
    dataset: dict[str, list[str]] | None = None
    """
    Filter datasets by facet values before solving.

    Keys are facet names (e.g. ``source_id``, ``variable_id``) and values are
    lists of allowed values.  Different facets are ANDed together; multiple
    values for the same facet are ORed.
    """


def apply_dataset_filters(
    data_catalog: Mapping[SourceDatasetType, DataCatalog | pd.DataFrame],
    dataset_filters: dict[str, list[str]],
) -> dict[SourceDatasetType, DataCatalog | pd.DataFrame]:
    """
    Filter data catalogs by facet values

    Each facet filter is applied independently to each data catalog.
    Different facets are ANDed together; multiple values for the same facet are ORed.
    Facets that do not exist as columns in a given catalog are skipped for that catalog.

    When a DataCatalog is provided, the returned value preserves the DataCatalog
    wrapper (with adapter and database references) so that downstream finalisation
    still works.

    Parameters
    ----------
    data_catalog
        Data catalogs keyed by source dataset type
    dataset_filters
        Mapping of facet names to lists of allowed values

    Returns
    -------
    :
        Filtered data catalogs
    """
    filtered: dict[SourceDatasetType, DataCatalog | pd.DataFrame] = {}
    for source_type, catalog in data_catalog.items():
        if isinstance(catalog, DataCatalog):
            df = catalog.to_frame()
            mask = pd.Series(True, index=df.index)
            for facet, values in dataset_filters.items():
                if facet not in df.columns:
                    continue
                mask &= df[facet].isin(values)
            filtered[source_type] = DataCatalog(
                database=catalog.database, adapter=catalog.adapter, df=df[mask]
            )
        else:
            mask = pd.Series(True, index=catalog.index)
            for facet, values in dataset_filters.items():
                if facet not in catalog.columns:
                    continue
                mask &= catalog[facet].isin(values)
            filtered[source_type] = catalog[mask]
    return filtered


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
    data_catalog: dict[SourceDatasetType, DataCatalog]

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
                SourceDatasetType.CMIP6: DataCatalog(database=db, adapter=CMIP6DatasetAdapter(config=config)),
                SourceDatasetType.CMIP7: DataCatalog(database=db, adapter=CMIP7DatasetAdapter()),
                SourceDatasetType.obs4MIPs: DataCatalog(database=db, adapter=Obs4MIPsDatasetAdapter()),
                SourceDatasetType.PMPClimatology: DataCatalog(
                    database=db, adapter=PMPClimatologyDatasetAdapter()
                ),
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
        data_catalog: Mapping[SourceDatasetType, DataCatalog | pd.DataFrame] = self.data_catalog
        if filters and filters.dataset:
            data_catalog = apply_dataset_filters(data_catalog, filters.dataset)

        for provider in self.provider_registry.providers:
            for diagnostic in provider.diagnostics():
                # Filter the diagnostic based on the provided filters
                if not matches_filter(diagnostic, filters):
                    logger.debug(f"Skipping {diagnostic.full_slug()} due to filter")
                    continue
                logger.info(f"Solving {diagnostic.full_slug()}")
                try:
                    yield from solve_executions(data_catalog, diagnostic, provider)
                except InvalidDiagnosticException as e:
                    # Skip diagnostics that don't have matching data
                    logger.debug(f"Skipping {diagnostic.full_slug()}: {e}")
                    continue


DEFAULT_STALE_EXECUTION_AGE_SECONDS = 6 * 60 * 60


def fail_stale_in_progress_executions(
    db: Database,
    *,
    stale_after_seconds: int = DEFAULT_STALE_EXECUTION_AGE_SECONDS,
) -> int:
    """
    Mark abandoned in-progress executions as failed so the next solve can retry them.

    An execution is considered abandoned when it has ``successful=None`` and was
    created longer ago than ``stale_after_seconds``. This commonly happens when
    a worker is killed (OOM, walltime, segfault) before its result-handling
    callback ran, or when the join loop crashed mid-flight.

    The execution group's ``dirty`` flag is left untouched so the existing
    retry logic (``ExecutionGroup.should_run``) picks the work back up.

    Parameters
    ----------
    db
        The database to inspect
    stale_after_seconds
        Minimum age in seconds before an in-progress execution is considered
        abandoned. Defaults to 6 hours, matching the Celery and LocalExecutor
        per-task time limits.

    Returns
    -------
    :
        The number of executions that were marked failed.
    """
    cutoff = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - datetime.timedelta(
        seconds=stale_after_seconds
    )

    with db.session.begin():
        stale = (
            db.session.query(Execution)
            .filter(Execution.successful.is_(None), Execution.created_at < cutoff)
            .all()
        )
        for execution in stale:
            logger.warning(
                f"Marking abandoned execution {execution.id} (group "
                f"{execution.execution_group_id}) as failed; created at {execution.created_at}"
            )
            execution.mark_failed()

    if stale:
        logger.info(f"Marked {len(stale)} stale in-progress executions as failed")
    return len(stale)


def solve_required_executions(  # noqa: PLR0912, PLR0913, PLR0915
    db: Database,
    dry_run: bool = False,
    execute: bool = True,
    solver: ExecutionSolver | None = None,
    config: Config | None = None,
    timeout: int = 60,
    wait: bool = True,
    one_per_provider: bool = False,
    one_per_diagnostic: bool = False,
    filters: SolveFilterOptions | None = None,
    limit: int | None = None,
    rerun_failed: bool = False,
) -> None:
    """
    Solve for executions that require recalculation

    This may trigger a number of additional calculations depending on what data has been ingested
    since the last solve.

    When ``wait`` is True (the default) this blocks until all executions complete,
    copying their outputs to the results directory and ingesting them.
    ``timeout`` bounds that wait; a non-positive ``timeout`` (``<= 0``) waits with no time limit.
    ``wait=False`` queues the executions and returns immediately without collecting their results.

    Raises
    ------
    TimeoutError
        If the executions aren't completed within a positive ``timeout``
    """
    if config is None:
        config = Config.default()
    if solver is None:
        solver = ExecutionSolver.build_from_db(config, db)

    logger.info("Solving for diagnostics that require recalculation...")

    # Reap any executions that were left in-progress by a previous run that
    # crashed, hit walltime, or otherwise lost its result-handling callback.
    # Without this sweep, ``ExecutionGroup.should_run`` keeps returning False
    # for these rows and the group is never retried.
    fail_stale_in_progress_executions(db)

    executor = config.executor.build(config, db)

    if not wait and getattr(executor, "collects_results_on_join", False):
        logger.warning(
            f"--no-wait was requested with the {getattr(executor, 'name', 'configured')!r} executor, "
            f"which only persists results while waiting (during join). "
            f"Queued executions will run, "
            f"but their outputs will be left in the scratch directory and never copied to the results "
            f"directory or ingested into the database. Re-run without --no-wait, or recover existing "
            f"scratch outputs with `ref executions reingest --include-failed`."
        )

    diagnostic_count: dict[str, int] = {}
    provider_count: dict[str, int] = {}
    total_count = 0

    # Prefetch all (provider_slug, diagnostic_slug) -> diagnostic_id mappings
    # so the per-iteration loop body avoids an N+1 SELECT against the
    # diagnostic table.
    with db.session.begin():
        diagnostic_id_by_slug: dict[tuple[str, str], int] = {
            (provider_slug, diagnostic_slug): diagnostic_id
            for diagnostic_id, diagnostic_slug, provider_slug in db.session.query(
                DiagnosticModel.id, DiagnosticModel.slug, ProviderModel.slug
            ).join(DiagnosticModel.provider)
        }

    for potential_execution in solver.solve(filters):
        definition = potential_execution.build_execution_definition(output_root=config.paths.scratch)
        provider_slug = potential_execution.provider.slug
        diagnostic_full_slug = potential_execution.diagnostic.full_slug()

        logger.debug(f"Identified candidate execution {definition.key} for {diagnostic_full_slug}")

        provider_count.setdefault(provider_slug, 0)
        diagnostic_count.setdefault(diagnostic_full_slug, 0)

        # Submission to the executor must happen after the DB transaction commits.
        # Holding a transaction across a Redis send_task / process pool submit can
        # block writes from worker callbacks (e.g. handle_result) and create
        # "database is locked" loops on SQLite, leaving Execution rows stuck with
        # successful=None.
        pending: tuple[ExecutionDefinition, Execution] | None = None
        limit_reached = False

        diagnostic_id = diagnostic_id_by_slug[(provider_slug, potential_execution.diagnostic.slug)]
        diagnostic_version = potential_execution.diagnostic.version

        # Use a transaction to make sure that the models
        # are created correctly before potentially executing out of process
        with db.session.begin():
            # diagnostic_version is part of the lookup key so bumping
            # ``Diagnostic.version`` produces a fresh v2 group instead of reusing the v1 row.
            execution_group, created = db.get_or_create(
                ExecutionGroup,
                key=definition.key,
                diagnostic_id=diagnostic_id,
                diagnostic_version=diagnostic_version,
                defaults={
                    "selectors": potential_execution.selectors,
                    "dirty": True,
                },
            )

            if created:
                logger.info(f"Created new execution group: {potential_execution.execution_slug()!r}")
                db.session.flush()
                recompute_promoted_version(diagnostic_id, db.session)

            # TODO: Move this logic to the solver
            # Check if we should run given the one_per_provider or one_per_diagnostic flags
            one_of_check_failed = (one_per_provider and provider_count.get(provider_slug, 0) > 0) or (
                one_per_diagnostic and diagnostic_count.get(diagnostic_full_slug, 0) > 0
            )

            logger.debug(
                f"one_per_provider={one_per_provider}, one_per_diagnostic={one_per_diagnostic}, "
                f"one_of_check_failed={one_of_check_failed}, diagnostic_count={diagnostic_count}, "
                f"provider_count={provider_count}"
            )

            if not execution_group.should_run(definition.datasets.hash, rerun_failed=rerun_failed):
                continue

            if (one_per_provider or one_per_diagnostic) and one_of_check_failed:
                logger.info(
                    f"Skipping execution due to one-of check: {potential_execution.execution_slug()!r}"
                )
                continue

            if dry_run:
                provider_count[provider_slug] += 1
                diagnostic_count[diagnostic_full_slug] += 1
                total_count += 1
                if limit is not None and total_count >= limit:
                    limit_reached = True
            else:
                logger.info(
                    f"Running new execution for execution group: {potential_execution.execution_slug()!r}"
                )

                execution = Execution(
                    execution_group=execution_group,
                    dataset_hash=definition.datasets.hash,
                    output_fragment=PLACEHOLDER_FRAGMENT,
                    provider_version=potential_execution.provider.version,
                )
                db.session.add(execution)

                fragment = assign_execution_fragment(
                    db.session,
                    execution,
                    provider_slug=provider_slug,
                    diagnostic_slug=potential_execution.diagnostic.slug,
                    selectors=potential_execution.selectors,
                    group_id=execution_group.id,
                )

                # Rebuild the definition so the executor sees the resolved output path.
                definition = attrs.evolve(
                    definition,
                    output_directory=config.paths.scratch.resolve() / pathlib.Path(fragment),
                )

                # Add links to the datasets used in the execution
                execution.register_datasets(db, definition.datasets)

                if execute:
                    # Detach the row before the surrounding ``with begin()`` commits.
                    # Otherwise expire-on-commit marks ``execution.id`` stale,
                    # and the next attribute access inside ``executor.run`` autobegins a fresh transaction,
                    # which collides with the ``with begin()`` at the top of the next loop iteration.
                    # The detached instance keeps its loaded attributes
                    # and is still mergeable for ``SynchronousExecutor``.
                    db.session.expunge(execution)
                    pending = (definition, execution)

                provider_count[provider_slug] += 1
                diagnostic_count[diagnostic_full_slug] += 1
                total_count += 1
                if limit is not None and total_count >= limit:
                    limit_reached = True

        # Transaction has committed; safe to submit to the executor.
        if pending is not None:
            pending_definition, pending_execution = pending
            executor.run(
                definition=pending_definition,
                execution=pending_execution,
            )

        if limit_reached:
            logger.info(f"Reached execution limit of {limit}")
            break

    logger.info("Solve complete")
    logger.info(f"Found {sum(diagnostic_count.values())} new executions")
    for diag, count in diagnostic_count.items():
        logger.info(f"  {diag}: {count} new executions")
    for prov, count in provider_count.items():
        logger.info(f"  {prov}: {count} new executions")

    if wait:
        executor.join(timeout=timeout)
        logger.info("All executions complete")
