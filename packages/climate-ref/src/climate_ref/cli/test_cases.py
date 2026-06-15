"""
Test data management commands for diagnostic development.

These commands are intended for developers working on diagnostics and require
a source checkout of the project with test data directories available.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from climate_ref.cli._git_utils import collect_regression_file_info, get_repo_for_path
from climate_ref.cli._utils import format_size
from climate_ref.config import Config
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    InvalidDiagnosticException,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from climate_ref.datasets import DatasetAdapter
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.testing import TestCase


app = typer.Typer(help=__doc__)


def _build_catalog(dataset_adapter: DatasetAdapter, file_paths: list[Path]) -> pd.DataFrame:
    """
    Parses a list of datasets using a dataset adapter

    Parameters
    ----------
    file_paths
        List of files to build a catalog from

    Returns
    -------
    pd.DataFrame
        DataFrame catalog of datasets with metadata and paths
    """
    import pandas as pd

    # Collect unique parent directories since the adapter scans directories
    parent_dirs = list({fp.parent for fp in file_paths})

    catalog_dfs = []
    for parent_dir in parent_dirs:
        try:
            df = dataset_adapter.find_local_datasets(parent_dir)

            # Filter to only include the files we fetched
            fetched_files = {str(fp) for fp in file_paths}
            df = df[df["path"].isin(fetched_files)]
            if df.empty:
                logger.warning(f"No matching files found in catalog for {parent_dir}")
            catalog_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to parse {parent_dir}: {e}")

    if not catalog_dfs:
        return pd.DataFrame()
    return pd.concat(catalog_dfs, ignore_index=True)


def _solve_test_case(
    diagnostic: Diagnostic,
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
) -> ExecutionDatasetCollection:
    """
    Solve for test case datasets by applying the diagnostic's data requirements.

    Runs the solver to determine which datasets from the catalog
    satisfy the diagnostic's requirements.
    """
    from climate_ref.solver import solve_executions

    executions = list(solve_executions(data_catalog, diagnostic, diagnostic.provider))

    if not executions:
        raise ValueError(f"No valid executions found for diagnostic {diagnostic.slug}")

    return executions[0].datasets


def _fetch_and_build_catalog(
    diag: Diagnostic,
    tc: TestCase,
    *,
    force: bool = False,
) -> tuple[ExecutionDatasetCollection, bool]:
    """
    Fetch test data and build catalog.

    This function:
    1. Fetches ESGF data using ESGFFetcher (files stored in intake-esgf cache)
    2. Uses CMIP6DatasetAdapter to create a data catalog
    3. Solves for datasets using the diagnostic's data requirements
    4. Writes catalog YAML to .catalogs/{provider}/{diagnostic}/{test_case}.yaml
    5. Returns the solved datasets and whether the catalog was written

    By default, the catalog is only written if the content has changed.
    Use `force=True` to always write.

    Parameters
    ----------
    diag
        The diagnostic to fetch data for
    tc
        The test case to fetch data for
    force
        If True, always write the catalog even if unchanged

    Returns
    -------
    :
        Tuple of (datasets, catalog_was_written)
    """
    from climate_ref.datasets import (
        CMIP6DatasetAdapter,
        CMIP7DatasetAdapter,
        Obs4MIPsDatasetAdapter,
        PMPClimatologyDatasetAdapter,
    )
    from climate_ref_core.datasets import SourceDatasetType
    from climate_ref_core.esgf import ESGFFetcher
    from climate_ref_core.testing import TestCasePaths, save_datasets_to_yaml

    fetcher = ESGFFetcher()

    # Fetch all requests - returns DataFrame with metadata + paths
    combined = fetcher.fetch_for_test_case(tc.requests)

    if combined.empty:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Group paths by source type and use adapters to build proper catalog
    data_catalog: dict[SourceDatasetType, pd.DataFrame] = {}

    for source_type, group_df in combined.groupby("source_type"):
        file_paths = [Path(p) for p in group_df["path"].unique().tolist()]

        if source_type == "CMIP6":
            data_catalog[SourceDatasetType.CMIP6] = _build_catalog(CMIP6DatasetAdapter(), file_paths)

        elif source_type == "CMIP7":
            data_catalog[SourceDatasetType.CMIP7] = _build_catalog(CMIP7DatasetAdapter(), file_paths)

        elif source_type == "obs4MIPs":
            data_catalog[SourceDatasetType.obs4MIPs] = _build_catalog(Obs4MIPsDatasetAdapter(), file_paths)

        elif source_type == "PMPClimatology":
            data_catalog[SourceDatasetType.PMPClimatology] = _build_catalog(
                PMPClimatologyDatasetAdapter(), file_paths
            )

    if not data_catalog:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Solve for datasets
    datasets = _solve_test_case(diag, data_catalog)

    # Write catalog YAML to package-local test case directory
    catalog_written = False
    paths = TestCasePaths.from_diagnostic(diag, tc.name)
    if paths:
        paths.create()
        catalog_written = save_datasets_to_yaml(datasets, paths.catalog, force=force)

    return datasets, catalog_written


@app.command(name="fetch")
def fetch_test_data(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Specific provider to fetch data for (e.g., 'esmvaltool', 'ilamb')"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to fetch data for"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to fetch data for"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Show what would be fetched without downloading"),
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(help="Only fetch data for test cases without existing catalogs"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(help="Force overwrite catalog even if unchanged"),
    ] = False,
) -> None:
    """
    Fetch test data from ESGF for running diagnostic tests.

    Downloads full-resolution ESGF data based on diagnostic test_data_spec.
    Use --provider or --diagnostic to limit scope.

    Examples
    --------
        ref test-cases fetch                   # Fetch all test data
        ref test-cases fetch --provider ilamb  # Fetch ILAMB test data only
        ref test-cases fetch --diagnostic ecs  # Fetch ECS diagnostic data
        ref test-cases fetch --only-missing    # Skip test cases with existing catalogs
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.testing import TestCasePaths

    config = ctx.obj.config
    db = ctx.obj.database

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

    # Check if the requested provider exists in the registry
    available_providers = [p.slug for p in registry.providers]
    if provider and provider not in available_providers:
        logger.error(f"Provider '{provider}' is not configured")
        if available_providers:
            logger.error(f"Available providers: {', '.join(sorted(available_providers))}")
        else:
            logger.error("No providers are configured. Check your configuration file.")
        logger.error("To add a provider, update your config file or set REF_DIAGNOSTIC_PROVIDERS")
        raise typer.Exit(code=1)

    # Collect diagnostics to process
    diagnostics_to_process: list[Diagnostic] = []

    for provider_instance in registry.providers:
        if provider and provider_instance.slug != provider:
            continue

        for diag in provider_instance.diagnostics():
            if diagnostic and diag.slug != diagnostic:
                continue
            if diag.test_data_spec is None:
                continue
            diagnostics_to_process.append(diag)

    if not diagnostics_to_process:
        if provider:
            logger.warning(f"No diagnostics with test_data_spec found for provider '{provider}'")
        else:
            logger.warning("No diagnostics with test_data_spec found")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(diagnostics_to_process)} diagnostics with test data specifications")

    if dry_run:  # pragma: no cover
        for diag in diagnostics_to_process:
            logger.info(f"Would fetch data for: {diag.provider.slug}/{diag.slug}")
            if diag.test_data_spec:
                for tc in diag.test_data_spec.test_cases:
                    if test_case and tc.name != test_case:
                        continue
                    # Check if catalog exists when using --only-missing
                    if only_missing:
                        paths = TestCasePaths.from_diagnostic(diag, tc.name)
                        if paths and paths.catalog.exists():
                            logger.info(f"  Test case: {tc.name} - [SKIP: catalog exists]")
                            continue
                    logger.info(f"  Test case: {tc.name} - {tc.description}")
                    if tc.requests:
                        for req in tc.requests:
                            logger.info(f"    Request: {req.slug} ({req.source_type})")
        return

    # Process each diagnostic test case
    for diag in diagnostics_to_process:  # pragma: no cover
        logger.info(f"Fetching data for: {diag.provider.slug}/{diag.slug}")
        if diag.test_data_spec:
            for tc in diag.test_data_spec.test_cases:
                if test_case and tc.name != test_case:
                    continue
                # Skip if catalog exists when using --only-missing
                if only_missing:
                    paths = TestCasePaths.from_diagnostic(diag, tc.name)
                    if paths and paths.catalog.exists():
                        logger.info(f"  Skipping test case: {tc.name} (catalog exists)")
                        continue
                if tc.requests:
                    logger.info(f"  Processing test case: {tc.name}")
                    try:
                        _, catalog_written = _fetch_and_build_catalog(diag, tc, force=force)
                        if not catalog_written:
                            logger.info(f"  Catalog unchanged for {tc.name}")
                    except (DatasetResolutionError, ValueError) as e:
                        logger.warning(f"  Could not build catalog for {tc.name}: {e}")


@app.command(name="list")
def list_cases(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Filter by provider"),
    ] = None,
) -> None:
    """
    List test cases for all diagnostics.

    Shows which test cases are defined for each diagnostic and their descriptions.
    Also shows whether catalog and regression data exist for each test case.
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.testing import TestCasePaths

    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

    # Check if the requested provider exists in the registry
    available_providers = [p.slug for p in registry.providers]
    if provider and provider not in available_providers:
        logger.error(f"Provider '{provider}' is not configured")
        if available_providers:
            logger.error(f"Available providers: {', '.join(sorted(available_providers))}")
        else:
            logger.error("No providers are configured. Check your configuration file.")
        raise typer.Exit(code=1)

    table = Table(title="Test Data Specifications")
    table.add_column("Provider", style="cyan")
    table.add_column("Diagnostic", style="green")
    table.add_column("Test Case", style="yellow")
    table.add_column("Description")
    table.add_column("Requests", justify="right")
    table.add_column("Catalog", justify="center")
    table.add_column("Regression", justify="center")

    for provider_instance in registry.providers:
        if provider and provider_instance.slug != provider:
            continue

        for diag in provider_instance.diagnostics():
            if diag.test_data_spec is None:
                table.add_row(
                    provider_instance.slug,
                    diag.slug,
                    "-",
                    "(no test_data_spec)",
                    "0",
                    "-",
                    "-",
                )
                continue

            for tc in diag.test_data_spec.test_cases:
                num_requests = len(tc.requests) if tc.requests else 0

                # Check if catalog and regression data exist
                paths = TestCasePaths.from_diagnostic(diag, tc.name)
                if paths:
                    catalog_status = "[green]yes[/green]" if paths.catalog.exists() else "[red]no[/red]"
                    regression_status = "[green]yes[/green]" if paths.regression.exists() else "[red]no[/red]"
                else:
                    catalog_status = "[dim]-[/dim]"
                    regression_status = "[dim]-[/dim]"

                table.add_row(
                    provider_instance.slug,
                    diag.slug,
                    tc.name,
                    tc.description,
                    str(num_requests),
                    catalog_status,
                    regression_status,
                )

    console.print(table)


def _print_regression_summary(  # pragma: no cover
    console: Console,
    regression_dir: Path,
    size_threshold_mb: float = 1.0,
) -> None:
    """
    Print a summary of the regression directory with file sizes and git status.

    Parameters
    ----------
    console
        Rich console for output
    regression_dir
        Path to the regression data directory
    size_threshold_mb
        Files larger than this (in MB) will be flagged
    """
    repo = get_repo_for_path(regression_dir)
    repo_root = Path(repo.working_dir) if repo else regression_dir

    threshold_bytes = int(size_threshold_mb * 1024 * 1024)
    file_info = collect_regression_file_info(regression_dir, repo, threshold_bytes)

    if not file_info:
        console.print("[yellow]No files in regression directory[/yellow]")
        return

    total_size = sum(f["size"] for f in file_info)
    large_files = sum(1 for f in file_info if f["is_large"])

    table = Table(title=f"Regression Data: {regression_dir.relative_to(repo_root)}")
    table.add_column("File", style="cyan", no_wrap=False)
    table.add_column("Size", justify="right")
    table.add_column("Git Status", justify="center")

    for info in file_info:
        # Format size with color
        size_str = format_size(info["size"])
        if info["is_large"]:
            size_str = f"[bold red]{size_str}[/bold red]"

        # Get git status with color
        git_status = info["git_status"]
        status_colors = {
            "new": "[green]new[/green]",
            "staged": "[green]staged[/green]",
            "modified": "[yellow]modified[/yellow]",
            "tracked": "[dim]tracked[/dim]",
            "untracked": "[red]untracked[/red]",
            "deleted": "[red]deleted[/red]",
            "unknown": "[dim]?[/dim]",
        }
        git_status_str = status_colors.get(git_status, f"[dim]{git_status}[/dim]")

        table.add_row(info["rel_path"], size_str, git_status_str)

    console.print(table)

    # Summary line
    summary = f"\n[bold]Total:[/bold] {len(file_info)} files, {format_size(total_size)}"
    if large_files > 0:
        summary += f" ([red]{large_files} files > {size_threshold_mb} MB[/red])"
    console.print(summary)


def _run_single_test_case(  # noqa: PLR0911, PLR0912, PLR0915
    config: Config,
    console: Console,
    diag: Diagnostic,
    tc: TestCase,
    output_directory: Path | None,
    force_regen: bool,
    fetch: bool,
    size_threshold: float,
    clean: bool,
) -> bool:
    """
    Run a single test case for a diagnostic.

    Returns True if successful, False otherwise.
    """
    import tempfile

    from climate_ref.testing import TestCaseRunner
    from climate_ref_core.regression.capture import capture_execution
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.testing import (
        TestCasePaths,
        get_catalog_hash,
        load_datasets_from_yaml,
    )

    provider_slug = diag.provider.slug
    diagnostic_slug = diag.slug
    test_case_name = tc.name

    # Resolve datasets: either fetch from ESGF or load from pre-built catalog
    if fetch:
        logger.info(f"Fetching test data for {provider_slug}/{diagnostic_slug}/{test_case_name}")
        try:
            datasets, _ = _fetch_and_build_catalog(diag, tc)
        except (DatasetResolutionError, InvalidDiagnosticException) as e:
            logger.exception(
                f"Failed to fetch data for {provider_slug}/{diagnostic_slug}/{test_case_name}: {e}"
            )
            return False
    else:
        paths = TestCasePaths.from_diagnostic(diag, test_case_name)
        if paths is None:
            logger.error(f"Could not determine test data directory for {provider_slug}/{diagnostic_slug}")
            return False

        if not paths.catalog.exists():
            logger.error(f"No catalog file found for {provider_slug}/{diagnostic_slug}/{test_case_name}")
            logger.error("Run 'ref test-cases fetch' first or use --fetch flag")
            return False

        logger.info(f"Loading catalog from {paths.catalog}")
        datasets = load_datasets_from_yaml(paths.catalog)

    # Create runner and execute
    runner = TestCaseRunner(config=config, datasets=datasets)

    logger.info(f"Running test case {test_case_name!r} for {provider_slug}/{diagnostic_slug}")

    try:
        result = runner.run(diag, test_case_name, output_directory, clean=clean)
    except NoTestDataSpecError:
        logger.error(f"Diagnostic {provider_slug}/{diagnostic_slug} has no test_data_spec")
        return False
    except TestCaseNotFoundError:
        logger.error(f"Test case {test_case_name!r} not found for {provider_slug}/{diagnostic_slug}")
        if diag.test_data_spec:
            logger.error(f"Available test cases: {diag.test_data_spec.case_names}")
        return False
    except DatasetResolutionError as e:
        logger.error(str(e))
        logger.error("Have you run 'ref test-cases fetch' first?")
        return False
    except Exception as e:
        case_id = f"{provider_slug}/{diagnostic_slug}/{test_case_name}"
        logger.error(f"Diagnostic execution failed for {case_id}: {e!s}")
        return False

    if not result.successful:
        logger.error(f"Execution failed: {provider_slug}/{diagnostic_slug}/{test_case_name}")
        return False

    logger.info(f"Execution completed: {provider_slug}/{diagnostic_slug}/{test_case_name}")
    if result.metric_bundle_filename:
        logger.info(f"Metric bundle: {result.to_output_path(result.metric_bundle_filename)}")
    if result.output_bundle_filename:
        logger.info(f"Output bundle: {result.to_output_path(result.output_bundle_filename)}")

    # Handle regression baseline comparison/regeneration
    paths = TestCasePaths.from_diagnostic(diag, test_case_name)

    if paths is None:
        logger.warning("Could not determine test case directory for provider package")
        return True

    if force_regen:
        paths.create()

    if force_regen or not paths.regression.exists():
        if paths.regression.exists():
            shutil.rmtree(paths.regression)
        paths.regression.mkdir(parents=True, exist_ok=True)

        defn = result.definition
        output_dir = defn.output_directory
        fragment = defn.output_fragment()
        scratch_root = output_dir.parent

        # results_base must differ from the scratch root for copy_execution_outputs.
        # Keep it alive only until capture completes (run() does not mint to a store).
        with tempfile.TemporaryDirectory() as results_base:
            committed_digests, native = capture_execution(
                scratch_root,
                Path(results_base),
                fragment,
                result,
                regression_dir=paths.regression,
                output_dir=output_dir,
                test_data_dir=paths.test_data_dir,
            )

        # Refresh the manifest's committed block only; the native block is mint-owned.
        if paths.manifest.exists():
            previous = Manifest.load(paths.manifest)
            manifest = Manifest(
                schema=previous.schema,
                test_case_version=previous.test_case_version,
                committed=committed_digests,
                native=previous.native,
            )
        else:
            manifest = Manifest.seed_v1(committed_digests)
        manifest.dump(paths.manifest)

        logger.info(f"Captured {len(native)} native output file(s) (not minted)")

        # Store catalog hash for change detection
        catalog_hash = get_catalog_hash(paths.catalog)
        if catalog_hash:
            paths.regression_catalog_hash.write_text(catalog_hash)

        logger.info(f"Updated regression data: {paths.regression}")
        _print_regression_summary(console, paths.regression, size_threshold)
    elif paths.regression.exists():
        logger.info(f"Regression data exists at: {paths.regression}")
        logger.info("Use --force-regen to update the baseline")

    return True


@app.command(name="run")
def run_test_case(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to run (e.g., 'global-mean-timeseries')"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to run (e.g., 'default')"),
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option(help="Output directory for execution results"),
    ] = None,
    force_regen: Annotated[
        bool,
        typer.Option(help="Force regeneration of regression baselines"),
    ] = False,
    fetch: Annotated[
        bool,
        typer.Option(help="Fetch test data from ESGF before running"),
    ] = False,
    size_threshold: Annotated[
        float,
        typer.Option(help="Flag files larger than this size in MB (default: 1.0)"),
    ] = 1.0,
    dry_run: Annotated[
        bool,
        typer.Option(help="Show what would be run without executing"),
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(help="Only run test cases without existing regression data"),
    ] = False,
    if_changed: Annotated[
        bool,
        typer.Option(help="Only run if catalog has changed since regression data was generated"),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(help="Delete existing output directory before running"),
    ] = False,
) -> None:
    """
    Run test cases for diagnostics.

    Executes diagnostics using pre-defined datasets from the test_data_spec
    and optionally compares against regression baselines.

    Use --provider to select which provider's diagnostics to run (required).
    Use --diagnostic and --test-case to further narrow the scope.

    Examples
    --------
        ref test-cases run --provider ilamb              # Run all ILAMB test cases
        ref test-cases run --provider example --diagnostic global-mean-timeseries
        ref test-cases run --provider ilamb --test-case default --fetch
        ref test-cases run --provider pmp --only-missing # Skip test cases with regression data
        ref test-cases run --provider pmp --if-changed   # Only run if catalog changed
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.testing import (
        TestCasePaths,
        catalog_changed_since_regression,
    )

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    # Build provider registry
    registry = ProviderRegistry.build_from_config(config, db)

    # Find the provider
    provider_instance = None
    for p in registry.providers:
        if p.slug == provider:
            provider_instance = p
            break

    if provider_instance is None:
        logger.error(f"Provider '{provider}' not found")
        available = [p.slug for p in registry.providers]
        logger.error(f"Available providers: {available}")
        raise typer.Exit(code=1)

    # Collect test cases to run
    test_cases_to_run: list[tuple[Diagnostic, TestCase]] = []
    skipped_cases: list[tuple[Diagnostic, TestCase]] = []

    for diag in provider_instance.diagnostics():
        if diagnostic and diag.slug != diagnostic:
            continue
        if diag.test_data_spec is None:
            continue

        for tc in diag.test_data_spec.test_cases:
            if test_case and tc.name != test_case:
                continue
            # Skip if regression exists when using --only-missing
            paths = TestCasePaths.from_diagnostic(diag, tc.name)
            if only_missing:
                if paths and paths.regression.exists():
                    skipped_cases.append((diag, tc))
                    continue
            # Skip if catalog hasn't changed when using --if-changed
            if if_changed:
                if paths and not catalog_changed_since_regression(paths):
                    skipped_cases.append((diag, tc))
                    continue
            test_cases_to_run.append((diag, tc))

    if not test_cases_to_run:
        logger.warning(f"No test cases found for provider '{provider}'")
        if diagnostic:
            logger.warning(f"  with diagnostic filter: {diagnostic}")
        if test_case:
            logger.warning(f"  with test case filter: {test_case}")
        if only_missing and skipped_cases:
            logger.info(f"  ({len(skipped_cases)} test case(s) skipped due to --only-missing)")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(test_cases_to_run)} test case(s) to run")
    if skipped_cases:
        logger.info(f"Skipping {len(skipped_cases)} test case(s) with existing regression data")

    if dry_run:  # pragma: no cover
        table = Table(title="Test Cases to Run")
        table.add_column("Provider", style="cyan")
        table.add_column("Diagnostic", style="green")
        table.add_column("Test Case", style="yellow")
        table.add_column("Description")
        table.add_column("Status", justify="center")

        for diag, tc in test_cases_to_run:
            table.add_row(provider, diag.slug, tc.name, tc.description, "[green]will run[/green]")

        for diag, tc in skipped_cases:
            table.add_row(provider, diag.slug, tc.name, tc.description, "[dim]skip (regression exists)[/dim]")

        console.print(table)
        return

    # Run each test case
    successes = 0
    failures = 0
    failed_cases: list[str] = []

    for diag, tc in test_cases_to_run:
        success = _run_single_test_case(
            config=config,
            console=console,
            diag=diag,
            tc=tc,
            output_directory=output_directory,
            force_regen=force_regen,
            fetch=fetch,
            size_threshold=size_threshold,
            clean=clean,
        )
        if success:
            successes += 1
        else:
            failures += 1
            failed_cases.append(f"{provider}/{diag.slug}/{tc.name}")

    # Print summary
    console.print()
    if failures == 0:
        console.print(f"[green]All {successes} test case(s) passed[/green]")
    else:
        console.print(f"[yellow]Results: {successes} passed, {failures} failed[/yellow]")
        console.print("[red]Failed test cases:[/red]")
        for case in failed_cases:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)


def _iter_test_cases(
    registry: ProviderRegistry,
    *,
    provider: str | None = None,
    diagnostic: str | None = None,
    test_case: str | None = None,
) -> Iterator[tuple[Diagnostic, TestCase]]:
    """
    Yield ``(diagnostic, test_case)`` pairs from the registry, applying filters.

    Parameters
    ----------
    registry
        The provider registry to enumerate.
    provider
        Optional provider slug filter.
    diagnostic
        Optional diagnostic slug filter.
    test_case
        Optional test case name filter.

    Yields
    ------
    :
        Matching ``(diagnostic, test_case)`` pairs.
    """
    for provider_instance in registry.providers:
        if provider and provider_instance.slug != provider:
            continue
        for diag in provider_instance.diagnostics():
            if diagnostic and diag.slug != diagnostic:
                continue
            if diag.test_data_spec is None:
                continue
            for tc in diag.test_data_spec.test_cases:
                if test_case and tc.name != test_case:
                    continue
                yield diag, tc


@app.command(name="sync")
def sync_native(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Limit sync to a single provider slug"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Limit sync to a single diagnostic slug"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Limit sync to a single test case name"),
    ] = None,
) -> None:
    """
    Fetch native baseline blobs referenced by committed manifests into the store cache.

    Reads each committed ``manifest.json``'s ``native`` block
    and ensures every referenced blob is present in the read store (public, credential-free).
    Blobs already cached are skipped (idempotent).
    A referenced digest the store cannot serve is a hard failure.

    Examples
    --------
        ref test-cases sync                  # Sync all providers
        ref test-cases sync --provider ilamb # Sync a single provider
    """
    import tempfile

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.regression.store import build_native_store
    from climate_ref_core.testing import TestCasePaths

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    store = build_native_store(config.native_store, writable=False)

    # When a specific case is named, a missing manifest is a hard failure.
    named = bool(diagnostic or test_case)

    fetched = 0
    skipped = 0
    failures: list[str] = []

    for diag, tc in _iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case):
        case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        if paths is None or not paths.manifest.exists():
            if named:
                logger.error(f"No manifest.json for {case_id}; run `ref test-cases mint` first")
                failures.append(case_id)
            continue
        manifest = Manifest.load(paths.manifest)
        for relpath, entry in manifest.native.items():
            if store.has(entry.sha256):
                skipped += 1
                continue
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    store.fetch(entry.sha256, Path(tmp) / "blob")
                except Exception as exc:
                    failures.append(f"{case_id}: cannot serve native blob {entry.sha256} ({relpath}): {exc}")
                    continue
            fetched += 1

    console.print(f"[green]Synced native blobs:[/green] {fetched} fetched, {skipped} already cached")
    if failures:
        console.print("[red]Failed to fetch referenced native blobs:[/red]")
        for failure in failures:
            console.print(f"  - {failure}")
        raise typer.Exit(code=1)


@app.command(name="replay")
def replay_test_case(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to replay"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to replay"),
    ] = None,
) -> None:
    """
    Replay committed baselines from native blobs and compare to the committed bundle.

    Materialises the committed manifest's native blobs (public, credential-free)
    into a fresh output directory at their stored relative paths, re-runs ``build_execution_result``,
    and compares the regenerated committed bundle to the in-repo copy using the tolerant content comparator.

    Exits non-zero on drift.

    Examples
    --------
        ref test-cases replay --provider example
        ref test-cases replay --provider example --diagnostic global-mean-timeseries
    """
    import tempfile

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.diagnostics import ExecutionDefinition
    from climate_ref_core.output_files import from_placeholders
    from climate_ref_core.regression import (
        COMMITTED_BUNDLE_FILES,
        Manifest,
        Tolerance,
        assert_bundle_regression,
        materialise_native,
        verify_committed_integrity,
    )
    from climate_ref_core.regression.store import build_native_store
    from climate_ref_core.testing import TestCasePaths, load_datasets_from_yaml

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    store = build_native_store(config.native_store, writable=False)

    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))
    if not cases:
        logger.warning(f"No test cases found for provider {provider!r}")
        raise typer.Exit(code=0)

    # When a specific case is named, a missing manifest/catalog is a hard failure.
    named = bool(diagnostic or test_case)

    successes = 0
    failures: list[str] = []

    for diag, tc in cases:
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        case_id = f"{provider}/{diag.slug}/{tc.name}"
        if paths is None:
            logger.warning(f"Could not determine test case directory for {case_id}")
            continue
        if not paths.manifest.exists():
            message = f"No manifest.json for {case_id}; run `ref test-cases mint` first"
            if named:
                logger.error(message)
                failures.append(case_id)
            else:
                logger.warning(message)
            continue
        if not paths.catalog.exists():
            logger.error(f"No catalog file for {case_id}; run `ref test-cases fetch` first")
            failures.append(case_id)
            continue

        manifest = Manifest.load(paths.manifest)

        # The byte-exact digest check is advisory that the committed baseline is not bitwise identical.
        # The tolerant bundle comparison below may still find them equivalent within tolerance.
        mismatches = verify_committed_integrity(manifest, paths.regression)
        if mismatches:
            logger.warning(
                f"{case_id}: committed baseline differs from the digests recorded in {paths.manifest}"
            )
            for mismatch in mismatches:
                logger.warning(f"  - {mismatch}")

        if not manifest.native:
            logger.error(
                f"{case_id}: manifest has no native baselines — not yet minted. "
                "Run `ref test-cases mint` first."
            )
            failures.append(case_id)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            output_dir = tmp_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            try:
                materialise_native(manifest.native, store, output_dir)
            except Exception as exc:
                logger.error(f"{case_id}: failed to materialise native blobs: {exc}")
                failures.append(case_id)
                continue

            # Expand placeholders so the re-run reads concrete paths (mirrors the validator).
            from_placeholders(output_dir, output_dir=output_dir, test_data_dir=paths.test_data_dir)

            datasets = load_datasets_from_yaml(paths.catalog)
            definition = ExecutionDefinition(
                diagnostic=diag,
                key=f"test-{tc.name}",
                datasets=datasets,
                output_directory=output_dir,
                root_directory=tmp_dir,
            )

            try:
                result = diag.build_execution_result(definition)
            except Exception as exc:
                logger.error(f"{case_id}: build_execution_result failed during replay: {exc}")
                failures.append(case_id)
                continue

            replacements = {
                str(output_dir): "<OUTPUT_DIR>",
                str(paths.test_data_dir): "<TEST_DATA_DIR>",
            }
            # The comparator silently skips a bundle file with no committed copy,
            compared = [f for f in COMMITTED_BUNDLE_FILES if (paths.regression / f).exists()]
            try:
                for filename in compared:
                    assert_bundle_regression(
                        paths.regression / filename,
                        result.to_output_path(filename),
                        slug=diag.slug,
                        tol=Tolerance(),
                        replacements=replacements,
                    )
            except AssertionError as exc:
                logger.error(f"{case_id}: replay drift detected:\n{exc}")
                failures.append(case_id)
                continue

        successes += 1
        if mismatches:
            # The byte-level warning above was reconciled by the tolerant comparison.
            logger.info(
                f"Replay reconciled committed bundle: {case_id} "
                f"({len(manifest.native)} native file(s) materialised, "
                f"{len(compared)} bundle file(s) equivalent within tolerance)"
            )
        else:
            logger.info(
                f"Replay matched committed bundle: {case_id} "
                f"({len(manifest.native)} native file(s) materialised, "
                f"{len(compared)} bundle file(s) compared)"
            )

    console.print()
    if failures:
        console.print(f"[yellow]Replay: {successes} passed, {len(failures)} failed[/yellow]")
        console.print("[red]Failed replays:[/red]")
        for case in failures:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)
    console.print(f"[green]All {successes} replay(s) matched the committed bundle[/green]")


@app.command(name="mint")
def mint_native(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to mint"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to mint"),
    ] = None,
    bump_version: Annotated[
        bool,
        typer.Option(help="Increment test_case_version when authoring the manifest"),
    ] = False,
) -> None:
    """
    Mint canonical native baselines

    Runs each test case, stores its native snapshot in the writable store,
    and authors the committed ``manifest.json``'s ``native`` block.

    This requires write credentials and is generally run by the CI.

    Examples
    --------
        ref test-cases mint --provider example
        ref test-cases mint --provider example --bump-version
    """
    import tempfile

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref.testing import TestCaseRunner
    from climate_ref_core.regression.capture import capture_execution
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.regression.store import build_native_store
    from climate_ref_core.testing import TestCasePaths, get_catalog_hash, load_datasets_from_yaml

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    try:
        store = build_native_store(config.native_store, writable=True)
    except NotImplementedError as exc:
        logger.error(
            "Cannot mint: no writable native store is configured. "
            "Set REF_NATIVE_STORE_URL to a writable location\n"
            f"(a local file:// path for development): {exc}"
        )
        raise typer.Exit(code=1) from exc

    registry = ProviderRegistry.build_from_config(config, db)
    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))
    if not cases:
        logger.warning(f"No test cases found for provider {provider!r}")
        raise typer.Exit(code=0)

    minted = 0
    failures: list[str] = []

    for diag, tc in cases:
        case_id = f"{provider}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        if paths is None:
            logger.warning(f"Could not determine test case directory for {case_id}")
            continue
        if not paths.catalog.exists():
            logger.error(f"No catalog file for {case_id}; run `ref test-cases fetch` first")
            failures.append(case_id)
            continue

        datasets = load_datasets_from_yaml(paths.catalog)
        runner = TestCaseRunner(config=config, datasets=datasets)
        try:
            result = runner.run(diag, tc.name, None, clean=True)
        except Exception as exc:
            logger.error(f"{case_id}: execution failed during mint: {exc}")
            failures.append(case_id)
            continue
        if not result.successful:
            logger.error(f"{case_id}: execution was not successful")
            failures.append(case_id)
            continue

        defn = result.definition
        output_dir = defn.output_directory
        fragment = defn.output_fragment()
        scratch_root = output_dir.parent

        # Any existing results are wiped to make sure we have a clean slate
        paths.create()
        if paths.regression.exists():
            shutil.rmtree(paths.regression)
        paths.regression.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as results_base:
            results_dir = Path(results_base)
            committed_digests, native = capture_execution(
                scratch_root,
                results_dir,
                fragment,
                result,
                regression_dir=paths.regression,
                output_dir=output_dir,
                test_data_dir=paths.test_data_dir,
            )

            base_dir = results_dir / fragment
            mismatch = False
            for relpath, entry in native.items():
                digest = store.put(base_dir / relpath)
                if digest != entry.sha256:
                    logger.error(
                        f"{case_id}: digest mismatch for {relpath} (store={digest}, captured={entry.sha256})"
                    )
                    mismatch = True
            if mismatch:
                failures.append(case_id)
                continue

        # Author the committed manifest: native block written ONLY here.
        if paths.manifest.exists():
            previous = Manifest.load(paths.manifest)
            version = previous.test_case_version + 1 if bump_version else previous.test_case_version
        else:
            version = 1
        manifest = Manifest(
            schema=1,
            test_case_version=version,
            committed=committed_digests,
            native=native,
        )
        manifest.dump(paths.manifest)

        catalog_hash = get_catalog_hash(paths.catalog)
        if catalog_hash:
            paths.regression_catalog_hash.write_text(catalog_hash)

        minted += 1
        logger.info(
            f"Minted native baseline: {case_id} "
            f"({len(native)} native file(s), {len(committed_digests)} committed file(s), "
            f"test_case_version={version})"
        )

    console.print()
    if failures:
        console.print(f"[yellow]Mint: {minted} minted, {len(failures)} failed[/yellow]")
        console.print("[red]Failed mints:[/red]")
        for case in failures:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)
    console.print(f"[green]Minted {minted} native baseline(s)[/green]")


def _provider_source_root(diag: Diagnostic, repo_root: Path) -> str | None:
    """
    Return the diagnostic's provider package source directory, relative to the repo root.

    Used to decide whether a changed file touches the diagnostic's extraction code.
    The returned path is POSIX-style so it can be compared against
    ``git diff --name-only`` output.

    Parameters
    ----------
    diag
        The diagnostic whose provider source directory is wanted.
    repo_root
        The repository working-tree root.

    Returns
    -------
    :
        The provider package source directory relative to ``repo_root``, or ``None``
        if it cannot be located or lies outside the repository.
    """
    import importlib.util

    top_package = type(diag).__module__.split(".")[0]
    try:
        spec = importlib.util.find_spec(top_package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    try:
        return package_dir.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _core_regression_root(repo_root: Path) -> str | None:
    """
    Return the core regression package directory, relative to the repo root.

    A change anywhere under this directory affects replay/execute for every test case,
    so it is treated as an extraction change. Derived from the installed package
    location (rather than a hardcoded path) so it survives a package move.

    Parameters
    ----------
    repo_root
        The repository working-tree root.

    Returns
    -------
    :
        The ``climate_ref_core.regression`` directory relative to ``repo_root``, or
        ``None`` if it lies outside the repository (e.g. an installed wheel).
    """
    import climate_ref_core.regression as regression_pkg

    package_dir = Path(regression_pkg.__file__).resolve().parent
    try:
        return package_dir.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


@app.command(name="ci-gate")
def ci_gate(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    base: Annotated[
        str,
        typer.Option(help="Git ref to compare against (the PR base branch)"),
    ] = "origin/main",
    provider: Annotated[
        str | None,
        typer.Option(help="Limit the gate to a single provider slug"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Limit the gate to a single diagnostic slug"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Limit the gate to a single test case name"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the per-case decisions as JSON on stdout"),
    ] = False,
) -> None:
    """
    Decide how CI should verify each test case's regression baseline.

    Compares each committed ``manifest.json`` to its counterpart on the base branch
    and reports the action CI should take per case: ``replay`` (cheap, against the
    cached native baseline), ``execute`` (full re-run, when ``test_case_version`` was
    bumped), ``skip`` (nothing relevant changed), or ``fail`` (an unauthorised
    baseline change). Exits non-zero if any case is gated ``fail``.

    The ``--json`` output is intended for CI to dispatch ``replay``/``run`` jobs.

    Examples
    --------
        ref test-cases ci-gate                       # Gate all cases against origin/main
        ref test-cases ci-gate --base origin/develop
        ref test-cases ci-gate --provider example --json
    """
    import json as _json

    from git import GitCommandError

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.gate import Action, decide_coupling, paths_under
    from climate_ref_core.regression.manifest import Manifest, compute_committed_digests
    from climate_ref_core.testing import TestCasePaths

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    repo = get_repo_for_path(Path.cwd())
    if repo is None or repo.working_dir is None:
        logger.error("ci-gate must be run inside a git repository")
        raise typer.Exit(code=1)
    repo_root = Path(repo.working_dir)

    # Resolve the set of files changed on this branch relative to the base ref.
    # `base...HEAD` diffs against the merge-base, so unrelated base-branch churn
    # is excluded.
    try:
        diff_output = repo.git.diff("--name-only", f"{base}...HEAD")
    except GitCommandError as exc:
        logger.error(f"Could not diff against base ref {base!r}: {exc}")
        raise typer.Exit(code=1) from exc
    changed_files = [line.strip() for line in diff_output.splitlines() if line.strip()]

    # The core regression machinery affects every replay/execute, so a change there
    # counts as an extraction change for all cases. Extraction-change detection is
    # deliberately coarse: any change under a diagnostic's provider package (see
    # `_provider_source_root`) or under the core regression package counts for every
    # case in that provider. This errs toward REPLAY (cheap, credential-free), never
    # away from it.
    core_root = _core_regression_root(repo_root)
    core_changed = paths_under(changed_files, [core_root] if core_root else [])

    registry = ProviderRegistry.build_from_config(config, db)

    decisions: list[dict[str, str]] = []
    has_failure = False

    def record(case: str, action: Action, reason: str) -> None:
        nonlocal has_failure
        if action is Action.FAIL:
            has_failure = True
        decisions.append({"case": case, "action": action.value, "reason": reason})

    for diag, tc in _iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case):
        case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)

        # A corrupt manifest authored in this change is a hard failure for that case,
        # not a crash for the whole gate.
        manifest: Manifest | None = None
        if paths is not None and paths.manifest.exists():
            try:
                manifest = Manifest.load(paths.manifest)
            except ValueError as exc:
                logger.error(f"{case_id}: invalid manifest.json: {exc}")
                record(case_id, Action.FAIL, f"invalid manifest.json: {exc}")
                continue

        base_manifest: Manifest | None = None
        if paths is not None:
            try:
                rel_manifest = paths.manifest.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                rel_manifest = None
            if rel_manifest is not None:
                try:
                    base_text = repo.git.show(f"{base}:{rel_manifest}")
                except GitCommandError:
                    base_manifest = None
                else:
                    # A corrupt manifest on the base branch can't be compared against;
                    # fall back to seeding (REPLAY) rather than aborting the gate.
                    try:
                        base_manifest = Manifest.loads(base_text, source=f"{base}:{rel_manifest}")
                    except ValueError as exc:
                        logger.warning(
                            f"{case_id}: base manifest at {base}:{rel_manifest} is invalid "
                            f"({exc}); treating as newly added"
                        )
                        base_manifest = None

        source_root = _provider_source_root(diag, repo_root)
        extraction_roots = [r for r in (source_root,) if r]
        extraction_changed = core_changed or paths_under(changed_files, extraction_roots)

        # Verify the committed bundle on disk still matches the manifest digests.
        # A drift (edited/added/removed committed file without regenerating the
        # manifest) must fail closed rather than slip through as SKIP.
        committed_integrity_ok = True
        if manifest is not None and paths is not None:
            committed_integrity_ok = compute_committed_digests(paths.regression) == manifest.committed

        decision = decide_coupling(
            manifest,
            base_manifest,
            extraction_changed=extraction_changed,
            committed_integrity_ok=committed_integrity_ok,
        )
        record(case_id, decision.action, decision.reason)

    if output_json:
        console.print_json(_json.dumps(decisions))
    else:
        table = Table(title=f"CI coupling gate (base: {base})")
        table.add_column("Test case", style="cyan", no_wrap=True)
        table.add_column("Action")
        table.add_column("Reason")
        style_for = {
            Action.FAIL.value: "red",
            Action.EXECUTE.value: "yellow",
            Action.REPLAY.value: "green",
            Action.SKIP.value: "dim",
        }
        for entry in decisions:
            style = style_for[entry["action"]]
            table.add_row(entry["case"], f"[{style}]{entry['action']}[/{style}]", entry["reason"])
        console.print(table)

    if has_failure:
        raise typer.Exit(code=1)
